import os
import re
from fpdf import FPDF
from dotenv import load_dotenv
import gradio as gr
import google.generativeai as genai
import logging

# LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# ----- CONFIGURATION -----
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate essential environment variable
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable. Please set it.")

# Gemini setup (will be used for both vision and text)
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Gemini API configured successfully.")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    raise # Exit if Gemini API cannot be configured

# Use Gemini 2.5 Flash for both vision and text evaluation
GEMINI_MODEL_NAME_VISION = "gemini-2.5-flash" # Use 2.5 Flash for multimodal input
GEMINI_MODEL_NAME_LLM = "gemini-2.5-flash"   # Use 2.5 Flash for text-only evaluation

def gemini_vision_text(image_path):
    """
    Extracts text from an image using Gemini 2.5 Flash's multimodal capabilities.
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME_VISION)
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()

        # Upload the file to Gemini's internal storage for generation
        uploaded_file = genai.upload_file(image_path)
        logger.info(f"Uploaded file for Gemini Vision: {image_path}")

        response = model.generate_content([
            uploaded_file, # Direct file object as input
            "Extract all the text from this image. Provide the text comprehensively and accurately, maintaining original formatting as much as possible for readability."
        ])
        
        # Clean up the uploaded file to manage storage
        genai.delete_file(uploaded_file.name)
        logger.info(f"Deleted uploaded file: {uploaded_file.name}")

        if response.text:
            return response.text.strip()
        else:
            logger.warning(f"Gemini Vision ({GEMINI_MODEL_NAME_VISION}) returned no text for {os.path.basename(image_path)}. Parts: {response.parts}")
            return ""
    except Exception as e:
        logger.error(f"Error in gemini_vision_text for {image_path}: {e}")
        # Ensure the uploaded file is deleted even if an error occurs
        if 'uploaded_file' in locals() and uploaded_file.name:
            try:
                genai.delete_file(uploaded_file.name)
                logger.info(f"Deleted uploaded file after error: {uploaded_file.name}")
            except Exception as delete_e:
                logger.error(f"Error deleting uploaded file {uploaded_file.name}: {delete_e}")
        raise

# LLM Setup
# Refined prompt for more structured output, making parsing easier
prompt_template = ChatPromptTemplate.from_template("""
You are an expert educator tasked with evaluating a student's answer.
Analyze the student's response thoroughly and provide constructive feedback.
Finally, assign marks based on the correct answer and max marks.

Question:
{question}

Correct Answer:
{correct_answer}

Max Marks: {max_marks}

Student Answer:
{student_answer}

Provide your feedback and marks in the following structured format, ensuring the "Marks:" line is clearly parseable.

Feedback: [Your detailed feedback here, explaining strengths and weaknesses.]
Marks: [Score]/[Max Marks]
""")

def get_llm():
    """Returns an instance of Gemini 2.5 Flash for LLM evaluation."""
    try:
        # Use ChatVertexAI for LangChain integration with Gemini
        # Ensure your Google Cloud project is configured for Vertex AI
        return ChatVertexAI(model=GEMINI_MODEL_NAME_LLM, temperature=0)
    except Exception as e:
        logger.error(f"Error getting LLM {GEMINI_MODEL_NAME_LLM}: {e}")
        raise

# ----- Text Extraction (now exclusively Gemini 2.5 Flash for images) ----
def extract_text(file):
    """
    Extracts text from an image file using Gemini 2.5 Flash.
    This version only supports image files.
    """
    file_path = file.name # Gradio passes a NamedTemporaryFile object, use .name for path
    ext = os.path.splitext(file_path)[-1].lower()

    if ext in [".jpg", ".jpeg", ".png", ".gif", ".tiff", ".tif", ".webp"]:
        logger.info(f"Extracting text from image using {GEMINI_MODEL_NAME_VISION}: {os.path.basename(file_path)}")
        return gemini_vision_text(file_path)
    else:
        logger.error(f"Unsupported file type: {ext} for {os.path.basename(file_path)}. Only image formats are supported in this mode.")
        raise ValueError(f"Unsupported file type: {ext}. Only common image formats are supported for Gemini Vision.")

# ----- QA Parsing (remains the same) -----
def extract_qas(text):
    """
    Extracts questions and answers from the given text.
    Improved robustness for various numbering and 'Answer' indicators.
    Assumes a structure like: Q[num]. [Question Text] Answer: [Answer Text]
    """
    qas = []
    # Pattern to find 'Q' followed by number, optionally with a period and space
    # and then capture everything until 'Answer:' or the next 'Q' or end of string.
    # Case-insensitive for 'Answer:'
    q_blocks = re.split(r"(?i)(Q\d+\.?\s*)", text) # Split by 'Q' followed by number

    for i in range(1, len(q_blocks), 2):
        q_label = q_blocks[i].strip() # e.g., "Q1."
        block_content = q_blocks[i+1].strip()

        answer_match = re.search(r"(?i)(Answer|Ans|Solution):\s*(.*)", block_content, re.DOTALL)
        if answer_match:
            question = block_content[:answer_match.start()].strip()
            answer = answer_match.group(2).strip()
            question = re.sub(r"Q\d+\.?\s*", "", question, count=1).strip()
            if question and answer:
                qas.append((question, answer))
            else:
                logger.warning(f"Skipping malformed QA block (empty question or answer) for {q_label}: '{block_content[:50]}...'")
        else:
            logger.warning(f"Could not find answer separator for {q_label}: '{block_content[:50]}...'")
            question = re.sub(r"Q\d+\.?\s*", "", block_content, count=1).strip()
            if question:
                 qas.append((question, ""))

    return qas

# ----- Evaluation (LLM usage simplified to Gemini 2.5 Flash) -----
def evaluate_with_model(llm, q, correct, student, max_marks):
    """
    Evaluates student answer using LLM and extracts score and feedback.
    Improved parsing of LLM output.
    """
    prompt = prompt_template.format_messages(
        question=q,
        correct_answer=correct,
        student_answer=student,
        max_marks=max_marks
    )
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        logger.debug(f"LLM Raw Response: {text[:500]}...") # Log part of the raw response

        feedback_match = re.search(r"(?i)Feedback:\s*(.*?)(?=Marks:|$)", text, re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else "No feedback provided by LLM."

        score_match = re.search(r"(?i)Marks:\s*(\d+)/(\d+)", text)
        if score_match:
            score = int(score_match.group(1))
            total_possible = int(score_match.group(2))
            if total_possible != max_marks:
                logger.warning(f"LLM returned max marks ({total_possible}) different from expected ({max_marks}) for Q: {q[:30]}. Using LLM's score.")
            return score, feedback
        else:
            logger.warning(f"Could not parse exact 'Marks: X/Y' from LLM response for Q: {q[:30]}. Response: {text[:200]}")
            single_digit_score = re.search(r"Score:\s*(\d+)", text) or re.search(r"(\d+)/", text)
            if single_digit_score:
                logger.info(f"Falling back to partial score parsing for Q: {q[:30]}")
                return int(single_digit_score.group(1)), feedback
            return 0, feedback + "\n[Could not parse score from LLM response]"
    except Exception as e:
        logger.error(f"Error evaluating Q: {q[:50]} with LLM: {e}")
        return 0, f"Error during evaluation: {e}"

def create_pdf(text, output_path="result.pdf"):
    """
    Creates a PDF from the given text.
    Handles basic Latin-1 encoding for FPDF.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    try:
        safe_text = text.encode("latin-1", "ignore").decode("latin-1")
    except Exception as e:
        logger.warning(f"Could not encode text for PDF: {e}. Attempting direct write (might fail for special chars).")
        safe_text = text

    lines = safe_text.split("\n")
    for line in lines:
        try:
            pdf.multi_cell(0, 10, line)
        except Exception as e:
            logger.error(f"FPDF multi_cell error for line: '{line[:50]}...' Error: {e}")
            pdf.multi_cell(0, 10, "[Error rendering line]")
            continue
    pdf.output(output_path)
    return output_path

# ----- Main -----
def evaluate_app(correct_file, student_file):
    """Main function to orchestrate the evaluation process using Gemini 2.5 Flash exclusively."""
    logs = []
    def log(message):
        logs.append(message)
        logger.info(message)
        return message

    log("--- Starting Evaluation with Gemini 2.5 Flash ---")

    try:
        # Step 1: Extract text from files using Gemini 2.5 Flash Vision
        log(f"Extracting text from Correct Answer File ({os.path.basename(correct_file.name)}) using {GEMINI_MODEL_NAME_VISION}...")
        correct_text = extract_text(correct_file)
        if not correct_text.strip():
            return "Error", None, log("❌ Correct Answer file extraction resulted in empty text. Please check the image content.")

        log(f"Extracting text from Student Answer File ({os.path.basename(student_file.name)}) using {GEMINI_MODEL_NAME_VISION}...")
        student_text = extract_text(student_file)
        if not student_text.strip():
            return "Error", None, log("❌ Student Answer file extraction resulted in empty text. Please check the image content.")

        # Step 2: Parse QAs
        log("Parsing Questions and Answers from extracted text...")
        correct_qas = extract_qas(correct_text)
        student_qas = extract_qas(student_text)

        if not correct_qas:
            return "Error", None, log("❌ No questions found in the Correct Answer file. Ensure 'Q#' and 'Answer:' format is used.")
        if not student_qas:
            log("⚠️ No answers found in the Student Answer file. Proceeding with evaluation, but student answers will be empty.")
        
        log(f"✅ Detected {len(correct_qas)} questions in Correct Answer file.")
        log(f"✅ Detected {len(student_qas)} student answer blocks in Student Answer file.")

        # Step 3: Initialize LLM (Gemini 2.5 Flash)
        log(f"Initializing LLM: {GEMINI_MODEL_NAME_LLM}...")
        llm = get_llm()
        log("✅ LLM initialized.")

        # Step 4: Evaluate each QA pair
        results = []
        total_score_obtained = 0
        total_possible_score = len(correct_qas) * 5 # Assuming 5 max marks per question

        for i, (q, correct_ans) in enumerate(correct_qas):
            student_ans = ""
            if i < len(student_qas):
                # We only need the answer part from student_qas tuple
                _, student_ans = student_qas[i]
            else:
                log(f"⚠️ No corresponding student answer found for Q{i+1}. Treating student answer as empty.")

            log(f"🤖 Evaluating Q{i+1} with {GEMINI_MODEL_NAME_LLM}...")
            score, feedback = evaluate_with_model(llm, q, correct_ans, student_ans, 5) # Pass 5 as max_marks
            results.append(f"Q{i+1}: {q}\nCorrect Answer: {correct_ans}\nStudent Answer: {student_ans}\nMarks: {score}/5\nFeedback: {feedback}\n")
            total_score_obtained += score

        # Step 5: Generate Final Report
        final_text = (
            f"--- Evaluation Report ({GEMINI_MODEL_NAME_LLM}) ---\n"
            f"Overall Score: {total_score_obtained}/{total_possible_score}\n\n"
            + "\n".join(results)
        )
        log("Generating PDF report...")
        pdf_path = create_pdf(final_text)
        log(f"✅ PDF report created at: {pdf_path}")

        log("--- Evaluation Complete ---")
        return final_text, pdf_path, "\n".join(logs)

    except Exception as e:
        error_message = f"An unexpected error occurred during evaluation: {e}"
        logger.exception(error_message)
        return "Error", None, "\n".join(logs + [f"❌ {error_message}"])


# ----- UI -----
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Gemini 2.5 Flash Student Answer Evaluator")
    gr.Markdown("Upload images of the correct answer key and the student's answer sheet for evaluation.")
    gr.Markdown(f"**Using {GEMINI_MODEL_NAME_LLM} for both OCR (Image to Text) and LLM Evaluation.**")

    with gr.Row():
        # Only image file types allowed now
        image_file_types = [".jpg", ".jpeg", ".png", ".gif", ".tiff", ".tif", ".webp"]
        correct_file = gr.File(label="✅ Correct QA Image File", file_types=image_file_types)
        student_file = gr.File(label="👨‍🎓 Student Answer Image File", file_types=image_file_types)

    submit = gr.Button("🚀 Evaluate")
    result_output = gr.Textbox(label="📊 Evaluation Result", lines=25, interactive=False)
    log_output = gr.Textbox(label="📜 Logs", lines=10, interactive=False)
    pdf_download = gr.File(label="⬇️ Download PDF Result", file_count="single", interactive=False)

    # Removed model_choice and ocr_choice as they are now hardcoded to Gemini 2.5 Flash
    submit.click(fn=evaluate_app, inputs=[correct_file, student_file], outputs=[result_output, pdf_download, log_output])

if __name__ == "__main__":
    demo.launch(debug=True, share=False)