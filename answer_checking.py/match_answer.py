import os
import re
import fitz  # PyMuPDF
from fpdf import FPDF
from dotenv import load_dotenv
import gradio as gr
from google.cloud import documentai_v1 as documentai
import google.generativeai as genai
import base64

# LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_google_vertexai import ChatVertexAI

load_dotenv()

# ----- CONFIGURATION -----
PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
PROCESSOR_ID = os.getenv("DOCUMENT_AI_PROCESSOR_ID")
LOCATION = "us"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini Vision setup
genai.configure(api_key=GOOGLE_API_KEY)

def gemini_vision_text(image_path):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        uploaded = genai.upload_file(image_path)
        response = model.generate_content([
            uploaded,
            "Extract all the text from this image clearly."
        ])
        genai.delete_file(uploaded.name)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini Vision OCR failed: {e}]"

# LLM Setup
prompt_template = ChatPromptTemplate.from_template("""
Evaluate this student's answer:

Question:
{question}

Correct Answer:
{correct_answer}

Max Marks: {max_marks}

Student Answer:
{student_answer}

Give marks and feedback.
""")

def get_llm(name: str):
    if name == "DeepSeek-R1":
        return ChatOllama(model="deepseek-r1", temperature=0)
    elif name == "LLaMA3":
        return ChatOllama(model="llama3", temperature=0)
    elif name == "Gemini-1.5":
        return ChatVertexAI(model="gemini-pro", temperature=0)
    else:
        raise ValueError("Unsupported model")

# ----- OCR ----
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def extract_text_with_docai(image_path):
    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(project=PROJECT_ID, location=LOCATION, processor=PROCESSOR_ID)
    with open(image_path, "rb") as f:
        raw_document = documentai.RawDocument(content=f.read(), mime_type="image/jpeg")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)
    return result.document.text

def extract_text(files, ocr_mode="Document AI"):
    combined_text = []
    for file in files:
        ext = os.path.splitext(file.name)[-1].lower()
        if ext == ".pdf":
            text = extract_text_from_pdf(file.name)
        elif ocr_mode == "Gemini Vision":
            text = gemini_vision_text(file.name)
        else:
            text = extract_text_with_docai(file.name)
        combined_text.append(f"--- Extracted from {file.name} ({ocr_mode}) ---\n{text}\n")
    return "\n".join(combined_text)

# ----- QA Parsing -----
def extract_qas(text):
    blocks = re.split(r"Q\d+\.\s*", text)
    qas = []
    for block in blocks[1:]:
        parts = block.strip().split("Answer:", maxsplit=1)
        if len(parts) == 2:
            question = parts[0].strip()
            answer = parts[1].strip()
            qas.append((question, answer))
    return qas

# ----- Evaluation -----
def evaluate_with_model(llm, q, correct, student, max_marks):
    prompt = prompt_template.format_messages(
        question=q,
        correct_answer=correct,
        student_answer=student,
        max_marks=max_marks
    )
    print("\n--- PROMPT TO AI ---\n", prompt[0].content)
    response = llm.invoke(prompt)
    text = response.content.strip()
    print("--- AI RESPONSE ---\n", text)
    score = int(re.search(r"(\d+)[/\\](\d+)", text).group(1)) if re.search(r"(\d+)[/\\](\d+)", text) else 0
    return score, text

def create_pdf(text, output_path="result.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line.encode("latin-1", "ignore").decode("latin-1"))
    pdf.output(output_path)
    return output_path

# ----- Main -----
def evaluate_app(correct_file, student_files, model_name, ocr_mode):
    logs = []
    log = lambda m: (logs.append(m), print(m))

    correct_text = extract_text([correct_file], ocr_mode)
    student_text = extract_text(student_files, ocr_mode)

    logs.append("\n--- Extracted Correct QA Text ---\n" + correct_text[:1000])
    logs.append("\n--- Extracted Student Answer Text ---\n" + student_text[:1000])

    correct_qas = extract_qas(correct_text)
    student_qas = extract_qas(student_text)

    log(f"✅ Detected {len(correct_qas)} questions and {len(student_qas)} student answers")

    llm = get_llm(model_name)
    results = []
    total = 0
    for i, ((q, correct), (_, student)) in enumerate(zip(correct_qas, student_qas)):
        log(f"🤖 Evaluating Q{i+1} with {model_name}...")
        score, feedback = evaluate_with_model(llm, q, correct, student, 5)
        results.append(f"Q{i+1}: {q}\nStudent Answer: {student}\nMarks: {score}/5\nFeedback: {feedback}\n")
        total += score

    final_text = f"--- {model_name} ---\nScore: {total}/{len(correct_qas)*5}\n\n" + "\n".join(results)
    pdf_path = create_pdf(final_text)
    return final_text, pdf_path, "\n".join(logs)

# ----- UI -----
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Multi-LLM Student Answer Evaluator")

    with gr.Row():
        correct_file = gr.File(label="✅ Correct QA File", file_types=[".pdf", ".jpg", ".jpeg"])
        student_files = gr.File(label="👨‍🎓 Student Answer Sheets", file_types=[".pdf", ".jpg", ".jpeg"], file_count="multiple")

    with gr.Row():
        model_choice = gr.Radio(["DeepSeek-R1", "LLaMA3", "Gemini-1.5"], value="DeepSeek-R1", label="🧠 Choose Model")
        ocr_choice = gr.Radio(["Document AI", "Gemini Vision"], value="Document AI", label="👁️ OCR Method")

    submit = gr.Button("🚀 Evaluate")
    result_output = gr.Textbox(label="📊 Evaluation Result", lines=25)
    log_output = gr.Textbox(label="📜 Logs", lines=20)
    pdf_download = gr.File(label="⬇️ Download PDF Result")

    submit.click(fn=evaluate_app, inputs=[correct_file, student_files, model_choice, ocr_choice], outputs=[result_output, pdf_download, log_output])

if __name__ == "__main__":
    demo.launch()
