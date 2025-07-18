import gradio as gr
from PIL import Image
import pytesseract
import easyocr
import os
import io
from dotenv import load_dotenv
from google.cloud import documentai_v1 as documentai
from google.cloud.documentai_v1.types import document

# --- Load env ---
load_dotenv()
GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not GOOGLE_CREDS:
    raise ValueError("❌ Missing GOOGLE_APPLICATION_CREDENTIALS in .env file.")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDS

# --- Setup ---
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
easyocr_reader = easyocr.Reader(['en'], gpu=True)

# --- OCR Engines ---

def extract_easyocr(image_path):
    result = easyocr_reader.readtext(image_path)
    return "\n".join([r[1] for r in result])

def extract_tesseract(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)

def extract_docai(image_path):
    project_id = os.getenv("GOOGLE_PROJECT_ID")
    location = "us"  # or "eu"
    processor_id = os.getenv("DOCUMENT_AI_PROCESSOR_ID")  # you must set this in .env

    if not project_id or not processor_id:
        return "❌ Missing GOOGLE_PROJECT_ID or DOCUMENT_AI_PROCESSOR_ID in .env"

    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(project=project_id, location=location, processor=processor_id)

    with open(image_path, "rb") as image:
        image_content = image.read()

    mime_type = "application/pdf" if image_path.endswith(".pdf") else "image/jpeg"
    raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)   

    request = documentai.ProcessRequest(name=name, raw_document=raw_document)

    result = client.process_document(request=request)
    doc = result.document

    return doc.text if doc.text else "❌ No text found by Document AI."

# --- Dispatcher ---

def run_ocr(image_path, engine):
    if engine == "EasyOCR":
        return extract_easyocr(image_path)
    elif engine == "Tesseract":
        return extract_tesseract(image_path)
    elif engine == "Google Document AI":
        return extract_docai(image_path)
    return "Invalid engine"

# --- Gradio UI ---

gr.Interface(
    fn=run_ocr,
    inputs=[
        gr.Image(type="filepath", label="Upload Handwritten/Printed Image"),
        gr.Radio(
            ["EasyOCR", "Tesseract", "Google Document AI"],
            label="Choose OCR Engine",
            value="Google Document AI"
        )
    ],
    outputs=gr.Textbox(label="📄 Extracted Text", lines=30),
    title="📘 OCR Tool | Google Document AI + EasyOCR + Tesseract",
    description="Test different OCR engines. Google Document AI works well with handwriting and printed forms."
).launch()
