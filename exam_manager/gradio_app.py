import gradio as gr
import os
from bson import ObjectId
from pymongo import MongoClient
from upload_books_n_chunks import upload_book_pdf
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load ENV
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["sample_mflix"]
books = db["books"]
book_chunks = db["book_chunks"]

llm = OllamaLLM(model="llama3")

qa_prompt = PromptTemplate.from_template(
    """You are an expert CBSE teacher. Based ONLY on the content below, generate:

    - 5 MCQs
    - 3 short questions
    - 2 long questions

    Do not use outside knowledge. Keep it fair and exam-ready.

    Content:
    {context}

    Questions:"""
)

# ---------- 🧠 Step 1: Upload Book Tab ----------

def upload_book_ui(file, title, subject, class_level):
    if not file:
        return "⚠️ Please upload a file."

    save_path = f"./docs/{file.name}"
    file.save(save_path)

    book_id = upload_book_pdf(
        file_path=save_path,
        title=title,
        subject=subject,
        class_level=int(class_level),
        uploaded_by="teacher001"
    )
    return f"✅ Book uploaded and embedded! Book ID: {book_id}"

upload_tab = gr.Interface(
    fn=upload_book_ui,
    inputs=[
        gr.File(label="📄 Upload Book (PDF)"),
        gr.Textbox(label="Title", placeholder="e.g. Math Class 12"),
        gr.Textbox(label="Subject", placeholder="e.g. Math"),
        gr.Textbox(label="Class", placeholder="e.g. 12")
    ],
    outputs=gr.Textbox(label="Upload Status"),
    title="📚 Upload Book"
)

# ---------- 📘 Step 2: Select Book & Chapters ----------

def list_books():
    all_books = list(books.find({}, {"_id": 1, "title": 1}))
    return {book["title"]: str(book["_id"]) for book in all_books}

def list_chapters(book_id):
    book = books.find_one({"_id": ObjectId(book_id)})
    if not book:
        return []
    return book["chapters"]

book_choices = gr.Dropdown(label="📘 Select Book", choices=list_books().keys())
chapter_range = gr.Slider(minimum=1, maximum=20, step=1, value=5, label="📚 Chapter Range End")

def on_book_selected(book_title):
    book_id = list_books().get(book_title)
    chapters = list_chapters(book_id)
    if chapters:
        return gr.update(maximum=max(chapters), value=min(chapters) if chapters else 1)
    return gr.update(maximum=20, value=5)

# book_select_tab = gr.Interface(
#     fn=None,
#     inputs=[],
#     outputs=[],
#     title="🔍 Select Book & Chapter (Handled Inside Tab 3)"
# )

# ---------- 🧠 Step 3: Generate Questions ----------

def generate_questions(book_title, chapter_start, chapter_end):
    book_id = list_books().get(book_title)
    if not book_id:
        return "❌ Invalid book selected."

    query = {
        "bookId": ObjectId(book_id),
        "chapter": {"$gte": int(chapter_start), "$lte": int(chapter_end)}
    }

    chunks = list(book_chunks.find(query))
    if not chunks:
        return "❌ No content found for selected chapters."

    combined_text = "\n\n".join([c["text"] for c in chunks])
    prompt_text = qa_prompt.format(context=combined_text)
    result = llm.invoke(prompt_text)

    return result.strip()

generate_tab = gr.Interface(
    fn=generate_questions,
    inputs=[
        gr.Dropdown(label="📘 Select Book", choices=list_books().keys(), interactive=True),
        gr.Number(label="Start Chapter", value=1),
        gr.Number(label="End Chapter", value=5)
    ],
    outputs=gr.Textbox(label="📝 Generated Question Paper", lines=20),
    title="🧠 Generate Question Paper"
)

# ---------- 📲 Launch All Tabs ----------
gr.TabbedInterface(
    interface_list=[upload_tab, generate_tab],
    tab_names=["📚 Upload Book", "🧠 Generate Questions"]
).launch()
