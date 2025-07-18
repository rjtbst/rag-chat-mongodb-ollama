import os
import re
import pdfplumber
from datetime import datetime
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Load environment
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client.sample_mflix
books = db.books
book_chunks = db.book_chunks

# Embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 🧹 Text Cleaner
def clean_text(text):
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# 📖 Chapter Splitter
def split_by_chapters(full_text):
    # Adjust regex depending on your chapter heading style
    pattern = r"\n?(Chapter\s+\d+[:.\s]*)"
    split = re.split(pattern, full_text)

    chunks = []
    for i in range(1, len(split), 2):
        title = split[i].strip()
        content = clean_text(split[i + 1])
        chapter_num = int(re.search(r'\d+', title).group())
        chunks.append({
            "chapter": chapter_num,
            "title": title,
            "text": content
        })
    return chunks

# 📥 Main upload + embed function
def upload_book_pdf(file_path, title, subject, class_level, uploaded_by="teacher001"):
    print(f"📄 Reading {file_path}")
    with pdfplumber.open(file_path) as pdf:
        full_text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])

    chapter_chunks = split_by_chapters(full_text)

    book_id = books.insert_one({
        "title": title,
        "subject": subject,
        "class": class_level,
        "chapters": [c["chapter"] for c in chapter_chunks],
        "uploadedBy": uploaded_by,
        "uploadDate": datetime.utcnow()
    }).inserted_id

    for ch in chapter_chunks:
        embedding = model.encode(ch["text"], normalize_embeddings=True).tolist()

        book_chunks.insert_one({
            "bookId": book_id,
            "chapter": ch["chapter"],
            "title": ch["title"],
            "text": ch["text"],
            "embedding": embedding,
            "type": "chapter"
        })
        print(f"✅ Chapter {ch['chapter']} embedded and stored.")

    print(f"\n📘 Book '{title}' uploaded with {len(chapter_chunks)} chapters.")
    return str(book_id)



if __name__ == "__main__":
    pdf_path = "docs/math_class12.pdf"
    upload_book_pdf(
        file_path=pdf_path,
        title="Class 12 Mathematics",
        subject="Math",
        class_level=12
    )
