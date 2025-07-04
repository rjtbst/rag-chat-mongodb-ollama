import os
import re
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from load_data import load_docs_from_folder
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client.sample_mflix
collection = db.docs

# Embedding model (384-dim)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Smart chunking (for long strings only)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,       # was 500
    chunk_overlap=200, 
)

# Clean text (used only for string-based input)
def clean_text(text: str) -> str:
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # fix mid-line newlines
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse excessive blank lines
    return text.strip()

# Prevent duplicate storage
def is_duplicate(filename: str, chunk_id: int) -> bool:
    return collection.count_documents({"filename": filename, "chunk_id": chunk_id}) > 0

# Embed and store
def embed_and_store(docs):
    total_chunks = 0
    for doc in docs:
        filename = doc["filename"]
        content = doc["content"]

        # Case 1: Already a list of Documents (from PDF/DOCX)
        if isinstance(content, list) and isinstance(content[0], Document):
            chunks = content

        # Case 2: It's a raw string (e.g., from TXT/HTML/JSON)
        else:
            cleaned = clean_text(content)
            if len(cleaned) > 600:
                texts = splitter.split_text(cleaned)
            else:
                texts = [cleaned]

            chunks = [
                Document(page_content=txt, metadata={"filename": filename, "chunk_id": i})
                for i, txt in enumerate(texts)
            ]


        # Process and store
        for i, doc in enumerate(chunks):
            chunk_id = doc.metadata.get("chunk_id", i)
            if is_duplicate(filename, chunk_id):
                continue

            embedding = model.encode(doc.page_content, normalize_embeddings=True).tolist()
            collection.insert_one({
                "filename": filename,
                "chunk_id": chunk_id,
                "content": doc.page_content,
                "embedding": embedding
            })

        print(f"✅ {filename} → {len(chunks)} chunks stored.")
        total_chunks += len(chunks)

    print(f"\n🔢 Total chunks embedded: {total_chunks}")

# Main
if __name__ == "__main__":
    folder_path = os.path.join(os.path.dirname(__file__), "docs")
    docs = load_docs_from_folder(folder_path)

    if docs:
        embed_and_store(docs)
        print("✅ All documents embedded and stored.")
    else:
        print("⚠️ No documents found.")
