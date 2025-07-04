# embed_store.py

from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from load_data import load_docs_from_folder  # your custom doc loader

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client.sample_mflix
collection = db.docs

# 384-dim encoder
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

def embed_and_store(docs):
    for doc in docs:
        chunks = splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk, normalize_embeddings=True).tolist()
            collection.insert_one({
                "filename": doc["filename"],
                "chunk_id": i,
                "content": chunk,
                "embedding": embedding
            })

if __name__ == "__main__":
    folder_path = os.path.join(os.path.dirname(__file__), "docs")
    docs = load_docs_from_folder(folder_path)
    if docs:
        embed_and_store(docs)
    else:
        print("No documents to embed.")
