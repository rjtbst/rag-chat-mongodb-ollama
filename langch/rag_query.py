# rag_query.py

import requests
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.sample_mflix
collection = db.docs

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def search(query, k=3):
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    results = collection.aggregate([
        {
            "$search": {
                "index": "docSemanticSearch",
                "knnBeta": {
                    "vector": query_vector,
                    "path": "embedding",
                    "k": k
                }
            }
        },
        {
            "$project": {
                "filename": 1,
                "content": 1,
                "score": {"$meta": "searchScore"}
            }
        },
         {
        "$match": {
            "score": {"$gte": 0.6}          # Uncomment this line if you want to use vector search
        }
    },
    ])
    return list(results)

def rag_answer(context, question):
    prompt = f"""Answer the question based on the context below:

Context:
{context}

Question:
{question}

Answer:"""
  

    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    return res.json()["response"]

if __name__ == "__main__":
    question = "who is rajat, what does he do ?"
    results = search(question)
    context = "\n\n".join([r["content"] for r in results])
    print("Context Sent to LLM:\n", context)
    print("Answer:\n", rag_answer(context, question))
