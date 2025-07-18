import os
import logging
import torch
import time
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load .env
load_dotenv()

# Mongo connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["sample_mflix"]
doctors_col = db["doctors"]
services_col = db["services"]

# Embedding setup
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

def generate_doctor_summary(doc):
    """Creates fresh summary from live doctor document."""
    return (
        f"Dr. {doc['name']} specializes in {doc['speciality']}. "
        f"They are available on {doc.get('available_days', 'N/A')} "
        f"during {doc.get('available_times', 'N/A')}. "
        f"Current status: {doc.get('available', 'Unknown')}"
    )

def generate_service_summary(doc):
    """Creates summary from service description."""
    return f"Service: {doc['name']}. Description: {doc.get('description', '')}"

def update_embeddings_for_doctors():
    updated = 0
    for doc in doctors_col.find({"name": {"$exists": True}}):
        summary = generate_doctor_summary(doc)
        embedding = embedding_model.embed_query(summary)

        doctors_col.update_one(
            {"_id": doc["_id"]},
            {"$set": {"content": summary, "embedding": embedding, "_last_embedded": datetime.now().timestamp()}}
        )
        logger.info(f"✅ Updated embedding for doctor: {doc['name']}")
        updated += 1

    logger.info(f"🔁 Total doctors updated: {updated}")

def update_embeddings_for_services():
    updated = 0
    for doc in services_col.find({"name": {"$exists": True}}):
        summary = generate_service_summary(doc)
        embedding = embedding_model.embed_query(summary)

        services_col.update_one(
            {"_id": doc["_id"]},
            {"$set": {"content": summary, "embedding": embedding, "_last_embedded": datetime.now().timestamp()}}
        )
        logger.info(f"✅ Updated embedding for service: {doc['name']}")
        updated += 1

    logger.info(f"🔁 Total services updated: {updated}")

if __name__ == "__main__":
    update_embeddings_for_doctors()
    update_embeddings_for_services()
    print("✅ Fresh summaries and embeddings updated for doctors and services.")
