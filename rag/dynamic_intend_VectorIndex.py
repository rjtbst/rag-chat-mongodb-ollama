import os
import logging
import torch
import time
import re
import gradio as gr
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from numpy import dot
from numpy.linalg import norm

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
info_col = db["docs"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Embedding setup
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

# Intent setup for routing
COLLECTION_INTENTS = [
    {
        "name": "doctors",
        "intent": "Information about doctors, specialities, availability, and timings.",
        "collection": doctors_col,
        "index_name": "doctorSemanticSearch"
    },
    {
        "name": "services",
        "intent": "Information about hospital services, tests, scans like MRI, X-Ray.",
        "collection": services_col,
        "index_name": "serviceSemanticSearch"
    },
    {
        "name": "docs",
        "intent": "Hospital general policies, insurance, claim procedures, documents.",
        "collection": info_col,
        "index_name": "docsSemanticSearch"
    }
]

# Generate intent embeddings
for item in COLLECTION_INTENTS:
    item["intent_embedding"] = embedding_model.embed_query(item["intent"])

# Similarity function for routing
def detect_collection_by_embedding(query: str):
    query_emb = embedding_model.embed_query(query)

    def cosine_sim(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    best = None
    best_score = -1

    for item in COLLECTION_INTENTS:
        score = cosine_sim(query_emb, item["intent_embedding"])
        if score > best_score:
            best = item
            best_score = score

    logger.info(f"\U0001F4CC Query routed to collection: {best['name']} (score: {best_score:.3f})")
    return best

# Prompt setup
system_prompt = """You are a helpful assistant. You MUST reply in English using ONLY the context provided.

**Rules**:
- Use ONLY the information in the context below.
- If the context is empty or irrelevant, reply: \"I don’t know based on the given information.\"
- DO NOT begin with phrases like “Based on the context” or “According to the document”.
"""

qa_prompt = PromptTemplate.from_template("""
Follow the instructions carefully and answer based on the context provided.

Context:
{context}

Question:
{question}

Answer:
""")

def get_llm(model_name):
    return ChatOllama(
        model=model_name,
        system=system_prompt,
        temperature=0.0,
        top_p=0.95,
        stream=False
    )

def get_documents_from_routed_collection(query: str, top_k: int = 3):
    routing = detect_collection_by_embedding(query)
    vector_store = MongoDBAtlasVectorSearch(
        collection=routing["collection"],
        embedding=embedding_model,
        index_name=routing["index_name"],
        text_key="content",
        embedding_key="embedding"
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(query)
    return docs, routing["name"]

def respond(message, chat_history, model_choice):
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": "⌛ Retrieving context..."})
    yield "", chat_history, "", ""

    try:
        llm = get_llm(model_choice)
        docs, routed_collection = get_documents_from_routed_collection(message)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = qa_prompt.format(context=context, question=message)

        response = llm.invoke(prompt)
        answer = response.content.strip()

        chat_history[-1]["content"] = answer + f"\n\n📂 Collection: {routed_collection}"
        yield "", chat_history, context[:1000], prompt.strip()

    except Exception as e:
        chat_history[-1]["content"] = f"❌ Error: {e}"
        yield "", chat_history, "", ""

# Gradio Interface
def launch_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 🏥 Hospital Assistant (LLM + MongoDB + Routing)")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat History", type="messages")
                user_input = gr.Textbox(label="Ask a Question", placeholder="E.g. What are MRI timings?")
                model_radio = gr.Radio(["deepseek-r1", "llama3"], label="Model Choice", value="deepseek-r1")
                clear_btn = gr.Button("Clear Chat")

            with gr.Column(scale=1):
                retrieved_view = gr.Textbox(label="🔍 Retrieved Context", lines=12)
                prompt_view = gr.Textbox(label="📝 Final Prompt", lines=12)

        user_input.submit(
            fn=respond,
            inputs=[user_input, chatbot, model_radio],
            outputs=[user_input, chatbot, retrieved_view, prompt_view]
        )

        clear_btn.click(lambda: [], None, chatbot)

    demo.launch()

if __name__ == "__main__":
    launch_ui()
