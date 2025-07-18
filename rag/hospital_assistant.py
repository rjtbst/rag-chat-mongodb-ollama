import os
import logging
import torch
import re
import time
import gradio as gr
import sys
from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

from rag_query import search

log_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
    logging.FileHandler("chatbot.log", encoding="utf-8"),
    logging.StreamHandler(sys.stdout)
])
logger = logging.getLogger(__name__)

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "sample_mflix"
COLLECTION_NAME = "docs"
client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

def book_appointment(name: str, date: str, time_: str):
    logger.info(f"Booking appointment for {name} at {date} {time_}")
    appointments = client[DB_NAME]["appointments"]
    appointments.insert_one({"name": name, "date": date, "time": time_, "booked_at": time.time()})
    client[DB_NAME]["users"].update_one(
        {"name": name},
        {"$push": {"appointments": {"date": date, "time": time_, "doctor": "Not specified"}}},
        upsert=True
    )
    return f"✅ Appointment booked for {name} on {date} at {time_}"

def find_doctors_by_speciality(speciality: str):
    logger.info(f"Finding doctors for speciality: {speciality}")
    doctors = client[DB_NAME]["doctors"]
    result = doctors.find({"speciality": {"$regex": speciality, "$options": "i"}})
    return [f"{doc['name']} ({doc['speciality']}) - Available: {doc.get('available', 'Unknown')}" for doc in result]

def check_doctor_availability(name: str):
    logger.info(f"Checking availability for doctor: {name}")
    doctors = client[DB_NAME]["doctors"]
    doc = doctors.find_one({"name": {"$regex": name, "$options": "i"}})
    if doc:
        return f"{doc['name']} is available on {doc.get('available_days', 'N/A')} at {doc.get('available_times', 'N/A')}"
    return "Doctor not found."

def get_hospital_services():
    logger.info("Fetching hospital services")
    services = client[DB_NAME]["services"].find()
    return "\n".join([f"{s['name']}: {s.get('description', '')}" for s in services])

def get_emergency_contact():
    logger.info("Fetching emergency contact info")
    emergency_info = client[DB_NAME]["info"].find_one({"type": "emergency"})
    return emergency_info.get("contact", "Not Available") if emergency_info else "Emergency contact not found."

def get_last_appointment(email: str):
    logger.info(f"Fetching last appointment for {email}")
    user = client[DB_NAME]["users"].find_one({"email": email})
    appts = user.get("appointments", []) if user else []
    if not appts:
        return "No appointments found."
    latest = appts[-1]
    return f"Your last appointment was with {latest['doctor']} on {latest['date']} at {latest['time']}"

system_prompt = """You are a helpful assistant. You MUST reply in English using ONLY the context provided.

**Rules**:
- Use ONLY the information in the context below.
- If the context doesn't fully answer the user, consider calling one of the backend actions using this format:
  <action>function_name(...)</action>

**Available Actions**:
- book_appointment(name="NAME", date="YYYY-MM-DD", time="HHAM/PM")
- get_last_appointment(email="{user_email}")
- find_doctors_by_speciality(speciality="")
- check_doctor_availability(name="")
- get_emergency_contact()
- get_hospital_services()

- If unsure, respond: "I don’t know based on the given information."
- DO NOT begin answers with phrases like “Based on the context” or “According to the document”.
"""

qa_prompt = PromptTemplate.from_template("""
Follow the instructions carefully and answer based on the context provided.

Context:
{context}

Question:
{question}

Answer:
""")

class MongoDBCustomRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str):
        logger.info(f"Running custom Mongo vector search for query: {query}")
        raw_results = search(query)
        return [Document(page_content=doc["content"], metadata={"filename": doc.get("filename", "Unknown")}) for doc in raw_results]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding_model,
    index_name="docSemanticSearch",
    text_key="content",
    embedding_key="embedding"
)

def get_llm(model_name, user_email):
    logger.info(f"Creating LLM with model: {model_name}")
    return ChatOllama(
        model=model_name,
        system=system_prompt.format(user_email=user_email),
        temperature=0.0,
        top_p=0.95,
        stream=True
    )

def respond(email, user_message, chat_history, retriever_option, model_choice):
    logger.info(f"Received question from {email}: {user_message}")
    chat_history = chat_history + [[user_message, None]]
    yield user_message, chat_history, "", "", "", "", ""

    try:
        users = client[DB_NAME]["users"]
        user = users.find_one({"email": email})
        if not user:
            logger.info("New user, inserting record.")
            user = {"email": email, "name": "", "conversations": [], "appointments": []}
            users.insert_one(user)
        else:
            logger.info("Returning user found.")

        llm = get_llm(model_choice, email)

        if retriever_option == "Custom Mongo Vector Search":
            retriever = MongoDBCustomRetriever()
            docs = retriever._get_relevant_documents(user_message)
        else:
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(user_message)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = qa_prompt.format(context=context, question=user_message)
        logger.info(f"Generated prompt:\n{prompt}")

        streamed_answer = ""
        for chunk in llm.stream(prompt):
            streamed_answer += chunk.content
            chat_history[-1][1] = streamed_answer + "▌"
            yield user_message, chat_history, context[:1000], "\n".join([doc.metadata.get("filename", "") for doc in docs]), prompt.strip(), "", ""

        answer = streamed_answer.strip()
        logger.info(f"LLM final response: {answer}")

        thinking_match = re.search(r"<think>(.*?)</think>", answer, re.DOTALL)
        thinking_text = thinking_match.group(1).strip() if thinking_match else ""

        action_match = re.search(r"<action>(.*?)</action>", answer, re.DOTALL)
        if action_match:
            action_code = action_match.group(1).strip()
            logger.info(f"Detected action: {action_code}")
            try:
                result = eval(action_code, {
                    "book_appointment": book_appointment,
                    "find_doctors_by_speciality": find_doctors_by_speciality,
                    "check_doctor_availability": check_doctor_availability,
                    "get_hospital_services": get_hospital_services,
                    "get_emergency_contact": get_emergency_contact,
                    "get_last_appointment": get_last_appointment,
                    "email": email
                })
                answer = f"{answer.replace(action_match.group(0), '').strip()}\n\n➡️ Action Result:\n{result}"
                logger.info(f"Action result: {result}")
            except Exception as e:
                logger.exception("Action execution failed")
                answer = f"{answer.replace(action_match.group(0), '').strip()}\n\n❌ Failed to execute action: {e}"

        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        chat_history[-1][1] = answer

        users.update_one(
            {"email": email},
            {"$push": {"conversations": {"question": user_message, "answer": answer}}}
        )

        yield user_message, chat_history, context[:1000], "\n".join([doc.metadata.get("filename", "") for doc in docs]), prompt.strip(), thinking_text, ""

    except Exception as e:
        logger.exception("Exception occurred during respond()")
        chat_history[-1][1] = f"❌ Error: {e}"
        yield user_message, chat_history, "", "", "", "", ""

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Hospital Chatbot with Memory")

    user_id_box = gr.Textbox(label="🔑 Your Email", placeholder="Enter your email")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History")
            user_input = gr.Textbox(label="Ask a Question", placeholder="E.g. What are cardiology timings?")

            retriever_radio = gr.Radio([
                "Custom Mongo Vector Search", "LangChain Similarity Search"
            ], label="Retriever Type", value="Custom Mongo Vector Search")

            model_radio = gr.Radio([
                "deepseek-r1", "llama3"
            ], label="Model Choice", value="deepseek-r1")

            clear_btn = gr.Button("Clear Chat")

        with gr.Column(scale=1):
            retrieved_view = gr.Textbox(label="🔍 Retrieved Context", lines=12)
            filenames_view = gr.Textbox(label="📁 Source Files", lines=3)
            final_prompt_view = gr.Textbox(label="📝 Final Prompt", lines=8)
            thinking_view = gr.Textbox(label="🧠 Thinking", lines=2)
            timing_view = gr.Textbox(label="⏱️ Timing Info", lines=2)

    user_input.submit(
        fn=respond,
        inputs=[user_id_box, user_input, chatbot, retriever_radio, model_radio],
        outputs=[user_input, chatbot, retrieved_view, filenames_view, final_prompt_view, thinking_view, timing_view]
    )

    clear_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()
