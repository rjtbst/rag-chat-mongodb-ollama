import os
import gradio as gr
import re
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.agents import Tool, initialize_agent
from langchain.tools import tool
from langchain.callbacks import StdOutCallbackHandler

load_dotenv()

# ------------------ Mongo Setup ------------------
ATLAS_URI = os.getenv("MONGO_URI")
DB_NAME = "sample_mflix"
COLLECTION_NAME = "hospitals"

client = MongoClient(ATLAS_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# ------------------ Embedding ------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ------------------ Vector Store ------------------
vector_search = MongoDBAtlasVectorSearch(
    collection,
    embedding_model,
    index_name="vector_index"
)
retriever = vector_search.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ------------------ Prompt Template ------------------
prompt = PromptTemplate.from_template("""
System: You are a helpful hospital assistant. Always answer based on the provided context. If not sure, say \"I don't know\".

Context: {context}
User: {question}
Assistant:""")

# ------------------ Models ------------------
llm_models = {
    "Ollama (phi3)": ChatOllama(model="phi3", temperature=0),
}

# ------------------ Tool Functions ------------------
@tool
def search_doctor_by_name(name: str) -> str:
    """Search for a doctor by name and return their specialty and contact."""
    record = collection.find_one({"doctor_name": {"$regex": name, "$options": "i"}})
    if record:
        return f"Doctor: {record['doctor_name']} | Specialty: {record['speciality']} | Contact: {record['contact']}"
    return "Doctor not found."

@tool
def search_hospitals_by_location(location: str) -> str:
    """List hospitals in a specific location."""
    hospitals = collection.find({"location": {"$regex": location, "$options": "i"}})
    return "\n".join([f"{h['hospital_name']} - Contact: {h['contact']}" for h in hospitals]) or "No hospitals found."

@tool
def list_available_facilities(hospital_name: str) -> str:
    """List facilities available in a specific hospital."""
    record = collection.find_one({"hospital_name": {"$regex": hospital_name, "$options": "i"}})
    if record and "facilities" in record:
        return f"Facilities in {record['hospital_name']}:\n" + ", ".join(record["facilities"])
    return "Hospital or facilities not found."

@tool
def get_hospital_timings(hospital_name: str) -> str:
    """Get the available timings for a hospital."""
    record = collection.find_one({"hospital_name": {"$regex": hospital_name, "$options": "i"}})
    if record and "available_timings" in record:
        return f"{record['hospital_name']} is open: {record['available_timings']}"
    return "Timings not found."

@tool
def list_emergency_services(location: str) -> str:
    """List hospitals offering emergency services in a location."""
    hospitals = collection.find({"location": {"$regex": location, "$options": "i"}, "emergency_services": True})
    return "\n".join([h["hospital_name"] for h in hospitals]) or "No emergency services found in that location."

@tool
def hospital_qa_tool(query: str) -> str:
    """Answer hospital-related questions from documents (via RAG)."""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_models["Ollama (phi3)"],
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain.run(query)

# ------------------ Tool Agent ------------------
tools = [
    search_doctor_by_name,
    search_hospitals_by_location,
    list_available_facilities,
    get_hospital_timings,
    list_emergency_services,
    hospital_qa_tool
]

agent = initialize_agent(
    tools,
    llm_models["Ollama (phi3)"],
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
    callback_manager=StdOutCallbackHandler()
)

# ------------------ Chat Handler ------------------
def respond(message, history, debug_history, model_choice):
    logs = []
    logs.append("🧾 ----- New Interaction -----")
    logs.append(f"🤖 Model: {model_choice}")
    logs.append(f"💬 User: {message}")

    try:
        response = agent.run(message)
        logs.append(f"🧠 Agent Response:\n{response}")
    except Exception as e:
        logs.append(f"❌ ERROR: {str(e)}")
        response = "Sorry, something went wrong."

    debug_history.append("\n".join(logs))
    return response, history + [(message, response)], "\n\n---\n\n".join(debug_history)

# ------------------ Gradio UI ------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🏥 Hospital Assistant with Logs & Model Switcher")

    with gr.Row():
        model_selector = gr.Dropdown(choices=list(llm_models.keys()), label="Choose Model", value="Ollama (phi3)")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question")
    debug_console = gr.Textbox(label="🔍 Debug Log", lines=20, interactive=False, show_copy_button=True)

    chat_state = gr.State([])
    debug_state = gr.State([])

    def on_submit(user_msg, history, debug_history, model_choice):
        response, updated_history, debug_output = respond(user_msg, history, debug_history, model_choice)
        return "", updated_history, debug_output

    msg.submit(
        on_submit,
        inputs=[msg, chat_state, debug_state, model_selector],
        outputs=[msg, chatbot, debug_console],
    )

    demo.launch()
