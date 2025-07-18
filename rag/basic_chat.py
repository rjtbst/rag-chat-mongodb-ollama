import os
import logging
import torch
import re
import time
import gradio as gr
from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

from rag_query import search  # Your custom Mongo vector search function

# ------------------------------- 🔧 Config
logging.basicConfig(level=logging.INFO)
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "sample_mflix"
COLLECTION_NAME = "docs"

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------- 🧠 System Prompt
system_prompt = """You are a helpful assistant. You MUST reply in English using ONLY the context provided.

**Rules**:
- Use ONLY the information in the context below.
- If the context is empty or irrelevant, reply: "I don’t know based on the given information."
- DO NOT begin with phrases like “Based on the context” or “According to the document”.
"""

# ------------------------------- 📝 Prompt Template
qa_prompt = PromptTemplate.from_template("""
Follow the instructions carefully and answer based on the context provided.

Context:
{context}

Question:
{question}

Answer:
""")

# ------------------------------- 🧠 LLM Factory
def get_llm(model_name):
    return ChatOllama(
        model=model_name,
        system=system_prompt,
        temperature=0.0,
        top_p=0.95,
        stream=False
    )

# ------------------------------- 🧠 Memory (optional)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question",
    output_key="answer"
)

# ------------------------------- 🧾 Embeddings + Vector Store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding_model,
    index_name="docSemanticSearch",
    text_key="content",
    embedding_key="embedding"
)

# ------------------------------- 🔍 Custom MongoDB Retriever
class MongoDBCustomRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str):
        raw_results = search(query)
        return [
            Document(page_content=doc["content"], metadata={"filename": doc.get("filename", "Unknown")})
            for doc in raw_results
        ]

# ------------------------------- 💬 Respond Function with Stage-wise Yield
def respond(user_message, chat_history, retriever_option, model_choice):
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": "⌛ Retrieving context..."})
    yield "", chat_history, "", "", "", "", ""

    try:
        overall_start = time.time()
        llm = get_llm(model_choice)

        # Step 1: Retrieval
        context_start = time.time()
        if retriever_option == "Custom Mongo Vector Search":
            retriever = MongoDBCustomRetriever()
            docs = retriever._get_relevant_documents(user_message)
        elif retriever_option == "LangChain Similarity Search":
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(user_message)
        elif retriever_option == "LangChain Filtered Retriever":
            retriever = vector_store.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.75})
            docs = retriever.get_relevant_documents(user_message)
        else:
            chat_history[-1]["content"] = "❌ Invalid retriever option"
            yield "", chat_history, "", "", "", "", ""
            return

        context_time = time.time() - context_start

        context = "\n\n".join([doc.page_content for doc in docs])
        context_snippets = "\n\n---\n\n".join([doc.page_content[:500] for doc in docs])
        file_names = "\n".join([f"📄 {doc.metadata.get('filename', 'unknown')}" for doc in docs])
        prompt = qa_prompt.format(context=context, question=user_message)

        # Step 2: Yield context info early
        chat_history[-1]["content"] = "⌛ Calling LLM..."
        yield "", chat_history, context_snippets.strip(), file_names.strip(), prompt.strip(), "", ""

        # Step 3: LLM Call
        llm_start = time.time()
        response = llm.invoke(prompt)
        llm_time = time.time() - llm_start

        answer = str(response.content).strip()
        thinking = ""

        match = re.search(r"<think>(.*?)</think>", answer, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

        total_time = time.time() - overall_start
        timing_info = f"⏱️ Embedding+Retrieval: {context_time:.2f}s | LLM: {llm_time:.2f}s | Total: {total_time:.2f}s"

        # Step 4: Final response
        chat_history[-1]["content"] = f"{answer}\n\n🤖 Model: `{model_choice}`\n{timing_info}"
        yield "", chat_history, context_snippets.strip(), file_names.strip(), prompt.strip(), thinking, timing_info

    except Exception as e:
        chat_history[-1]["content"] = f"❌ Error: {e}"
        yield "", chat_history, "", "", "", "", ""

# ------------------------------- 🎛 Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Chat with RAG (DeepSeek & LLaMA3 + MongoDB + Ollama)")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History", type="messages")
            user_input = gr.Textbox(label="Ask a Question", placeholder="E.g. What are Rajat's skills?")

            retriever_radio = gr.Radio(
                ["Custom Mongo Vector Search", "LangChain Similarity Search", "LangChain Filtered Retriever"],
                label="Retriever Type",
                value="Custom Mongo Vector Search"
            )

            model_radio = gr.Radio(
                ["deepseek-r1", "llama3"],
                label="Model Choice",
                value="deepseek-r1"
            )

            clear_btn = gr.Button("Clear Chat")

        with gr.Column(scale=1):
            retrieved_view = gr.Textbox(label="🔍 Retrieved Context", lines=12)
            filenames_view = gr.Textbox(label="📁 Source Files", lines=3)
            final_prompt_view = gr.Textbox(label="📝 Final Prompt Sent to LLM", lines=12)
            thinking_view = gr.Textbox(label="🧠 Thinking (if any)", lines=12)
            timing_view = gr.Textbox(label="⏱️ Timing Info", lines=2)

    user_input.submit(
        fn=respond,
        inputs=[user_input, chatbot, retriever_radio, model_radio],
        outputs=[user_input, chatbot, retrieved_view, filenames_view, final_prompt_view, thinking_view, timing_view]
    )

    clear_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()