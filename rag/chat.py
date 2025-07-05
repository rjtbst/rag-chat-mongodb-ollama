import os
import logging
import torch
import re
import gradio as gr
from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

from rag_query import search  # your custom Mongo vector search function

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
system_prompt = """You are a helpful assistant. You MUST reply to answer clearly and politely using ONLY the context provided.

**Rules**:
- Use ONLY the information in the context below.
- If the context is empty or irrelevant, reply: "I don’t know based on the given information."
- DO NOT begin with phrases like “Based on the context” or “According to the document”.

"""

# ------------------------------- 📝 Prompt Template
qa_prompt = PromptTemplate.from_template("""
follow user instructions carefully and answer the question based on the context provided.

Context:
{context}

Question:
{question}

Answer:
""")

# ------------------------------- 🧠 LLM & Memory
llm = ChatOllama(
    model="deepseek-r1",  # or "llama3"
    system=system_prompt,
    temperature=0.0,
    top_p=0.95,
)

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
    def _get_relevant_documents(self, query: str) -> list[Document]:
        raw_results = search(query)
        return [
            Document(page_content=doc["content"], metadata={"filename": doc.get("filename", "Unknown")})
            for doc in raw_results
        ]

# ------------------------------- 🔁 RAG Chain
def get_chain(retriever):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        output_key="answer",
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

# ------------------------------- 💬 Chat Function
def chat_interface(user_input, retriever_option):
    try:
        if retriever_option == "Custom Mongo Vector Search":
            retriever = MongoDBCustomRetriever()
        elif retriever_option == "LangChain Similarity Search":
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        elif retriever_option == "LangChain Filtered Retriever":
            retriever = vector_store.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.75})
        else:
            return "❌ Invalid retriever option", "", "", "", ""

        chain = get_chain(retriever)
        result = chain.invoke({"question": user_input})

        raw_answer = result["answer"].strip()
        model_name = llm.model
        print(f"\n--- Raw Answer ---\n{raw_answer}\n")

        thinking = ""
        answer = raw_answer

        # Extract <think> block only for deepseek
        if model_name == "deepseek-r1":
            match = re.search(r"<think>(.*?)</think>", raw_answer, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

        for unwanted in ["Based on the context", "According to the document", "As per the context"]:
            if answer.lower().startswith(unwanted.lower()):
                answer = answer[len(unwanted):].strip(",. ")

        source_docs = result.get("source_documents", [])
        context_snippets = "\n\n---\n\n".join([doc.page_content[:500] for doc in source_docs])
        file_names = "\n".join([f"📄 {doc.metadata.get('filename', 'unknown')}" for doc in source_docs])
        final_prompt = qa_prompt.format(
            context="\n\n".join([doc.page_content for doc in source_docs]),
            question=user_input
        )

        return answer, context_snippets.strip(), file_names.strip(), final_prompt.strip(), thinking

    except Exception as e:
        return f"❌ Error: {e}", "", "", "", ""

# ------------------------------- 🎛 Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Chat with RAG (LangChain + Ollama + MongoDB + deepseek-r1/llama3.1)")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History", type="messages")
            user_input = gr.Textbox(label="Ask a Question", placeholder="E.g. What are Rajat's skills?")
            retriever_radio = gr.Radio(
                ["Custom Mongo Vector Search", "LangChain Similarity Search", "LangChain Filtered Retriever"],
                label="Retriever Type",
                value="Custom Mongo Vector Search"
            )
            clear_btn = gr.Button("Clear Chat")

        with gr.Column(scale=1):
            retrieved_view = gr.Textbox(label="🔍 Retrieved Context", lines=12)
            filenames_view = gr.Textbox(label="📁 Source Files", lines=3)
            final_prompt_view = gr.Textbox(label="📝 Final Prompt Sent to LLM", lines=12)
            thinking_view = gr.Textbox(label="🧠 Thinking (DeepSeek)", lines=12)

    def respond(user_message, chat_history, retriever_option):
        answer, context_snippets, file_names, final_prompt, thinking = chat_interface(user_message, retriever_option)

        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": answer})

        return "", chat_history, context_snippets, file_names, final_prompt, thinking

    user_input.submit(
        respond,
        inputs=[user_input, chatbot, retriever_radio],
        outputs=[user_input, chatbot, retrieved_view, filenames_view, final_prompt_view, thinking_view]
    )

    clear_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()
