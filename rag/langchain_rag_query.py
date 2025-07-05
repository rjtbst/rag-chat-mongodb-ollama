import os
import logging
import torch
import gradio as gr
from dotenv import load_dotenv
from pymongo import MongoClient

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import PromptTemplate

from rag_query import search  # Your custom MongoDB vector search

# ---------------------------------------------
# 🔧 Configuration & Setup
# ---------------------------------------------

logging.basicConfig(level=logging.INFO)
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "sample_mflix"
COLLECTION_NAME = "docs"

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# Prompt Template
qa_prompt = PromptTemplate.from_template(
    """Answer the question based on the context below:

    Context:
    {context}

    Question:
    {question}

    Answer:"""
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

# Vector Store
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding_model,
    index_name="docSemanticSearch",
    text_key="content",
    embedding_key="embedding"
)

# Custom Retriever
class MongoDBCustomRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> list[Document]:
        raw_results = search(query)
        return [
            Document(page_content=doc["content"], metadata={"filename": doc.get("filename", "Unknown")})
            for doc in raw_results
        ]

# LLM
llm = OllamaLLM(model="llama3")

# ---------------------------------------------
# 🔍 RAG Pipeline with Prompt Display
# ---------------------------------------------

def run_rag_pipeline(user_question: str, retriever_choice: str):
    try:
        retrieved_docs = []

        if retriever_choice == "Custom Mongo Vector Search":
            retriever = MongoDBCustomRetriever()
            raw_results = search(user_question)
            retrieved_docs = [
                Document(page_content=doc["content"], metadata={"filename": doc.get("filename", "Unknown")})
                for doc in raw_results
            ]

        elif retriever_choice == "LangChain Similarity Search":
            retrieved_docs = vector_store.similarity_search(user_question, k=3)
            class StaticRetriever(BaseRetriever):
                def _get_relevant_documents(self, _: str):
                    return retrieved_docs
            retriever = StaticRetriever()

        elif retriever_choice == "LangChain Filtered Retriever":
            retriever = vector_store.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.6})
            retrieved_docs = retriever.get_relevant_documents(user_question)

        else:
            return "❌ Invalid retriever selected.", "", "", ""

        # Manually generate final prompt
        combined_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        final_prompt_text = qa_prompt.format(context=combined_context, question=user_question)

        # Run the RAG chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )

        result = rag_chain.invoke({"query": user_question})
        answer = result["result"]
        source_docs = result["source_documents"]

        # Retrieved Docs Summary
        retrieved_preview = "\n\n".join(
            f"📄 {doc.metadata.get('filename', 'Unknown')}:\n{doc.page_content[:300]}..."
            for doc in retrieved_docs
        )

        source_filenames = "\n".join(
            f"✅ {doc.metadata.get('filename', 'Unknown')}" for doc in source_docs
        )

        return answer.strip(), retrieved_preview.strip(), source_filenames.strip(), final_prompt_text.strip()

    except Exception as err:
        return f"❌ Error: {err}", "", "", ""

# ---------------------------------------------
# 💬 Gradio UI
# ---------------------------------------------

interface = gr.Interface(
    fn=run_rag_pipeline,
    inputs=[
        gr.Textbox(label="🔍 Ask a Question", placeholder="E.g. What are Rajat's skills?", lines=2),
        gr.Radio(
            ["Custom Mongo Vector Search", "LangChain Similarity Search", "LangChain Filtered Retriever"],
            label="🔄 Choose Retriever",
            value="Custom Mongo Vector Search"
        ),
       
    ],
    outputs=[
        gr.Textbox(label="🧠 RAG Answer , add system prompt", lines=5),
        gr.Textbox(label="🔎 Retrieved Documents (Before RAG)", lines=10),
        gr.Textbox(label="📁 Source Filenames from RAG Output", lines=5),
        gr.Textbox(label="📜 Final Prompt Sent to LLM", lines=12)
       
    ],
    title="🧠 Chat with RAG (LangChain + Ollama + MongoDB)",
    description="Ask questions and compare how different retrievers affect document selection and final answers."
)

if __name__ == "__main__":
    interface.launch()
