# 🧠 RAG Chatbot with DeepSeek/LLaMA3 + MongoDB + Ollama + Gradio

A lightweight Retrieval-Augmented Generation (RAG) chatbot powered by:
- **DeepSeek/LLaMA3 (via Ollama)**
- **MongoDB Atlas Vector Search**
- **LangChain**
- **HuggingFace Embeddings**
- **Gradio UI**

---

## 🚀 Features

- 🧠 **Retrieval-Augmented Generation** using custom MongoDB vector search or LangChain retrievers
- ⚡ **Real-time performance timing** (embedding + LLM call breakdown)
- 📁 **File-level context tracing** (shows which documents were used)
- 🤖 **Switchable models** (`deepseek-r1` or `llama3`)
- 💬 **Gradio-based chatbot UI** with context visibility and developer tools

---


## 🔧 Setup

### 1. 🧪 Prerequisites

- Python 3.9+
- MongoDB Atlas (with [Vector Search](https://www.mongodb.com/docs/atlas/atlas-search/vector-search/) enabled)
- Ollama installed and models pulled:
  ```bash
  ollama pull deepseek
  ollama pull llama3

