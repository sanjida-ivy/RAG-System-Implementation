# 🧠 RAG-System-Implementation

This project implements a **Retrieval-Augmented Generation (RAG)** system that allows users to ask natural language questions, optionally uploading PDF documents to enhance the responses. It combines local Large Language Model (LLM) inference with document indexing using vector embeddings and Pinecone for semantic search.

## 🔧 Features

- ✅ Ask questions via a web-based UI (built with Streamlit)
- 📄 Upload one or more PDFs for the assistant to read and use as context
- 🧠 LLM responses powered by Ollama (e.g., `llama3`)
- 🔍 Semantic document retrieval using SentenceTransformers + Pinecone
- 🧱 Modular architecture using LangChain tools and ReAct agents

## 📁 Folder Structure

- `app.py` — Streamlit frontend for chat and file upload
- `agent.py` — Sets up LangChain agent with tools for document QA
- `utils.py` — Query answering logic (uses LLM directly or with PDF context)
- `pdf_processing.py` — Extracts and embeds PDF content into Pinecone
- `vector_utils.py` — Handles embeddings and upsert/query in Pinecone
- `pinecone_setup.py` — Pinecone index initialization and client management
- `config.py` — Loads environment settings
- `.env` — Configuration for Ollama, Pinecone, and embedding model

## 🚀 How It Works

1. **Without PDF:** Questions are answered directly using an LLM served locally by Ollama.
2. **With PDF:** Uploaded PDFs are parsed, chunked, embedded, and stored in Pinecone. Responses use this indexed content as the source of truth.

## 🧠 Models Used

- **LLM:** Via Ollama (`llama3:latest` by default)
- **Embeddings:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

## 🌐 Requirements

- Python 3.9+
- Dependencies: See `requirements.txt` (not provided in this repo yet)
- Running Ollama locally
- Pinecone account and API key

## ⚙️ Usage

1. Clone the repo and install dependencies.
2. Set up your `.env` file (see provided `.env` for structure).
3. Run the app:
   ```bash
   streamlit run app.py
