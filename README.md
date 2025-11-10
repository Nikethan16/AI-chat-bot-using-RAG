# 🩺 Healthcare AI Chatbot (RAG + Web Search)

An intelligent **healthcare-focused assistant** built with **Retrieval-Augmented Generation (RAG)** and **LLM fallback via Groq API**.  
It analyzes uploaded reports, retrieves relevant medical context, and generates clear, factual insights — using **live web search** when local knowledge is insufficient.

---

## 🚀 Features

- **RAG Integration** – Retrieves answers from a local medical knowledge base using FAISS vector search  
- **Live Web Search** – Performs real-time Google searches when RAG context is weak  
- **Concise & Detailed Modes** – Switch between summarized or in-depth responses  
- **Document Upload** – Upload health reports (PDFs) for contextual analysis  
- **Fallback Logic** – Automatically blends RAG + web results  
- **Streamlit UI** – Clean, responsive, and easy to use  

---

## 🧠 Architecture Overview
Project structure
📦 project_root/
│
├── app.py # Streamlit app (main entry)
├── config/
│ └── config.py # API keys, constants, and model configs
│
├── models/
│ ├── llm.py # LLM logic (Groq)
│ └── embeddings.py # Builds FAISS index from embeddings
│
├── utils/
│ ├── pdf_parser.py # Extracts text from PDFs
│ ├── chunking.py # Splits documents into small text chunks
│ ├── rag_search.py # Retrieves context using FAISS
│ └── web_search.py # Performs Google Custom Search fallback
│
├── data/ # Local dataset
│ ├── raw_pdfs/ # Uploaded PDFs
│ ├── processed_chunks.jsonl # Chunked text for RAG
│ └── faiss_index.bin # Vector index for retrieval
│
└── requirements.txt
## ⚙️ How It Works

1. **Upload medical PDFs** → Extracted using `pdfplumber`  
2. **Chunking & Embedding** → Text split into segments for vector search  
3. **FAISS Search (RAG)** → Retrieves the most relevant chunks for the query  
4. **Web Fallback** → If context is poor, performs a Google search  
5. **LLM Response** → Groq-hosted LLaMA 3.3 model generates structured, factual answers  

---

## 💻 Setup Guide

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
