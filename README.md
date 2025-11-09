# Healthcare Assistant — RAG Chatbot

This repository contains the Streamlit RAG chatbot for the NeoStats assessment.
Features:
- RAG using FAISS + e5-base-v2 embeddings
- Groq LLM (llama-3.3-70b) for generation
- Google Custom Search fallback
- Document upload + chunking + indexing

**Do NOT commit secret keys or `data/` files.** See `DEPLOY.md` for deploy steps and required secrets.
