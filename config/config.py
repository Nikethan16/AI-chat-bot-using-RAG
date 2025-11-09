import os
from dotenv import load_dotenv

# =====================================
# LOAD ENVIRONMENT VARIABLES
# =====================================
load_dotenv()

# =====================================
# API KEYS & ENDPOINTS
# =====================================

# Groq API (LLM)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Google Custom Search (Fallback Web Retrieval)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CX_ID = os.getenv("GOOGLE_CX_ID", "")
GOOGLE_SEARCH_URL = "https://customsearch.googleapis.com/customsearch/v1"
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"

# =====================================
# LLM SETTINGS (Groq)
# =====================================
LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.25  # Low = factual, high = creative
MAX_CONTEXT_TOKENS = 128000  # Groq supports large token window

# =====================================
# EMBEDDING MODEL (LOCAL)
# =====================================
EMBED_PROVIDER = "local"
EMBED_MODEL_LOCAL = os.getenv("EMBED_MODEL_LOCAL", "intfloat/e5-base-v2")
EMBED_DIM = 768  # Default for E5-base-v2

# =====================================
# CHUNKING / RAG PARAMETERS
# =====================================
CHUNK_SIZE = 550         # tokens (~350 words)
CHUNK_OVERLAP = 80       # overlap to preserve continuity
TOP_K = 6                # number of retrieved chunks

# =====================================
# PATHS
# =====================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")
CHUNKS_PATH = os.path.join(DATA_DIR, "processed_chunks.jsonl")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(DATA_DIR, "chunk_metadata.jsonl")

# =====================================
# SYSTEM PROMPT (CLEANED)
# =====================================
DEFAULT_SYSTEM_PROMPT = """
You are a reliable AI healthcare assistant trained on verified medical and public health resources.

Your goals:
1. Use the retrieved medical context to provide grounded, factual answers.
2. When no relevant context is found, indicate that information is insufficient.
3. Maintain a neutral, professional, and informative tone.
4. Do not include disclaimers or repetition — one is already displayed in the app UI.
5. Structure your answers clearly, with short paragraphs or bullet points when helpful.
6. If user-provided reports contain test results, interpret values cautiously and explain them in general terms.

Do not generate diagnoses or treatment plans.
"""
