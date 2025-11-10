import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- API KEYS & ENDPOINTS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CX_ID = os.getenv("GOOGLE_CX_ID", "")
GOOGLE_SEARCH_URL = "https://customsearch.googleapis.com/customsearch/v1"
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"

# LLM SETTINGS (Groq) 
LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.25           
MAX_CONTEXT_TOKENS = 128000     

# EMBEDDING MODEL 
EMBED_PROVIDER = "local"
EMBED_MODEL_LOCAL = os.getenv("EMBED_MODEL_LOCAL", "intfloat/e5-base-v2")
EMBED_DIM = 768

# CHUNKING SETTINGS 
CHUNK_SIZE = 550         # ~350 words
CHUNK_OVERLAP = 80       # preserve continuity
TOP_K = 6                # number of retrieved chunks

# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")
CHUNKS_PATH = os.path.join(DATA_DIR, "processed_chunks.jsonl")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(DATA_DIR, "chunk_metadata.jsonl")

# SYSTEM PROMPT
DEFAULT_SYSTEM_PROMPT = """
You are a reliable AI healthcare assistant trained on verified medical and public health resources.

Goals:
1. Use retrieved medical context to provide grounded, factual answers.
2. If context is missing, say information is insufficient.
3. Maintain a calm, professional, and clear tone.
4. Avoid disclaimers or repetition (already shown in UI).
5. Structure responses clearly — short paragraphs or bullet points.
6. When interpreting test results, be general and cautious.

Do not generate diagnoses or treatment plans.
"""
