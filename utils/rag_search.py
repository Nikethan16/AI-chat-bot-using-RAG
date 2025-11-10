import os
import json
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from config.config import (
    DATA_DIR,
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    EMBED_MODEL_LOCAL,
    TOP_K
)

# Load embedding model
print(f"Loading embedding model: {EMBED_MODEL_LOCAL}")
embed_model = SentenceTransformer(EMBED_MODEL_LOCAL)

# Load FAISS index safely
print(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
if not os.path.exists(FAISS_INDEX_PATH):
    print(f"FAISS index not found at {FAISS_INDEX_PATH}. Skipping RAG initialization.")
    index = None
else:
    index = faiss.read_index(FAISS_INDEX_PATH)

# Load metadata
metadata_path = os.path.join(DATA_DIR, "chunk_metadata.jsonl")
if os.path.exists(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]
else:
    metadata = []

def embed_query(query: str):
    """Convert user query to normalized embedding."""
    return embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0].astype("float32")

def load_chunks():
    """Load processed text chunks."""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def get_relevant_context(query: str, k: int = TOP_K, report_text: str = None):
    """Retrieve the most relevant document chunks for a given query."""
    if index is None:
        return "", []

    chunks = load_chunks()

    # If a medical report is uploaded, use extracted keywords for query enrichment
    if report_text:
        keywords = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", report_text)
        keywords = list(set([k for k in keywords if len(k) > 3]))
        if keywords:
            query = " ".join(keywords[:30])

    # Embed query and retrieve top-k chunks
    q_emb = embed_query(query).reshape(1, -1)
    distances, indices = index.search(q_emb, k)

    selected_chunks, sources, total_distance = [], set(), 0.0

    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        total_distance += distances[0][i]
        obj = chunks[idx]
        topic = obj.get("topic_title", "General")
        section = obj.get("section", "Unknown Section")
        text = obj.get("text", "").strip().replace("\n", " ")
        srcs = obj.get("sources", ["Unknown"])
        sources.update(srcs)
        selected_chunks.append(f"[{topic} - {section}]\n{text}\n")

    # Calculate average similarity (lower distance = better match)
    avg_distance = total_distance / max(len(selected_chunks), 1)
    print(f"RAG average similarity distance: {avg_distance:.3f}")

    # Relevance threshold check
    # - < 0.4: Strong similarity
    # - 0.4–0.55: Acceptable
    # - > 0.55: Weak, trigger web search
    if avg_distance > 0.5 or not selected_chunks:
        print(f"RAG context weak (distance {avg_distance:.3f}) — fallback to web search.")
        return "", list(sources)

    # Basic filter to ensure medical relevance
    medical_keywords = [
        "health", "disease", "treatment", "symptom", "diagnosis",
        "medical", "nutrition", "blood", "doctor"
    ]
    context_preview = " ".join(selected_chunks[:3]).lower()
    if not any(word in context_preview for word in medical_keywords) and avg_distance > 0.55:
        print("Context not medically relevant — switching to web search.")
        return "", list(sources)

    # Combine selected chunks
    combined_context = "\n\n".join(selected_chunks)
    return combined_context, list(sources)
