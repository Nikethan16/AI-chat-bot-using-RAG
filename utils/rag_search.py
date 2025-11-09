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

# ===== LOAD MODEL =====
print(f"🔍 Loading embedding model: {EMBED_MODEL_LOCAL}")
embed_model = SentenceTransformer(EMBED_MODEL_LOCAL)

# ===== SAFE LOAD FAISS INDEX =====
print(f"Loading FAISS index from: {FAISS_INDEX_PATH}")

if not os.path.exists(FAISS_INDEX_PATH):
    print(f"⚠️ FAISS index not found at {FAISS_INDEX_PATH}. Skipping RAG for now.")
    index = None
else:
    index = faiss.read_index(FAISS_INDEX_PATH)

# ===== LOAD METADATA =====
metadata_path = os.path.join(DATA_DIR, "chunk_metadata.jsonl")
if os.path.exists(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]
else:
    metadata = []

# ===== EMBEDDING HELPERS =====
def embed_query(query: str):
    """Convert user query to normalized embedding."""
    return embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0].astype("float32")

def load_chunks():
    """Load pre-processed chunks."""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# ===== MAIN CONTEXT RETRIEVAL =====
def get_relevant_context(query: str, k: int = TOP_K, report_text: str = None):
    if index is None:
        return "", []

    chunks = load_chunks()

    # Optional: use extracted keywords from uploaded report to enrich query
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

    # Compute average FAISS distance
    avg_distance = total_distance / max(len(selected_chunks), 1)
    print(f"🧩 RAG average distance: {avg_distance:.3f}")

    # -------------------------------
    # ✅ Adaptive Relevance Threshold
    # -------------------------------
    # - <0.4 : Strongly Similarity
    # - 0.4–0.55 : Acceptable
    # - >0.55 : Likely off-topic, prefer web search
    if avg_distance > 0.5 or not selected_chunks:
        print(f"⚠️ RAG context weak (avg distance: {avg_distance:.3f}) — switching to web search.")
        return "", list(sources)

    # Bonus: heuristic filter to avoid accidental non-medical text
    medical_keywords = ["health", "disease", "treatment", "symptom", "diagnosis", "medical", "nutrition", "blood", "doctor"]
    context_preview = " ".join(selected_chunks[:3]).lower()
    if not any(word in context_preview for word in medical_keywords) and avg_distance > 0.55:
        print(f"⚠️ Context lacks medical relevance — triggering web search.")
        return "", list(sources)

    # Combine context for RAG
    combined_context = "\n\n".join(selected_chunks)
    return combined_context, list(sources)
