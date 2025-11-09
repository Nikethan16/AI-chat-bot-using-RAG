import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config.config import DATA_DIR, FAISS_INDEX_PATH, EMBED_MODEL_LOCAL

# Load embedding model
print(f"Loading embedding model: {EMBED_MODEL_LOCAL}")
embed_model = SentenceTransformer(EMBED_MODEL_LOCAL)

def embed_texts(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = embed_model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embeddings.append(emb)
    return np.vstack(embeddings)

def build_faiss_index(chunks_path, index_path):
    
    print(f"Reading chunks from: {chunks_path}")
    texts, metadata = [], []

    # Load chunked JSONL file
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading chunks"):
            obj = json.loads(line.strip())
            texts.append(obj["text"])
            metadata.append({
                "topic": obj.get("topic_title"),
                "section": obj.get("section"),
                "filename": obj.get("filename")
            })

    print(f"Generating embeddings for {len(texts)} chunks...")
    vectors = embed_texts(texts).astype("float32")

    # Build FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Save FAISS index
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")

    # Save metadata separately
    metadata_path = os.path.join(DATA_DIR, "chunk_metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for meta in metadata:
            f.write(json.dumps(meta) + "\n")

    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    chunks_file = os.path.join(DATA_DIR, "processed_chunks.jsonl")
    build_faiss_index(chunks_file, FAISS_INDEX_PATH)
