import json
import os
import re
from typing import List, Dict

# Try fallback import for LangChain versions
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, filename: str = "uploaded_document") -> List[Dict]:
    """Split text into overlapping chunks for RAG embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n##", "\n#", ".", "\n", " "]
    )

    topic = filename.replace(".pdf", "").replace("_", " ").strip().title()
    chunks = []

    for i, chunk in enumerate(splitter.split_text(text)):
        try:
            section_match = re.search(r"## (.*?)\n", chunk)
            section = section_match.group(1).strip() if section_match else "General"

            chunks.append({
                "filename": filename,
                "topic_title": topic,
                "section": section,
                "chunk_id": i,
                "text": chunk.strip(),
                "sources": ["WHO", "CDC", "ICMR"],
                "published_year": 2024,
                "region": ["Global"]
            })
        except Exception as e:
            print(f"[Warning] Skipped chunk {i} due to parsing error: {e}")

    return chunks


def chunk_documents(input_json: str, output_json: str):
    """Split all documents in input JSON and save as JSONL."""
    try:
        if not os.path.exists(input_json):
            print(f"[Error] Input file not found: {input_json}")
            return

        with open(input_json, "r", encoding="utf-8") as f:
            docs = json.load(f)

        all_chunks = []
        for doc in docs:
            filename = doc.get("filename", "unknown.pdf")
            text = doc.get("text", "")
            if text.strip():
                doc_chunks = chunk_text(text, filename)
                all_chunks.extend(doc_chunks)

        with open(output_json, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk) + "\n")

        print(f"Chunked {len(all_chunks)} text segments → {output_json}")

    except Exception as e:
        print(f"[Chunking Error] {e}")


if __name__ == "__main__":
    chunk_documents("data/pdf_texts.json", "data/processed_chunks.jsonl")
