import json, os, re
from typing import List, Dict
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str, filename: str = "uploaded_document") -> List[Dict]:
    """
    Splits a single document into overlapping chunks for RAG embedding.

    Args:
        text (str): The extracted document text.
        filename (str): Name of the document for metadata tagging.

    Returns:
        List[Dict]: List of chunk dictionaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n##", "\n#", ".", "\n", " "]
    )

    topic = filename.replace(".pdf", "").replace("_", " ").strip().title()
    chunks = []
    for i, chunk in enumerate(splitter.split_text(text)):
        # Try to detect section heading if any
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

    return chunks


def chunk_documents(input_json: str, output_json: str):
    """
    Splits all documents in the input JSON into chunks and writes to JSONL.

    Args:
        input_json (str): Path to input JSON file containing extracted text.
        output_json (str): Output path for chunked JSONL file.
    """
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input file not found: {input_json}")

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

    print(f"✅ Chunked {len(all_chunks)} total chunks → {output_json}")


# Standalone usage for batch processing
if __name__ == "__main__":
    chunk_documents("data/pdf_texts.json", "data/processed_chunks.jsonl")
