import pdfplumber
import os
import json
from tqdm import tqdm

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a single PDF file."""
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return ""

    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"[PDF Extraction Error] {e}")
        return ""

    return text.strip()


def extract_text_from_pdfs(pdf_dir: str, output_path: str):
    """Extract text from all PDFs in a directory and save as JSON."""
    try:
        if not os.path.exists(pdf_dir):
            print(f"[Error] Directory not found: {pdf_dir}")
            return

        data = []
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

        for file in tqdm(pdf_files, desc="Extracting PDF text"):
            pdf_path = os.path.join(pdf_dir, file)
            text = extract_text_from_pdf(pdf_path)
            if text:
                data.append({"filename": file, "text": text})
            else:
                print(f"[Warning] No text extracted from {file}")

        if data:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Extracted text from {len(data)} PDFs → {output_path}")
        else:
            print("[Warning] No valid PDF text extracted. Output not created.")

    except Exception as e:
        print(f"[Batch Extraction Error] {e}")


if __name__ == "__main__":
    extract_text_from_pdfs("data/raw_pdfs", "data/pdf_texts.json")
