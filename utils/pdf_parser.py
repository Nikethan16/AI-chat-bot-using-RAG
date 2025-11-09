import pdfplumber
import os
import json
from tqdm import tqdm

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a single PDF file.

    Args:
        file_path (str): Path to the PDF file.
    Returns:
        str: Extracted text content.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")

    return text.strip()


def extract_text_from_pdfs(pdf_dir: str, output_path: str):
    """
    Extract text from all PDFs in a directory and save to JSON.

    Args:
        pdf_dir (str): Directory containing PDF files.
        output_path (str): Path to output JSON file.
    """
    data = []
    for file in tqdm(os.listdir(pdf_dir)):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, file)
            text = extract_text_from_pdf(pdf_path)
            data.append({"filename": file, "text": text})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Extracted text from {len(data)} PDFs → {output_path}")


# Allow standalone script usage (for batch extraction)
if __name__ == "__main__":
    extract_text_from_pdfs("data/raw_pdfs", "data/pdf_texts.json")
