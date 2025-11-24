# ai/extract.py
import os
import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from io import BytesIO
from typing import List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INGEST_DIR = os.path.join(REPO_ROOT, "backend", "data", "ingested")
PROCESSED_DIR = os.path.join(REPO_ROOT, "backend", "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ------------------- OCR HELPERS -------------------

def page_needs_ocr(text: str) -> bool:
    """Heuristic to detect scanned pages or pages with almost no extractable text."""
    if len(text.strip()) > 40:
        return False  # Probably digital text
    return True  # Very low text → likely scanned


def ocr_page(page_pix) -> str:
    """OCR a rendered page (pymupdf pixmap)."""
    img = Image.open(BytesIO(page_pix.tobytes("png")))
    return pytesseract.image_to_string(img)


# ------------------- CHUNKING LOGIC -------------------

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Split cleaned text into overlapping chunks."""
    cleaned = text.replace("\n", " ").replace("\t", " ")
    cleaned = " ".join(cleaned.split())  # remove extra spaces

    chunks = []
    start = 0
    end = chunk_size

    while start < len(cleaned):
        chunk = cleaned[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
        end = start + chunk_size

    return chunks


# ------------------- MAIN EXTRACTOR -------------------

def process_pdf(path: str, scheme_name: str):
    """Extract text from a single PDF and save pages + chunks."""
    doc = fitz.open(path)
    pages_text = []
    full_merged_text = ""

    print(f"\n🔍 Processing {scheme_name} ({len(doc)} pages)...")

    for i, page in enumerate(doc):
        text = page.get_text().strip()

        if page_needs_ocr(text):
            print(f"  • Page {i+1}: running OCR…")
            pix = page.get_pixmap(dpi=200)
            text = ocr_page(pix)

        else:
            print(f"  • Page {i+1}: digital extraction OK")

        pages_text.append(text)
        full_merged_text += "\n" + text

    # Save raw text
    txt_path = os.path.join(PROCESSED_DIR, f"{scheme_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_merged_text.strip())

    # Save page-wise JSON
    pages_path = os.path.join(PROCESSED_DIR, f"{scheme_name}_pages.json")
    with open(pages_path, "w", encoding="utf-8") as f:
        json.dump(pages_text, f, ensure_ascii=False, indent=2)

    # Save chunks JSON
    chunks = []
    for page_text in pages_text:
        chunks.extend(chunk_text(page_text))

    chunks_path = os.path.join(PROCESSED_DIR, f"{scheme_name}_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Completed: {scheme_name}")
    print(f"   → {txt_path}")
    print(f"   → {pages_path}")
    print(f"   → {chunks_path}")


# ------------------- RUN ALL SCHEMES -------------------

def main():
    print("📂 Starting PDF extraction…")
    pdf_files = [f for f in os.listdir(INGEST_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("❌ No PDFs found in ingested folder.")
        return

    for fname in pdf_files:
        scheme_name = os.path.splitext(fname)[0]
        full_path = os.path.join(INGEST_DIR, fname)
        process_pdf(full_path, scheme_name)

    print("\n🎉 Extraction finished for all PDFs!")


if __name__ == "__main__":
    main()