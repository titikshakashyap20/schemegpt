# ai/extract.py
import os
import json
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from typing import List

# Set your Tesseract path on Windows (adjust if different)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

INGEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "data", "ingested"))
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "data", "processed"))
os.makedirs(OUT_DIR, exist_ok=True)

def ocr_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)

def text_from_pdf_with_pdfplumber(path: str) -> List[str]:
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                txt = p.extract_text() or ""
                pages.append(txt.strip())
    except Exception as e:
        print(f"[pdfplumber] error for {path}: {e}")
    return pages

def ocr_pages_with_pymupdf(path: str) -> List[str]:
    pages = []
    try:
        doc = fitz.open(path)
        for page in doc:
            pix = page.get_pixmap(dpi=200)  # increase dpi for better OCR
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            txt = ocr_image(img)
            pages.append(txt.strip())
    except Exception as e:
        print(f"[pymupdf OCR] error for {path}: {e}")
    return pages

def extract_text(path: str) -> List[str]:
    """
    Return list of page-level strings. Strategy:
      1. Try pdfplumber extraction.
      2. If a page's text is short (e.g., < 50 chars), use OCR on that page via PyMuPDF->PIL->pytesseract.
      3. If pdfplumber fails entirely, fallback to OCR all pages.
    """
    pages = text_from_pdf_with_pdfplumber(path)
    if not pages:
        # full OCR fallback
        print(f"[extract_text] pdfplumber returned nothing for {path}, falling back to full OCR")
        return ocr_pages_with_pymupdf(path)

    # For pages that look empty or very short, run OCR
    final_pages = []
    try:
        doc = fitz.open(path)
        for i, ptext in enumerate(pages):
            if ptext and len(ptext.strip()) > 60:
                final_pages.append(ptext.strip())
            else:
                # OCR this page
                page = doc[i]
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                ocr_text = ocr_image(img).strip()
                # prefer OCR text if it's meaningfully longer
                chosen = ocr_text if len(ocr_text) > len(ptext) else ptext
                final_pages.append(chosen.strip())
    except Exception as e:
        print(f"[extract_text] Mixed extraction error, falling back to OCR for all pages: {e}")
        return ocr_pages_with_pymupdf(path)

    return final_pages

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """Simple character-based chunking with overlap."""
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + max_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, end - overlap)
    return chunks

def process_pdf_file(pdf_path: str):
    name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"[process_pdf_file] processing {name}")
    pages = extract_text(pdf_path)
    full_text = "\n\n".join([p for p in pages if p])

    os.makedirs(OUT_DIR, exist_ok=True)
    txt_path = os.path.join(OUT_DIR, f"{name}.txt")
    pages_path = os.path.join(OUT_DIR, f"{name}_pages.json")
    chunks_path = os.path.join(OUT_DIR, f"{name}_chunks.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    with open(pages_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    chunks = chunk_text(full_text)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[process_pdf_file] wrote: {txt_path}, {pages_path}, {chunks_path}")

def process_all_ingested():
    files = [f for f in os.listdir(INGEST_DIR) if f.lower().endswith(".pdf")]
    if not files:
        print("[process_all_ingested] No PDFs found in", INGEST_DIR)
        return
    for f in files:
        process_pdf_file(os.path.join(INGEST_DIR, f))

if __name__ == "__main__":
    process_all_ingested()
