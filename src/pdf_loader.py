"""
pdf_loader.py

Responsible for turning a PDF file on disk into a list of per-page text
records. Handles both normal (text-based) PDFs and scanned/image-only
PDFs by automatically falling back to OCR on a per-page basis.

Each returned page record looks like:
    {
        "page_number": 1,          # 1-indexed
        "text": "....",            # cleaned page text
        "method": "text" | "ocr",  # how the text was obtained
    }
"""

from io import BytesIO
from typing import List, Dict

import pymupdf as fitz  # PyMuPDF (new import name; 'fitz' alias kept for readability)
from PIL import Image
import pytesseract

from src import config


class PDFLoadError(Exception):
    """Raised when a PDF cannot be opened or processed at all."""
    pass


def _clean_text(text: str) -> str:
    """Light, safe text cleanup: collapse whitespace, strip stray control chars."""
    if not text:
        return ""
    # Normalize line endings and collapse excessive blank lines/spaces.
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line != ""]
    cleaned = "\n".join(lines)
    # Collapse runs of spaces that sometimes appear from PDF extraction artifacts.
    cleaned = " ".join(cleaned.split(" "))
    return cleaned.strip()


def _ocr_page(page: "fitz.Page") -> str:
    """Rasterize a PDF page and run Tesseract OCR on it."""
    matrix = fitz.Matrix(config.OCR_RENDER_ZOOM, config.OCR_RENDER_ZOOM)
    pix = page.get_pixmap(matrix=matrix)
    image = Image.open(BytesIO(pix.tobytes("png")))
    try:
        text = pytesseract.image_to_string(image)
    except Exception as exc:  # OCR engine missing / failed
        raise PDFLoadError(
            f"OCR failed on page {page.number + 1}: {exc}. "
            "Ensure Tesseract is installed on this machine."
        )
    return text


def extract_pages(file_path: str) -> List[Dict]:
    """
    Extract text from every page of a PDF, automatically using OCR for
    pages that contain no usable embedded text (i.e. scanned pages).

    Raises PDFLoadError for unreadable/corrupted/empty files.
    """
    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        raise PDFLoadError(f"Could not open PDF: {exc}")

    if doc.page_count == 0:
        doc.close()
        raise PDFLoadError("The PDF has no pages.")

    pages: List[Dict] = []
    for i in range(doc.page_count):
        page = doc[i]

        # 1. Try normal text extraction first.
        raw_text = page.get_text("text") or ""
        cleaned = _clean_text(raw_text)

        method = "text"
        # 2. If the page has effectively no usable text, fall back to OCR.
        if len(cleaned) < config.MIN_TEXT_LENGTH_FOR_NO_OCR:
            ocr_text = _ocr_page(page)
            ocr_cleaned = _clean_text(ocr_text)
            if len(ocr_cleaned) > len(cleaned):
                cleaned = ocr_cleaned
                method = "ocr"

        pages.append({
            "page_number": i + 1,
            "text": cleaned,
            "method": method,
        })

    doc.close()

    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars == 0:
        raise PDFLoadError(
            "No readable text could be extracted from this PDF, even after OCR."
        )

    return pages


def get_first_page_text(pages: List[Dict]) -> str:
    """Convenience accessor used by the chunker to build the metadata chunk."""
    if not pages:
        return ""
    return pages[0]["text"]
