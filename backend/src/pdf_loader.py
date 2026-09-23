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


def _extract_text_column_aware(page: "fitz.Page") -> str:
    """
    Extract page text in a column-aware reading order using block-level
    extraction. Plain `page.get_text("text")` reads left-to-right across
    the whole page width, which garbles two-column academic layouts (it
    interleaves lines from the left and right columns). Instead, we take
    each text block's bounding box, assign it to a column based on which
    half of the page width it starts in, then read column-by-column,
    top-to-bottom within each column. This is a lightweight heuristic
    (not full layout analysis) but works well for standard single- and
    two-column research-paper layouts.
    """
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    text_blocks = [b for b in blocks if len(b) >= 5 and b[4].strip()]
    if not text_blocks:
        return ""

    page_width = page.rect.width
    midpoint = page_width / 2.0

    left_column = [b for b in text_blocks if b[0] < midpoint]
    right_column = [b for b in text_blocks if b[0] >= midpoint]

    # If almost everything landed in one "column", the page is single-column
    # (or a wide table/figure) -- just sort by vertical position as normal.
    if not right_column or len(right_column) < 2:
        ordered = sorted(text_blocks, key=lambda b: (b[1], b[0]))
    else:
        left_column.sort(key=lambda b: b[1])   # top-to-bottom by y0
        right_column.sort(key=lambda b: b[1])
        ordered = left_column + right_column

    return "\n".join(b[4] for b in ordered)


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

    # Everything from here on must guarantee doc.close() runs, even on error,
    # or the underlying file handle stays open (this caused "WinError 32:
    # process cannot access the file" on Windows when a temp file was
    # deleted right after a failed extraction).
    try:
        if doc.page_count == 0:
            raise PDFLoadError("The PDF has no pages.")

        pages: List[Dict] = []
        for i in range(doc.page_count):
            page = doc[i]

            # 1. Try column-aware block extraction first (handles two-column
            #    research-paper layouts correctly). Fall back to plain
            #    extraction if block extraction yields nothing usable.
            raw_text = _extract_text_column_aware(page)
            if not raw_text.strip():
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

        total_chars = sum(len(p["text"]) for p in pages)
        if total_chars == 0:
            raise PDFLoadError(
                "No readable text could be extracted from this PDF, even after OCR."
            )

        return pages
    finally:
        doc.close()


def get_first_page_text(pages: List[Dict]) -> str:
    """Convenience accessor used by the chunker to build the metadata chunk."""
    if not pages:
        return ""
    return pages[0]["text"]
