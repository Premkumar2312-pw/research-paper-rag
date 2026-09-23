"""
document_loader.py

Format-detection and extraction layer. Every supported file type is
converted into the SAME internal representation -- a list of "segments" --
so the rest of the pipeline (chunker, vector_store, retriever, rag) never
needs per-format logic. This is the abstraction requested for multi-format
support: one common representation in, one RAG pipeline out.

A segment looks like:
    {
        "position": 1,                 # 1-indexed ordering within the doc
        "text": "....",
        "method": "text" | "ocr" | "native",
        "location_label": "Page 3" | "Slide 2" | "Section 1" | "Rows 1-25",
    }

Supported formats:
    Required:  PDF, DOCX, TXT, MD, PPTX
    Optional:  XLSX, CSV

PDF extraction (including OCR and column-aware layout handling) is left
entirely to pdf_loader.py -- this module only adapts its output into the
common segment shape.
"""

import csv
import io
from pathlib import Path
from typing import List, Dict

from src import config
from src import pdf_loader


class DocumentLoadError(Exception):
    """User-facing error raised when a file of any supported format can't be read."""
    pass


def detect_file_type(filename: str) -> str:
    """Return the lowercase extension (without the dot), e.g. 'pdf', 'docx'."""
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext


def is_supported(filename: str) -> bool:
    return detect_file_type(filename) in config.SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# PDF (delegates to pdf_loader.py, which already handles OCR + columns)
# ---------------------------------------------------------------------------
def _load_pdf(file_path: str) -> List[Dict]:
    pages = pdf_loader.extract_pages(file_path)
    segments = []
    for page in pages:
        segments.append({
            "position": page["page_number"],
            "text": page["text"],
            "method": page["method"],
            "location_label": f"Page {page['page_number']}",
        })
    return segments


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------
def _load_docx(file_bytes: bytes) -> List[Dict]:
    try:
        import docx  # python-docx
    except ImportError:
        raise DocumentLoadError(
            "python-docx is not installed. Run: pip install python-docx"
        )

    try:
        document = docx.Document(io.BytesIO(file_bytes))
    except Exception as exc:
        raise DocumentLoadError(f"Could not open DOCX file: {exc}")

    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    if not paragraphs:
        raise DocumentLoadError("No readable text found in this DOCX file.")

    segments = []
    group_size = config.DOCX_PARAGRAPHS_PER_SEGMENT
    for i in range(0, len(paragraphs), group_size):
        group = paragraphs[i:i + group_size]
        section_number = (i // group_size) + 1
        segments.append({
            "position": section_number,
            "text": "\n".join(group),
            "method": "native",
            "location_label": f"Section {section_number}",
        })
    return segments


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------
def _load_pptx(file_bytes: bytes) -> List[Dict]:
    try:
        from pptx import Presentation
    except ImportError:
        raise DocumentLoadError(
            "python-pptx is not installed. Run: pip install python-pptx"
        )

    try:
        presentation = Presentation(io.BytesIO(file_bytes))
    except Exception as exc:
        raise DocumentLoadError(f"Could not open PPTX file: {exc}")

    segments = []
    for i, slide in enumerate(presentation.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                texts.append(shape.text_frame.text.strip())
            if shape.has_table:
                for row in shape.table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip(" |"):
                        texts.append(row_text)
        slide_text = "\n".join(texts).strip()
        if slide_text:
            segments.append({
                "position": i,
                "text": slide_text,
                "method": "native",
                "location_label": f"Slide {i}",
            })

    if not segments:
        raise DocumentLoadError("No readable text found in this PPTX file.")
    return segments


# ---------------------------------------------------------------------------
# TXT / Markdown
# ---------------------------------------------------------------------------
def _load_plaintext(file_bytes: bytes) -> List[Dict]:
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1", errors="ignore")

    text = text.strip()
    if not text:
        raise DocumentLoadError("This file is empty.")

    chunk_size = config.PLAINTEXT_CHARS_PER_SEGMENT
    segments = []
    for i in range(0, len(text), chunk_size):
        section_number = (i // chunk_size) + 1
        piece = text[i:i + chunk_size].strip()
        if not piece:
            continue
        segments.append({
            "position": section_number,
            "text": piece,
            "method": "native",
            "location_label": f"Section {section_number}",
        })
    return segments


# ---------------------------------------------------------------------------
# CSV (optional)
# ---------------------------------------------------------------------------
def _load_csv(file_bytes: bytes) -> List[Dict]:
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1", errors="ignore")

    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise DocumentLoadError("This CSV file has no rows.")

    header = rows[0]
    data_rows = rows[1:] if len(rows) > 1 else []

    segments = []
    group_size = config.SPREADSHEET_ROWS_PER_SEGMENT
    if not data_rows:
        # Header-only CSV: still index the header so it's queryable.
        segments.append({
            "position": 1,
            "text": ", ".join(header),
            "method": "native",
            "location_label": "Header row",
        })
        return segments

    for i in range(0, len(data_rows), group_size):
        group = data_rows[i:i + group_size]
        start_row, end_row = i + 1, i + len(group)
        lines = [", ".join(header)] + [", ".join(row) for row in group]
        segments.append({
            "position": (i // group_size) + 1,
            "text": "\n".join(lines),
            "method": "native",
            "location_label": f"Rows {start_row}-{end_row}",
        })
    return segments


# ---------------------------------------------------------------------------
# XLSX (optional)
# ---------------------------------------------------------------------------
def _load_xlsx(file_bytes: bytes) -> List[Dict]:
    try:
        import openpyxl
    except ImportError:
        raise DocumentLoadError(
            "openpyxl is not installed. Run: pip install openpyxl"
        )

    try:
        workbook = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
    except Exception as exc:
        raise DocumentLoadError(f"Could not open XLSX file: {exc}")

    segments = []
    position = 0
    group_size = config.SPREADSHEET_ROWS_PER_SEGMENT

    for sheet in workbook.worksheets:
        rows = [
            [str(cell) if cell is not None else "" for cell in row]
            for row in sheet.iter_rows(values_only=True)
        ]
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        if not rows:
            continue

        header = rows[0]
        data_rows = rows[1:]
        if not data_rows:
            data_rows = [header]
            header = []

        for i in range(0, len(data_rows), group_size):
            group = data_rows[i:i + group_size]
            position += 1
            lines = ([", ".join(header)] if header else []) + [", ".join(r) for r in group]
            segments.append({
                "position": position,
                "text": "\n".join(lines),
                "method": "native",
                "location_label": f"Sheet '{sheet.title}' rows {i + 1}-{i + len(group)}",
            })

    if not segments:
        raise DocumentLoadError("No readable data found in this XLSX file.")
    return segments


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------
_LOADERS_NEEDING_BYTES = {
    config.FILE_TYPE_DOCX: _load_docx,
    config.FILE_TYPE_PPTX: _load_pptx,
    config.FILE_TYPE_TXT: _load_plaintext,
    config.FILE_TYPE_MD: _load_plaintext,
    config.FILE_TYPE_CSV: _load_csv,
    config.FILE_TYPE_XLSX: _load_xlsx,
}


def load_segments(filename: str, file_bytes: bytes, tmp_pdf_path: str = None) -> List[Dict]:
    """
    Dispatch to the correct extractor based on file extension and return
    a list of common-shape segments (see module docstring).

    `tmp_pdf_path` must be provided when the file type is PDF (PyMuPDF
    needs a real file path); document_manager.py is responsible for
    writing the uploaded bytes to a temp file first for that case.
    """
    file_type = detect_file_type(filename)

    if file_type not in config.SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(config.SUPPORTED_EXTENSIONS))
        raise DocumentLoadError(
            f"Unsupported file type '.{file_type}'. Supported types: {supported}."
        )

    if file_type == config.FILE_TYPE_PDF:
        if not tmp_pdf_path:
            raise DocumentLoadError("Internal error: PDF processing requires a temp file path.")
        return _load_pdf(tmp_pdf_path)

    loader_fn = _LOADERS_NEEDING_BYTES.get(file_type)
    if loader_fn is None:
        raise DocumentLoadError(f"No loader implemented for file type '.{file_type}'.")
    return loader_fn(file_bytes)
