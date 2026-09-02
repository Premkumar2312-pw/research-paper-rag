"""
document_manager.py

Orchestrates the ingestion pipeline for a single uploaded PDF:
    bytes -> temp file -> extract pages -> chunk -> embed & store

Also owns duplicate handling: re-processing the same file content
replaces its previous chunks rather than accumulating duplicates.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import List

from src import config
from src import pdf_loader
from src import chunker
from src import vector_store


class IngestionError(Exception):
    """User-facing error raised when a single PDF fails to process."""
    pass


@dataclass
class IngestionResult:
    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    ocr_pages_used: int


def process_uploaded_pdf(filename: str, file_bytes: bytes) -> IngestionResult:
    """
    Full ingestion pipeline for one uploaded PDF's raw bytes.
    Safe to call again for the same file: old chunks for that doc_id
    are deleted before the new ones are inserted.
    """
    if not file_bytes:
        raise IngestionError(f"'{filename}' is empty and cannot be processed.")

    doc_id = vector_store.compute_doc_id(filename, file_bytes)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            pages = pdf_loader.extract_pages(tmp_path)
        except pdf_loader.PDFLoadError as exc:
            raise IngestionError(f"'{filename}': {exc}")

        chunks = chunker.chunk_document(pages, doc_id=doc_id, filename=filename)
        if not chunks:
            raise IngestionError(f"'{filename}' produced no usable text chunks.")

        # Duplicate handling: clear any previous version of this exact file first.
        vector_store.delete_document(doc_id)
        vector_store.add_chunks(chunks)

        ocr_pages = sum(1 for p in pages if p["method"] == "ocr")

        return IngestionResult(
            doc_id=doc_id,
            filename=filename,
            page_count=len(pages),
            chunk_count=len(chunks),
            ocr_pages_used=ocr_pages,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def remove_document(doc_id: str) -> None:
    """Remove a document's chunks entirely from the vector store."""
    vector_store.delete_document(doc_id)


def get_active_documents() -> List[dict]:
    """List every document currently indexed in the vector store."""
    return vector_store.list_active_documents()
