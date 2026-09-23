"""
document_manager.py

Orchestrates the ingestion pipeline for a single uploaded file of any
supported format:
    bytes -> (temp file for PDF only) -> extract segments -> chunk -> embed & store

Also owns duplicate handling: re-processing the same file content
replaces its previous chunks rather than accumulating duplicates.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import List

from src import config
from src import document_loader
from src import chunker
from src import vector_store


class IngestionError(Exception):
    """User-facing error raised when a single file fails to process."""
    pass


@dataclass
class IngestionResult:
    doc_id: str
    filename: str
    file_type: str
    segment_count: int      # pages / slides / sections / row-groups, depending on format
    chunk_count: int
    ocr_segments_used: int  # non-zero only for scanned PDF pages


def process_uploaded_file(filename: str, file_bytes: bytes) -> IngestionResult:
    """
    Full ingestion pipeline for one uploaded file's raw bytes. Supports any
    extension in config.SUPPORTED_EXTENSIONS (PDF/DOCX/PPTX/TXT/MD/CSV/XLSX).
    Safe to call again for the same file: old chunks for that doc_id are
    deleted before the new ones are inserted.
    """
    if not file_bytes:
        raise IngestionError(f"'{filename}' is empty and cannot be processed.")

    file_type = document_loader.detect_file_type(filename)
    if not document_loader.is_supported(filename):
        supported = ", ".join(sorted(config.SUPPORTED_EXTENSIONS))
        raise IngestionError(
            f"'{filename}': unsupported file type '.{file_type}'. Supported: {supported}."
        )

    doc_id = vector_store.compute_doc_id(filename, file_bytes)

    tmp_path = None
    try:
        if file_type == config.FILE_TYPE_PDF:
            # PyMuPDF needs a real file path on disk, so PDFs alone go through
            # a temp file; every other format is read straight from memory.
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

        try:
            segments = document_loader.load_segments(filename, file_bytes, tmp_pdf_path=tmp_path)
        except document_loader.DocumentLoadError as exc:
            raise IngestionError(f"'{filename}': {exc}")

        chunks = chunker.chunk_document(segments, doc_id=doc_id, filename=filename, file_type=file_type)
        if not chunks:
            raise IngestionError(f"'{filename}' produced no usable text chunks.")

        # Duplicate handling: clear any previous version of this exact file first.
        vector_store.delete_document(doc_id)
        vector_store.add_chunks(chunks)

        ocr_segments = sum(1 for s in segments if s.get("method") == "ocr")

        return IngestionResult(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            segment_count=len(segments),
            chunk_count=len(chunks),
            ocr_segments_used=ocr_segments,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                # On Windows, some other process (e.g. antivirus scanning)
                # can briefly hold a lock on the temp file. This is harmless
                # cleanup litter in the OS temp folder, not a processing
                # failure, so we don't want it to surface as an error to
                # the user after the PDF has already been indexed.
                pass


# Backward-compatible alias: earlier versions of this project only supported
# PDFs and exposed this function name. Kept so any external callers/tests
# written against the old name keep working.
def process_uploaded_pdf(filename: str, file_bytes: bytes) -> IngestionResult:
    return process_uploaded_file(filename, file_bytes)


def remove_document(doc_id: str) -> None:
    """Remove a document's chunks entirely from the vector store."""
    vector_store.delete_document(doc_id)


def get_active_documents() -> List[dict]:
    """List every document currently indexed in the vector store."""
    return vector_store.list_active_documents()
