"""
chunker.py

Turns per-page text (from pdf_loader) into overlapping text chunks that
carry page-number and document metadata all the way through to retrieval.

Two kinds of chunks are produced:
  1. "first_page_metadata" - one (or two) special chunk(s) built from the
     first page only. This exists so that title/author/abstract questions
     don't depend on the generic chunk boundaries landing correctly.
  2. "body" - normal recursively-split chunks covering every page,
     including page 1, so first-page content is still searchable normally.
"""

import re
import uuid
from typing import List, Dict

from src import config

# Separators tried in order, largest structural unit first. This is the
# "recursive" part: if a piece is still too big after splitting on a
# separator, we recursively split it further down the list.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_text(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """Recursively split `text` on the given separators until pieces fit chunk_size."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # No separators left: hard-split as a last resort.
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest_seps = separators[0], separators[1:]
    parts = text.split(sep)

    chunks: List[str] = []
    current = ""
    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                # This single part is still too big; recurse with finer separators.
                chunks.extend(_split_text(part, chunk_size, rest_seps))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    """Prepend a tail slice of the previous chunk to each chunk for context continuity."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        overlapped.append((prev_tail + " " + chunks[i]).strip())
    return overlapped


def _make_chunk(text: str, doc_id: str, filename: str, page_number: int,
                 chunk_type: str) -> Dict:
    return {
        "chunk_id": f"{doc_id}_{uuid.uuid4().hex[:8]}",
        "doc_id": doc_id,
        "filename": filename,
        "page_number": page_number,
        "chunk_type": chunk_type,
        "text": text,
    }


def build_metadata_chunk(pages: List[Dict], doc_id: str, filename: str) -> List[Dict]:
    """
    Build a dedicated first-page representation used to answer title/author/
    abstract/keyword questions without relying on generic semantic chunking.
    """
    if not pages:
        return []

    first_page_text = pages[0]["text"]
    if not first_page_text.strip():
        return []

    # Keep this as a single chunk (not re-split) so the whole first page's
    # layout (title, authors, affiliations, abstract) stays together.
    # If the first page is unusually long, cap it generously rather than
    # discarding content.
    max_len = config.CHUNK_SIZE * 3
    text = first_page_text[:max_len]

    return [_make_chunk(
        text=text,
        doc_id=doc_id,
        filename=filename,
        page_number=1,
        chunk_type=config.CHUNK_TYPE_METADATA,
    )]


def build_body_chunks(pages: List[Dict], doc_id: str, filename: str) -> List[Dict]:
    """Recursively chunk every page's text into overlapping body chunks."""
    all_chunks: List[Dict] = []

    for page in pages:
        page_text = page["text"]
        if not page_text.strip():
            continue

        raw_pieces = _split_text(page_text, config.CHUNK_SIZE, _SEPARATORS)
        pieces = _add_overlap(raw_pieces, config.CHUNK_OVERLAP)

        for piece in pieces:
            if not piece.strip():
                continue
            all_chunks.append(_make_chunk(
                text=piece,
                doc_id=doc_id,
                filename=filename,
                page_number=page["page_number"],
                chunk_type=config.CHUNK_TYPE_BODY,
            ))

    return all_chunks


def chunk_document(pages: List[Dict], doc_id: str, filename: str) -> List[Dict]:
    """
    Full chunking pipeline for one document: first-page metadata chunk(s)
    plus normal recursive body chunks across all pages.
    """
    metadata_chunks = build_metadata_chunk(pages, doc_id, filename)
    body_chunks = build_body_chunks(pages, doc_id, filename)
    return metadata_chunks + body_chunks
