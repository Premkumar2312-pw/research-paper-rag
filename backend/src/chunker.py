"""
chunker.py

Turns per-segment text (from pdf_loader or document_loader) into overlapping
text chunks that carry location and document metadata all the way through
to retrieval. A "segment" is a format-agnostic unit of a document: a PDF
page, a PPTX slide, a DOCX section, a plain-text block, or a spreadsheet
row-group. Every segment has the same shape, so the chunker (and everything
downstream of it) doesn't need per-format logic:

    {
        "position": 1,                 # 1-indexed ordering within the doc
        "text": "....",
        "method": "text" | "ocr" | "native",
        "location_label": "Page 3" | "Slide 2" | "Section 1" | ...
    }

Two kinds of chunks are produced:
  1. "first_page_metadata" - one special chunk built from the first segment
     only. This exists so that title/author/abstract questions don't depend
     on the generic chunk boundaries landing correctly. Skipped for
     spreadsheet-style file types (see config.METADATA_ELIGIBLE_FILE_TYPES).
  2. "body" - normal recursively-split chunks covering every segment,
     including the first, so first-segment content is still searchable normally.
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
                 chunk_type: str, location_label: str, file_type: str) -> Dict:
    return {
        "chunk_id": f"{doc_id}_{uuid.uuid4().hex[:8]}",
        "doc_id": doc_id,
        "filename": filename,
        "page_number": page_number,       # kept for backward compatibility / sorting
        "location_label": location_label, # human-readable citation label
        "file_type": file_type,
        "chunk_type": chunk_type,
        "text": text,
    }


def build_metadata_chunk(segments: List[Dict], doc_id: str, filename: str,
                          file_type: str) -> List[Dict]:
    """
    Build a dedicated first-segment representation used to answer title/
    author/abstract/keyword questions without relying on generic semantic
    chunking. Skipped for spreadsheet-style formats (CSV/XLSX), which are
    data files rather than papers.
    """
    if not segments or file_type not in config.METADATA_ELIGIBLE_FILE_TYPES:
        return []

    first_segment = segments[0]
    first_text = first_segment["text"]
    if not first_text.strip():
        return []

    # Keep this as a single chunk (not re-split) so the whole first segment's
    # layout (title, authors, affiliations, abstract) stays together.
    # If it's unusually long, cap it generously rather than discarding content.
    max_len = config.CHUNK_SIZE * 3
    text = first_text[:max_len]

    return [_make_chunk(
        text=text,
        doc_id=doc_id,
        filename=filename,
        page_number=first_segment.get("position", 1),
        chunk_type=config.CHUNK_TYPE_METADATA,
        location_label=first_segment.get("location_label", "Section 1"),
        file_type=file_type,
    )]


def build_body_chunks(segments: List[Dict], doc_id: str, filename: str,
                       file_type: str) -> List[Dict]:
    """Recursively chunk every segment's text into overlapping body chunks."""
    all_chunks: List[Dict] = []

    for segment in segments:
        segment_text = segment["text"]
        if not segment_text.strip():
            continue

        raw_pieces = _split_text(segment_text, config.CHUNK_SIZE, _SEPARATORS)
        pieces = _add_overlap(raw_pieces, config.CHUNK_OVERLAP)

        for piece in pieces:
            if not piece.strip():
                continue
            all_chunks.append(_make_chunk(
                text=piece,
                doc_id=doc_id,
                filename=filename,
                page_number=segment.get("position", 1),
                chunk_type=config.CHUNK_TYPE_BODY,
                location_label=segment.get("location_label", "Section 1"),
                file_type=file_type,
            ))

    return all_chunks


def chunk_document(segments: List[Dict], doc_id: str, filename: str,
                    file_type: str = "pdf") -> List[Dict]:
    """
    Full chunking pipeline for one document: first-segment metadata chunk(s)
    plus normal recursive body chunks across all segments. `file_type`
    controls whether a metadata chunk is built at all (skipped for
    spreadsheet formats) and is stored on every chunk for citation display.
    """
    metadata_chunks = build_metadata_chunk(segments, doc_id, filename, file_type)
    body_chunks = build_body_chunks(segments, doc_id, filename, file_type)
    return metadata_chunks + body_chunks
