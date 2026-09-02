"""
retriever.py

Implements the hybrid retrieval strategy:
  - Semantic similarity search via ChromaDB/embeddings (vector_store.query)
  - Lightweight keyword/lexical overlap scoring on the semantic candidates
  - A combined score used to rank and select the final context chunks
  - A special fast path for metadata-style questions (title/author/abstract)
    that prioritizes the dedicated first-page chunk

All retrieval is scoped to the currently "active" (selected) documents so
information from unselected papers never leaks into an answer.
"""

import re
from typing import List, Dict, Optional

from src import config
from src import vector_store

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "for",
    "to", "and", "or", "this", "that", "what", "which", "who", "how",
    "does", "do", "did", "with", "by", "as", "be", "it", "its", "their",
    "paper", "papers", "used", "use", "using",
}


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _keyword_score(query_tokens: List[str], text: str) -> float:
    """Simple normalized keyword-overlap score between 0 and 1."""
    if not query_tokens:
        return 0.0
    text_tokens = set(_tokenize(text))
    if not text_tokens:
        return 0.0
    overlap = sum(1 for t in query_tokens if t in text_tokens)
    return overlap / len(query_tokens)


def _semantic_score(distance: float) -> float:
    """Convert a cosine distance (0=identical, 2=opposite) into a 0-1 similarity score."""
    similarity = 1.0 - (distance / 2.0)
    return max(0.0, min(1.0, similarity))


def is_metadata_question(question: str) -> bool:
    """Detects title/author/abstract/keyword-style questions."""
    q = question.lower()
    return any(keyword in q for keyword in config.METADATA_QUESTION_KEYWORDS)


def retrieve(question: str, doc_ids: Optional[List[str]] = None,
             top_k: int = None) -> List[Dict]:
    """
    Retrieve and rank the most relevant chunks for a question, scoped to
    `doc_ids` (the currently active/selected papers).

    Returns a list of chunk dicts sorted by combined relevance score,
    each augmented with a 'score' field.
    """
    top_k = top_k or config.TOP_K_FINAL
    query_tokens = _tokenize(question)

    metadata_hits: List[Dict] = []
    if is_metadata_question(question):
        # Fast path: pull directly from the dedicated first-page chunks first.
        metadata_hits = vector_store.query(
            query_text=question,
            doc_ids=doc_ids,
            top_k=len(doc_ids) if doc_ids else config.TOP_K_SEMANTIC,
            chunk_type=config.CHUNK_TYPE_METADATA,
        )
        for hit in metadata_hits:
            hit["score"] = 1.0  # Metadata chunks are authoritative for these questions

    # Always also run normal semantic retrieval over body chunks so that
    # supporting details (e.g. "who is the corresponding author, and why
    # were they chosen") still have body context available if needed.
    semantic_hits = vector_store.query(
        query_text=question,
        doc_ids=doc_ids,
        top_k=config.TOP_K_SEMANTIC,
    )

    scored: Dict[str, Dict] = {}
    for hit in metadata_hits:
        scored[hit["chunk_id"]] = hit

    for hit in semantic_hits:
        if hit["chunk_id"] in scored:
            continue
        sem_score = _semantic_score(hit["distance"])
        kw_score = _keyword_score(query_tokens, hit["text"])
        combined = (config.SEMANTIC_WEIGHT * sem_score) + (config.KEYWORD_WEIGHT * kw_score)
        hit["score"] = combined
        scored[hit["chunk_id"]] = hit

    ranked = sorted(scored.values(), key=lambda h: h["score"], reverse=True)
    return ranked[:top_k]


def has_any_indexed_content(doc_ids: Optional[List[str]] = None) -> bool:
    """Quick check used to give a clean error when retrieval has nothing to search."""
    hits = vector_store.query(query_text="overview", doc_ids=doc_ids, top_k=1)
    return len(hits) > 0
