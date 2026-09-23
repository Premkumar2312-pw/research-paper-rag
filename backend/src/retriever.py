"""
retriever.py

Implements the retrieval pipeline:

    query
      -> (deterministic query expansion for short/vague questions)
      -> semantic search (embeddings via ChromaDB)      -----\
      -> BM25 lexical search (rank_bm25, over active docs) ----> merge + hybrid score
      -> top candidate pool
      -> cross-encoder re-ranking (optional, with automatic fallback)
      -> final top-K chunks passed to the LLM

A special fast path for metadata-style questions (title/author/abstract)
prioritizes the dedicated first-segment chunk built by chunker.py.

All retrieval is scoped to the currently "active" (selected) documents so
information from unselected papers never leaks into an answer.

Why BM25 instead of plain keyword overlap?
BM25 is a proper lexical ranking function that accounts for term frequency
saturation (a word appearing 10 times isn't 10x more relevant than once)
and document-length normalization (long chunks don't win just by containing
more words). Plain overlap counting has neither property, so BM25 gives
noticeably better lexical matches for exact terms (model names, dataset
names, acronyms) that embeddings alone sometimes miss.

Why cross-encoder re-ranking?
A bi-encoder (the embedding model) scores the query and each chunk
independently and compares vectors -- fast, but less precise. A
cross-encoder reads the (query, chunk) pair together and outputs a direct
relevance score -- slower, but far more accurate. Running it only on a
small shortlist (not the whole database) keeps it fast while still
improving final ranking quality.
"""

import re
from typing import List, Dict, Optional

from src import config
from src import vector_store
from src import query_expansion

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

_cross_encoder = None
_cross_encoder_load_failed = False

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "for",
    "to", "and", "or", "this", "that", "what", "which", "who", "how",
    "does", "do", "did", "with", "by", "as", "be", "it", "its", "their",
    "paper", "papers", "used", "use", "using",
}


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _semantic_score(distance: float) -> float:
    """Convert a cosine distance (0=identical, 2=opposite) into a 0-1 similarity score."""
    similarity = 1.0 - (distance / 2.0)
    return max(0.0, min(1.0, similarity))


def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def is_metadata_question(question: str) -> bool:
    """Detects title/author/abstract/keyword-style questions."""
    q = question.lower()
    return any(keyword in q for keyword in config.METADATA_QUESTION_KEYWORDS)


def _get_cross_encoder():
    """
    Lazily load the cross-encoder re-ranker. On any failure (no internet,
    disk space, incompatible environment) we set a flag and never try again
    this process -- callers fall back to plain hybrid ranking instead of
    crashing the application.
    """
    global _cross_encoder, _cross_encoder_load_failed
    if _cross_encoder is not None or _cross_encoder_load_failed:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL_NAME)
    except Exception:
        _cross_encoder_load_failed = True
        _cross_encoder = None
    return _cross_encoder


def _bm25_candidates(query_text: str, doc_ids: Optional[List[str]], top_k: int) -> Dict[str, Dict]:
    """Run BM25 over the full chunk corpus of the active documents and return top_k hits."""
    if not _BM25_AVAILABLE:
        return {}

    corpus_chunks = vector_store.get_chunks_for_doc_ids(doc_ids) if doc_ids else []
    if not corpus_chunks:
        return {}

    tokenized_corpus = [_tokenize(c["text"]) for c in corpus_chunks]
    if not any(tokenized_corpus):
        return {}

    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return {}

    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = {}
    for idx in ranked_indices:
        if scores[idx] <= 0:
            continue
        chunk = dict(corpus_chunks[idx])
        chunk["bm25_score"] = float(scores[idx])
        results[chunk["chunk_id"]] = chunk
    return results


def retrieve(question: str, doc_ids: Optional[List[str]] = None,
             top_k: int = None) -> List[Dict]:
    """
    Retrieve and rank the most relevant chunks for a question, scoped to
    `doc_ids` (the currently active/selected papers).

    Pipeline: metadata fast-path -> semantic + BM25 hybrid merge ->
    cross-encoder re-ranking (if available) -> top_k final chunks.

    Each returned chunk is augmented with a 'score' field (final ranking
    score) plus, where available, 'semantic_score', 'bm25_score', and
    'rerank_score' for transparency/debugging.
    """
    top_k = top_k or config.TOP_K_FINAL
    retrieval_query = query_expansion.expand_query(question)

    metadata_hits: List[Dict] = []
    if is_metadata_question(question):
        # Fast path: pull directly from the dedicated first-segment chunks first.
        raw_hits = vector_store.query(
            query_text=retrieval_query,
            doc_ids=doc_ids,
            top_k=len(doc_ids) if doc_ids else config.TOP_K_SEMANTIC,
            chunk_type=config.CHUNK_TYPE_METADATA,
        )
        for hit in raw_hits:
            hit["score"] = 1.0
            hit["semantic_score"] = _semantic_score(hit["distance"])
            metadata_hits.append(hit)

    metadata_ids = {m["chunk_id"] for m in metadata_hits}

    # --- Semantic candidates -------------------------------------------------
    semantic_hits = vector_store.query(
        query_text=retrieval_query,
        doc_ids=doc_ids,
        top_k=config.CANDIDATE_POOL_SIZE,
    )
    semantic_scores = {}
    semantic_by_id = {}
    for hit in semantic_hits:
        sem = _semantic_score(hit["distance"])
        semantic_scores[hit["chunk_id"]] = sem
        semantic_by_id[hit["chunk_id"]] = hit

    # --- BM25 lexical candidates ----------------------------------------------
    bm25_by_id = _bm25_candidates(retrieval_query, doc_ids, config.CANDIDATE_POOL_SIZE)
    bm25_raw_scores = {cid: c["bm25_score"] for cid, c in bm25_by_id.items()}

    # --- Merge + hybrid score (normalized 0-1 within this candidate pool) ----
    norm_semantic = _minmax_normalize(semantic_scores)
    norm_bm25 = _minmax_normalize(bm25_raw_scores)

    merged: Dict[str, Dict] = {}
    all_ids = (set(semantic_by_id) | set(bm25_by_id)) - metadata_ids
    for cid in all_ids:
        base = semantic_by_id.get(cid) or bm25_by_id.get(cid)
        chunk = dict(base)
        sem = norm_semantic.get(cid, 0.0)
        kw = norm_bm25.get(cid, 0.0)
        chunk["semantic_score"] = semantic_scores.get(cid, sem)
        chunk["bm25_score"] = bm25_raw_scores.get(cid, 0.0)
        chunk["score"] = (config.SEMANTIC_WEIGHT * sem) + (config.KEYWORD_WEIGHT * kw)
        merged[cid] = chunk

    hybrid_ranked = sorted(merged.values(), key=lambda h: h["score"], reverse=True)

    # --- Cross-encoder re-ranking on a shortlist only -------------------------
    candidate_pool = hybrid_ranked[:config.RERANK_CANDIDATE_COUNT]
    if config.ENABLE_CROSS_ENCODER_RERANK and candidate_pool:
        cross_encoder = _get_cross_encoder()
        if cross_encoder is not None:
            try:
                pairs = [(question, c["text"]) for c in candidate_pool]
                rerank_scores = cross_encoder.predict(pairs)
                for c, rscore in zip(candidate_pool, rerank_scores):
                    c["rerank_score"] = float(rscore)
                    c["score"] = float(rscore)  # cross-encoder score becomes the final ranking signal
                candidate_pool.sort(key=lambda c: c["score"], reverse=True)
            except Exception:
                # Cross-encoder inference failed at runtime -- keep the
                # hybrid ranking instead of crashing the request.
                pass

    final_ranked = candidate_pool if candidate_pool else hybrid_ranked
    combined = metadata_hits + [c for c in final_ranked if c["chunk_id"] not in metadata_ids]
    return combined[:top_k]


def has_any_indexed_content(doc_ids: Optional[List[str]] = None) -> bool:
    """Quick check used to give a clean error when retrieval has nothing to search."""
    hits = vector_store.query(query_text="overview", doc_ids=doc_ids, top_k=1)
    return len(hits) > 0
