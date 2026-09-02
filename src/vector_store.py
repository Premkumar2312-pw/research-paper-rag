"""
vector_store.py

Thin wrapper around a persistent ChromaDB collection plus a local
sentence-transformers embedding model. Handles:
  - embedding chunk text
  - inserting chunks with full metadata (filename, page_number, doc_id, chunk_type)
  - re-processing the same file without creating duplicate/stale chunks
  - metadata-filtered similarity search for document isolation
"""

import hashlib
from typing import List, Dict, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from src import config

_embedding_model: Optional[SentenceTransformer] = None
_chroma_client = None
_collection = None


def get_embedding_model() -> SentenceTransformer:
    """Lazily load the local embedding model (loaded once per process)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_collection():
    """Lazily create/open the persistent Chroma collection."""
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = _chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def compute_doc_id(filename: str, file_bytes: bytes) -> str:
    """
    Stable document identifier based on filename + content hash, so that
    re-uploading the *same* file yields the same doc_id (enabling clean
    replacement) while a *different* file with the same name gets a new id.
    """
    hasher = hashlib.sha256()
    hasher.update(filename.encode("utf-8"))
    hasher.update(file_bytes)
    return hasher.hexdigest()[:16]


def delete_document(doc_id: str) -> None:
    """Remove all chunks belonging to a document (used before re-indexing)."""
    collection = get_collection()
    try:
        collection.delete(where={"doc_id": doc_id})
    except Exception:
        # Nothing to delete / collection empty - safe to ignore.
        pass


def add_chunks(chunks: List[Dict]) -> None:
    """Embed and insert a list of chunk dicts (see chunker.py for shape)."""
    if not chunks:
        return

    collection = get_collection()
    model = get_embedding_model()

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{
        "doc_id": c["doc_id"],
        "filename": c["filename"],
        "page_number": c["page_number"],
        "chunk_type": c["chunk_type"],
    } for c in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )


def list_active_documents() -> List[Dict]:
    """
    Return a de-duplicated list of {doc_id, filename} for everything
    currently stored in the vector database.
    """
    collection = get_collection()
    result = collection.get(include=["metadatas"])
    seen = {}
    for meta in result.get("metadatas", []):
        if meta and meta.get("doc_id") not in seen:
            seen[meta["doc_id"]] = meta.get("filename")
    return [{"doc_id": doc_id, "filename": name} for doc_id, name in seen.items()]


def query(query_text: str, doc_ids: Optional[List[str]] = None,
          top_k: int = None, chunk_type: Optional[str] = None) -> List[Dict]:
    """
    Semantic similarity search, optionally restricted to a set of doc_ids
    (this is how per-paper isolation is enforced) and/or a chunk_type
    (used for the metadata-question fast path).
    """
    collection = get_collection()
    model = get_embedding_model()
    top_k = top_k or config.TOP_K_SEMANTIC

    query_embedding = model.encode([query_text]).tolist()

    where_clause = None
    conditions = []
    if doc_ids:
        conditions.append({"doc_id": {"$in": doc_ids}})
    if chunk_type:
        conditions.append({"chunk_type": chunk_type})

    if len(conditions) == 1:
        where_clause = conditions[0]
    elif len(conditions) > 1:
        where_clause = {"$and": conditions}

    # Guard against asking for more results than exist in a fresh/small collection.
    try:
        count = collection.count()
    except Exception:
        count = top_k
    effective_k = max(1, min(top_k, count)) if count else 0
    if effective_k == 0:
        return []

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=effective_k,
        where=where_clause,
    )

    hits = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for i in range(len(ids)):
        hits.append({
            "chunk_id": ids[i],
            "text": docs[i],
            "doc_id": metas[i].get("doc_id"),
            "filename": metas[i].get("filename"),
            "page_number": metas[i].get("page_number"),
            "chunk_type": metas[i].get("chunk_type"),
            "distance": dists[i],
        })
    return hits


def get_all_chunks_for_doc(doc_id: str) -> List[Dict]:
    """Fetch every chunk for a single document (used for whole-paper summarization)."""
    collection = get_collection()
    result = collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
    chunks = []
    for i in range(len(result.get("ids", []))):
        meta = result["metadatas"][i]
        chunks.append({
            "chunk_id": result["ids"][i],
            "text": result["documents"][i],
            "doc_id": meta.get("doc_id"),
            "filename": meta.get("filename"),
            "page_number": meta.get("page_number"),
            "chunk_type": meta.get("chunk_type"),
        })
    # Sort by page number then keep metadata chunk first for readability.
    chunks.sort(key=lambda c: (c["chunk_type"] != config.CHUNK_TYPE_METADATA, c["page_number"]))
    return chunks
