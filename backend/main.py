"""
main.py

FastAPI backend for the Research Paper Intelligence & Lightweight RAG System.
This replaces the previous Streamlit UI: the same underlying pipeline
(src/document_manager.py, src/retriever.py, src/rag.py, etc.) is now
exposed over HTTP so a separate React frontend can drive it.

Run with:
    uvicorn main:app --reload --port 8000
"""

from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import config
from src import document_manager
from src import rag

app = FastAPI(
    title="Research Paper Intelligence API",
    description="Lightweight RAG backend for research-paper Q&A, summarization, and comparison.",
    version="2.0.0",
)

# The React dev server runs on a different port than the API, so CORS must
# be enabled for local development. Adjust allow_origins for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",  # Vite default
        "http://localhost:3000", "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    doc_ids: List[str]
    top_k: Optional[int] = None  # lets the UI expose an adjustable retrieval depth


class SummarizeRequest(BaseModel):
    doc_id: str
    filename: str


# ---------------------------------------------------------------------------
# Health / config
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "groq_configured": config.is_groq_configured(),
        "supported_extensions": config.SUPPORTED_EXTENSIONS,
    }


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------
@app.get("/api/documents")
def list_documents():
    return {"documents": document_manager.get_active_documents()}


@app.post("/api/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    results = []
    errors = []

    for upload in files:
        try:
            file_bytes = await upload.read()
            result = document_manager.process_uploaded_file(upload.filename, file_bytes)
            results.append({
                "doc_id": result.doc_id,
                "filename": result.filename,
                "file_type": result.file_type,
                "segment_count": result.segment_count,
                "chunk_count": result.chunk_count,
                "ocr_segments_used": result.ocr_segments_used,
            })
        except document_manager.IngestionError as exc:
            errors.append({"filename": upload.filename, "error": str(exc)})
        except Exception as exc:
            errors.append({"filename": upload.filename, "error": f"Unexpected error: {exc}"})

    return {"processed": results, "errors": errors}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    document_manager.remove_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@app.delete("/api/documents")
def clear_all_documents():
    docs = document_manager.get_active_documents()
    for doc in docs:
        document_manager.remove_document(doc["doc_id"])
    return {"status": "cleared", "count": len(docs)}


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------
@app.post("/api/ask")
def ask_question(payload: AskRequest):
    try:
        result = rag.answer_question(payload.question, doc_ids=payload.doc_ids, top_k=payload.top_k)
    except rag.RAGError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "confidence": result["confidence"],
        "is_comparison": result["is_comparison"],
        "chunks_used": [
            {
                "filename": c["filename"],
                "location_label": c.get("location_label"),
                "text": c["text"],
                "score": c.get("score"),
                "semantic_score": c.get("semantic_score"),
                "bm25_score": c.get("bm25_score"),
                "rerank_score": c.get("rerank_score"),
            }
            for c in result.get("chunks_used", [])
        ],
    }


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------
@app.post("/api/summarize")
def summarize_document(payload: SummarizeRequest):
    try:
        result = rag.summarize_paper(payload.doc_id, payload.filename)
    except rag.RAGError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

    return {"summary": result["summary"], "citations": result["citations"]}
