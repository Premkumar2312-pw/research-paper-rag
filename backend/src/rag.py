"""
rag.py

Ties retrieval to generation. Builds a strictly-grounded prompt from
retrieved chunks, calls the Groq LLM (with lightweight retry/backoff for
transient failures), and formats citations from the actual chunks used.

The LLM is explicitly instructed to answer ONLY from the provided context
and to say so clearly when the answer isn't present, so the system never
silently falls back on outside/parametric knowledge.
"""

import time
from typing import List, Dict, Optional

from groq import Groq

from src import config
from src import retriever
from src import vector_store

NOT_FOUND_MESSAGE = "I could not find this information in the uploaded research papers."

_SYSTEM_PROMPT = """You are a research-paper assistant. You must answer questions
using ONLY the context passages provided to you below. Each passage is labeled
with its source file and location (page/slide/section).

Strict rules:
1. Do not use any knowledge you have from outside the provided context.
2. Do not invent, guess, or infer citations, page numbers, authors, datasets,
   or results that are not explicitly present in the context.
3. If the context does not contain the answer, respond exactly with:
   "I could not find this information in the uploaded research papers."
4. Keep answers precise, factual, and directly tied to the given passages.
5. When comparing multiple papers, clearly attribute each fact to its source file.
6. Do not mistake authors cited within the paper's references for the actual
   authors of the paper itself unless the context clearly identifies them as such.
"""

_COMPARISON_SYSTEM_PROMPT = """You are a research-paper comparison assistant. You must
compare the uploaded documents using ONLY the context passages provided below.

Produce your answer as a Markdown table. Use the source filenames as column
headers (one column per document) and comparison aspects as rows (e.g.
Methodology, Dataset, Results, Contributions -- adjust the rows to what the
question actually asks about).

Strict rules:
1. Do not use any knowledge you have from outside the provided context.
2. If a piece of information for a given document/aspect is not present in
   the context, write "Not found in this paper" in that cell -- do not invent it.
3. Do not invent citations, page numbers, authors, datasets, or results.
4. After the table, you may add at most one short clarifying sentence if needed.
5. Do not mistake authors cited within a paper's references for that paper's
   actual authors.
"""

_COMPARISON_KEYWORDS = [
    "compare", "comparison", "versus", " vs ", " vs.", "difference between",
    "differences", "which paper", "which one performs", "better than",
    "compared to",
]


class RAGError(Exception):
    """Raised for configuration or LLM-call failures."""
    pass


def _get_groq_client() -> Groq:
    if not config.is_groq_configured():
        raise RAGError(
            "GROQ_API_KEY is not set. Please add it to your .env file "
            "(see .env.example) before asking questions."
        )
    return Groq(api_key=config.GROQ_API_KEY)


def _format_context(chunks: List[Dict]) -> str:
    blocks = []
    for chunk in chunks:
        location = chunk.get("location_label") or f"Page {chunk.get('page_number')}"
        blocks.append(
            f"[Source: {chunk['filename']} | {location}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def format_citations(chunks: List[Dict]) -> List[str]:
    """De-duplicated, human-readable citation lines from the chunks actually used."""
    seen = set()
    citations = []
    for chunk in chunks:
        location = chunk.get("location_label") or f"Page {chunk.get('page_number')}"
        key = (chunk["filename"], location)
        if key in seen:
            continue
        seen.add(key)
        citations.append(f"📄 {chunk['filename']} — {location}")
    return citations


def is_comparison_question(question: str) -> bool:
    """Detects comparison-style questions (compare/versus/difference/etc.)."""
    q = f" {question.lower()} "
    return any(keyword in q for keyword in _COMPARISON_KEYWORDS)


def compute_confidence(chunks: List[Dict]) -> str:
    """
    Estimate retrieval confidence from the semantic similarity of the top
    retrieved chunks. This reflects how strongly the evidence matches the
    question -- it does NOT verify that the generated answer is factually
    correct, only that relevant supporting text was found.
    """
    if not chunks:
        return "Low confidence — verify manually"

    top = chunks[:3]
    scores = [c.get("semantic_score") for c in top if c.get("semantic_score") is not None]
    if not scores:
        return "Medium confidence"

    avg_score = sum(scores) / len(scores)
    if avg_score >= config.CONFIDENCE_HIGH_THRESHOLD:
        return "High confidence"
    elif avg_score >= config.CONFIDENCE_MEDIUM_THRESHOLD:
        return "Medium confidence"
    return "Low confidence — verify manually"


def _is_transient_groq_error(exc: Exception) -> bool:
    """
    Heuristic classification of Groq errors: transient (rate limit, timeout,
    5xx/server-side) vs permanent (bad API key, invalid request). We only
    retry transient errors -- retrying an invalid API key repeatedly just
    wastes time and hides the real problem from the user.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code == 429 or status_code >= 500
    message = str(exc).lower()
    transient_signals = ["rate limit", "timeout", "timed out", "503", "502", "500", "temporarily"]
    return any(signal in message for signal in transient_signals)


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    client = _get_groq_client()
    last_exc = None

    for attempt in range(config.GROQ_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=config.GROQ_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_exc = exc
            is_last_attempt = attempt == config.GROQ_MAX_RETRIES - 1
            if is_last_attempt or not _is_transient_groq_error(exc):
                break
            time.sleep(config.GROQ_RETRY_BASE_DELAY_SECONDS * (2 ** attempt))

    if _is_transient_groq_error(last_exc):
        raise RAGError(
            "The language model service is temporarily unavailable (rate limit "
            "or transient error) even after retrying. Please try again shortly."
        )
    raise RAGError(f"The language model request failed: {last_exc}")


def answer_question(question: str, doc_ids: Optional[List[str]] = None,
                     top_k: Optional[int] = None) -> Dict:
    """
    Full RAG pipeline for a single question:
    retrieve -> build grounded prompt -> generate -> attach citations + confidence.

    `top_k` optionally overrides how many chunks are retrieved/passed to the
    LLM (exposed in the UI as an adjustable retrieval-depth control).

    Returns: {
        "answer": str, "citations": List[str], "chunks_used": List[Dict],
        "confidence": str, "is_comparison": bool
    }
    """
    if not question or not question.strip():
        raise RAGError("Please enter a question before asking.")

    if not doc_ids:
        raise RAGError("No papers are selected. Please upload and select at least one paper.")

    chunks = retriever.retrieve(question, doc_ids=doc_ids, top_k=top_k)

    if not chunks:
        return {
            "answer": NOT_FOUND_MESSAGE, "citations": [], "chunks_used": [],
            "confidence": "Low confidence — verify manually", "is_comparison": False,
        }

    comparison_mode = is_comparison_question(question) and len(doc_ids) > 1
    system_prompt = _COMPARISON_SYSTEM_PROMPT if comparison_mode else _SYSTEM_PROMPT

    context = _format_context(chunks)
    user_prompt = (
        f"Context passages from the uploaded document(s):\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. If multiple documents are involved, "
        "state clearly which document each piece of information comes from."
    )

    answer = _call_groq(system_prompt, user_prompt)
    confidence = compute_confidence(chunks)

    # If the model itself declares it can't find the answer, don't attach
    # citations that would misleadingly imply support.
    if NOT_FOUND_MESSAGE.lower() in answer.lower():
        return {
            "answer": NOT_FOUND_MESSAGE, "citations": [], "chunks_used": [],
            "confidence": "Low confidence — verify manually", "is_comparison": comparison_mode,
        }

    return {
        "answer": answer,
        "citations": format_citations(chunks),
        "chunks_used": chunks,
        "confidence": confidence,
        "is_comparison": comparison_mode,
    }


_SUMMARY_SYSTEM_PROMPT = """You are a research-paper summarization assistant. You must
summarize using ONLY the provided context from a single paper. Organize your summary
under these headings, in this order:

Title
Problem
Objective
Methodology
Dataset
Results
Contributions
Limitations
Conclusion

If the paper's provided context does not contain information for a heading,
write "Not explicitly stated in the provided content" under that heading instead
of inventing information. Do not use outside knowledge.
"""


def summarize_paper(doc_id: str, filename: str) -> Dict:
    """
    Structured, single-paper summary built only from that paper's indexed chunks.
    """
    chunks = vector_store.get_all_chunks_for_doc(doc_id)
    if not chunks:
        raise RAGError(f"No indexed content found for '{filename}'.")

    # Cap total context size sent to the LLM to keep the request reasonable
    # while still covering the whole paper (metadata chunk + representative body chunks).
    max_chunks = 25
    selected = chunks[:max_chunks]
    context = _format_context(selected)

    user_prompt = (
        f"Context passages from the document '{filename}':\n\n{context}\n\n"
        "Produce the structured summary described in your instructions."
    )

    summary = _call_groq(_SUMMARY_SYSTEM_PROMPT, user_prompt)
    return {
        "summary": summary,
        "citations": format_citations(selected),
    }
