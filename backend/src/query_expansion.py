"""
query_expansion.py

Lightweight, deterministic expansion of short/vague questions into fuller
retrieval queries. This intentionally does NOT call the LLM -- it's a
simple keyword-to-phrase lookup table, which keeps it fast, free, and
easy to explain in a viva ("if the question is very short and matches a
known pattern, we substitute a fuller phrasing before embedding it").

The expanded query is used ONLY for retrieval (to improve which chunks
get pulled from the vector store). The original question the user typed
is still what gets sent to the LLM for answer generation, so the answer's
wording always reflects what was actually asked.
"""

import re
from typing import Optional

from src import config

# Deterministic mapping: trigger keyword -> fuller retrieval phrasing.
# Matched against the question with punctuation stripped and lowercased.
_EXPANSION_MAP = {
    "results": "What were the experimental results and evaluation metrics reported in the research paper?",
    "result": "What were the experimental results and evaluation metrics reported in the research paper?",
    "method": "What methodology, model, or approach is proposed in the research paper?",
    "methods": "What methodology, model, or approach is proposed in the research paper?",
    "methodology": "What methodology, model, or approach is proposed in the research paper?",
    "dataset": "What dataset was used in the research paper?",
    "datasets": "What datasets were used in the research paper?",
    "conclusion": "What is the conclusion of the research paper?",
    "limitations": "What are the limitations of the research paper?",
    "limitation": "What are the limitations of the research paper?",
    "contribution": "What is the main contribution of the research paper?",
    "contributions": "What are the main contributions of the research paper?",
    "abstract": "What is the abstract of the research paper?",
    "title": "What is the title of the research paper?",
    "authors": "Who are the authors of the research paper?",
    "author": "Who are the authors of the research paper?",
    "objective": "What is the objective or goal of the research paper?",
    "future work": "What future work is suggested in the research paper?",
    "architecture": "What model or system architecture is described in the research paper?",
    "accuracy": "What accuracy or performance was achieved in the research paper?",
}


def _normalize(question: str) -> str:
    return re.sub(r"[^\w\s]", "", question.strip().lower())


def expand_query(question: str) -> str:
    """
    Return an expanded retrieval query for short/vague questions that match
    a known trigger phrase; otherwise return the original question unchanged.
    """
    if not question or not question.strip():
        return question

    normalized = _normalize(question)
    word_count = len(normalized.split())

    if word_count > config.SHORT_QUERY_WORD_THRESHOLD:
        return question  # Question is already specific enough; don't touch it.

    # Exact match first (e.g. "results", "dataset?", "method")
    if normalized in _EXPANSION_MAP:
        return _EXPANSION_MAP[normalized]

    # Fallback: check if the short question CONTAINS one of the trigger words
    # (e.g. "the results?" or "results pls").
    for trigger, expanded in _EXPANSION_MAP.items():
        if trigger in normalized.split():
            return expanded

    return question


def was_expanded(question: str) -> Optional[str]:
    """Returns the expanded form if expansion would apply, else None. Used for UI/debug display."""
    expanded = expand_query(question)
    return expanded if expanded != question else None
