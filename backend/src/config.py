"""
config.py

Central configuration for the Research Paper Intelligence system.
All tunable constants and environment-driven secrets live here so the
rest of the codebase never touches os.environ directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from a local .env file (if present) into the process environment.
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "papers"
CHROMA_DIR = BASE_DIR / "chroma_db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Groq LLM settings
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# A current, lightweight, high-quality Groq-hosted open-weight model.
# Kept as a single constant so the model can be swapped in one place.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Generation controls
LLM_TEMPERATURE = 0.1          # Low temperature -> grounded, non-creative answers
LLM_MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # Runs locally via sentence-transformers

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 900          # characters per chunk (approx.)
CHUNK_OVERLAP = 150       # character overlap between consecutive chunks

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
# If normal extraction yields fewer than this many usable characters,
# the page is treated as scanned/image-only and OCR is triggered.
MIN_TEXT_LENGTH_FOR_NO_OCR = 20
OCR_RENDER_ZOOM = 2.0     # Upscale factor when rasterizing a page for OCR

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_SEMANTIC = 8        # candidates pulled from the vector store per query
TOP_K_FINAL = 5           # final number of chunks passed to the LLM

# Hybrid ranking weights (must sum to ~1.0, not strictly enforced)
SEMANTIC_WEIGHT = 0.65
KEYWORD_WEIGHT = 0.35

# Chunk "types" used to distinguish first-page metadata from normal body text
CHUNK_TYPE_BODY = "body"
CHUNK_TYPE_METADATA = "first_page_metadata"

# ChromaDB collection name
CHROMA_COLLECTION_NAME = "research_papers"

# Keywords that indicate a metadata-oriented question (title/author/abstract/etc.)
METADATA_QUESTION_KEYWORDS = [
    "title", "paper name", "name of the paper", "author", "authors",
    "corresponding author", "abstract", "keyword", "keywords",
    "affiliation", "published", "journal", "conference",
]

# ---------------------------------------------------------------------------
# Hybrid retrieval: BM25 + cross-encoder re-ranking
# ---------------------------------------------------------------------------
# How many candidates to pull (from semantic search and from BM25) before
# merging and ranking. Kept larger than TOP_K_FINAL so the re-ranker has a
# meaningful pool to choose from without scoring the whole database.
CANDIDATE_POOL_SIZE = 20

# How many of the merged/hybrid-ranked candidates actually get passed to the
# (more expensive) cross-encoder re-ranker.
RERANK_CANDIDATE_COUNT = 10

# Cross-encoder model used to re-rank the candidate pool. Ships as part of
# the already-required `sentence-transformers` package (no new dependency).
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# If True, cross-encoder re-ranking is attempted; on any load/inference
# failure the system automatically falls back to plain hybrid ranking.
ENABLE_CROSS_ENCODER_RERANK = True

# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------
# Questions with a word count at or below this are considered "short/vague"
# and become eligible for deterministic expansion (see query_expansion.py).
SHORT_QUERY_WORD_THRESHOLD = 3

# ---------------------------------------------------------------------------
# Answer confidence indicator
# ---------------------------------------------------------------------------
# Thresholds are applied to the average semantic similarity (0-1) of the
# top retrieved chunks. These are estimated-confidence heuristics based on
# retrieval evidence, NOT a guarantee that the generated answer is correct.
CONFIDENCE_HIGH_THRESHOLD = 0.55
CONFIDENCE_MEDIUM_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Groq API reliability
# ---------------------------------------------------------------------------
GROQ_MAX_RETRIES = 3
GROQ_RETRY_BASE_DELAY_SECONDS = 1.5   # exponential backoff: 1.5s, 3s, 6s

# ---------------------------------------------------------------------------
# Multi-format document support
# ---------------------------------------------------------------------------
FILE_TYPE_PDF = "pdf"
FILE_TYPE_DOCX = "docx"
FILE_TYPE_PPTX = "pptx"
FILE_TYPE_TXT = "txt"
FILE_TYPE_MD = "md"
FILE_TYPE_CSV = "csv"
FILE_TYPE_XLSX = "xlsx"

SUPPORTED_EXTENSIONS = [
    FILE_TYPE_PDF, FILE_TYPE_DOCX, FILE_TYPE_PPTX,
    FILE_TYPE_TXT, FILE_TYPE_MD, FILE_TYPE_CSV, FILE_TYPE_XLSX,
]

# File types that are treated as "papers/documents" eligible for a
# first-segment metadata chunk (title/author-style fast path). Spreadsheet
# formats are data files, not papers, so they're excluded.
METADATA_ELIGIBLE_FILE_TYPES = [
    FILE_TYPE_PDF, FILE_TYPE_DOCX, FILE_TYPE_PPTX, FILE_TYPE_TXT, FILE_TYPE_MD,
]

# Approximate size (in paragraphs) of one DOCX "section" segment.
DOCX_PARAGRAPHS_PER_SEGMENT = 8

# Approximate size (in characters) of one TXT/MD segment before chunking.
PLAINTEXT_CHARS_PER_SEGMENT = 1200

# Approximate number of spreadsheet rows grouped into one segment.
SPREADSHEET_ROWS_PER_SEGMENT = 25


def is_groq_configured() -> bool:
    """Returns True if a Groq API key has been supplied via environment/.env."""
    return bool(GROQ_API_KEY and GROQ_API_KEY.strip())
