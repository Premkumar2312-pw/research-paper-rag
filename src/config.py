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
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

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


def is_groq_configured() -> bool:
    """Returns True if a Groq API key has been supplied via environment/.env."""
    return bool(GROQ_API_KEY and GROQ_API_KEY.strip())
