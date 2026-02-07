"""Application configuration."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
OUTPUT_DIR = DATA_DIR / "output"

# API Keys (from environment variables)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Model settings
DEFAULT_LLM_MODEL = "llama3-70b-8192"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Analysis settings
DEFAULT_N_CLUSTERS = 3
DEFAULT_BASELINE_WINDOW = 30

# Upload settings
MAX_UPLOAD_SIZE_MB = 50
ALLOWED_EXTENSIONS = [".csv", ".pdf"]
