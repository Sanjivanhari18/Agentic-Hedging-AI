"""
FastAPI backend entry point for the Agentic Trading Platform.

Provides REST API for upload, analysis, chat, and recommendations,
with CORS, static file serving for output plots, and health check.
"""

import logging
from pathlib import Path
from typing import List

from fastapi import File, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import (
    ALLOWED_EXTENSIONS,
    DATA_DIR,
    MAX_UPLOAD_SIZE_MB,
    OUTPUT_DIR,
)
from .routes import analysis, chat, recommend

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic Trading Platform API",
    description="REST API for behavioral analysis, chat, and recommendations.",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data and output dirs exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files for output plots (HTML and assets)
if OUTPUT_DIR.exists():
    app.mount("/static/plots", StaticFiles(directory=str(OUTPUT_DIR)), name="plots")

# Include routers
app.include_router(analysis.router, prefix="/api")
app.include_router(chat.router, prefix="/api/chat")
app.include_router(recommend.router, prefix="/api/recommend")


def _allowed_file(filename: str) -> bool:
    """Return True if filename has an allowed extension."""
    suffix = Path(filename).suffix.lower()
    return suffix in ALLOWED_EXTENSIONS


@app.get("/health")
def health_check() -> dict:
    """Health check endpoint for load balancers and monitoring."""
    return {"status": "ok", "service": "agentic-trading-platform"}


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
) -> dict:
    """
    Accept PDF or CSV file upload.

    Saves the file under data/cache and returns the stored path and metadata.
    Does not run analysis; use POST /api/analyze for that.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    if not _allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    try:
        contents = await file.read()
        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {MAX_UPLOAD_SIZE_MB} MB",
            )
        safe_name = Path(file.filename).name
        out_path = cache_dir / safe_name
        out_path.write_bytes(contents)
        return {
            "filename": file.filename,
            "path": str(out_path),
            "size_bytes": len(contents),
            "content_type": file.content_type or "application/octet-stream",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed: %s", e)
        raise HTTPException(status_code=500, detail="Upload failed") from e
