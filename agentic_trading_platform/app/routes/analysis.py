"""
Analysis API endpoints.

POST /analyze - run BehavioralAnalyzer pipeline on uploaded file.
GET /results - return analysis results (summary, report path, plot paths).
GET /report - return generated report text.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from ..config import (
    ALLOWED_EXTENSIONS,
    DATA_DIR,
    DEFAULT_BASELINE_WINDOW,
    DEFAULT_N_CLUSTERS,
    MAX_UPLOAD_SIZE_MB,
    OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])

# In-memory store for latest analysis (single-session; replace with DB/redis in production)
_analysis_store: Dict[str, Any] = {}


def _allowed_file(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in ALLOWED_EXTENSIONS


class AnalyzeResponse(BaseModel):
    """Response after running analysis."""

    success: bool
    message: str
    summary: Optional[Dict[str, Any]] = None
    report_path: Optional[str] = None
    plot_paths: Optional[Dict[str, str]] = None
    error: Optional[str] = None


class ResultsResponse(BaseModel):
    """Analysis results for GET /results."""

    has_results: bool
    summary: Optional[Dict[str, Any]] = None
    report_path: Optional[str] = None
    report_text: Optional[str] = None
    plot_paths: Optional[Dict[str, str]] = None
    xai_path: Optional[str] = None


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    n_clusters: int = DEFAULT_N_CLUSTERS,
    baseline_window: int = DEFAULT_BASELINE_WINDOW,
) -> AnalyzeResponse:
    """
    Run BehavioralAnalyzer pipeline on uploaded PDF or CSV.

    Saves the file temporarily, runs load -> enrich -> analyze -> visualize -> report,
    then returns summary, report path, and plot paths.
    """
    if not file.filename or not _allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or missing file. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    try:
        contents = await file.read()
        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max: {MAX_UPLOAD_SIZE_MB} MB",
            )
        safe_name = Path(file.filename).name
        input_path = cache_dir / safe_name
        input_path.write_bytes(contents)

        from behavioral_trading import BehavioralAnalyzer

        analyzer = BehavioralAnalyzer(
            n_clusters=n_clusters,
            baseline_window=baseline_window,
        )
        file_type = input_path.suffix.lower().lstrip(".")
        analyzer.load_tradebook(str(input_path), file_type=file_type)
        analyzer.enrich_with_market_data()
        results = analyzer.analyze()
        plot_paths = analyzer.visualize(results=results, output_dir=str(OUTPUT_DIR))
        report_path = os.path.join(str(OUTPUT_DIR), "behavioral_report.txt")
        analyzer.generate_report(results=results, output_file=report_path)
        summary = analyzer.get_summary()

        # Normalize plot_paths to names -> paths
        plot_paths_dict = dict(plot_paths) if plot_paths else {}

        # Store for GET /results and GET /report
        xai_path = os.path.join(str(OUTPUT_DIR), "xai_explanation.txt")
        _analysis_store["summary"] = summary
        _analysis_store["report_path"] = report_path
        _analysis_store["report_text"] = Path(report_path).read_text(encoding="utf-8")
        _analysis_store["plot_paths"] = plot_paths_dict
        _analysis_store["xai_path"] = xai_path
        _analysis_store["xai_text"] = (
            Path(xai_path).read_text(encoding="utf-8") if os.path.isfile(xai_path) else ""
        )
        _analysis_store["results"] = results

        # Update the chat agent orchestrator with new analysis results
        try:
            from .chat import update_chat_analysis_results
            update_chat_analysis_results(results)
        except Exception as e:
            logger.debug("Could not update chat agent with analysis: %s", e)

        return AnalyzeResponse(
            success=True,
            message="Analysis completed successfully.",
            summary=summary,
            report_path=report_path,
            plot_paths=plot_paths_dict,
        )
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Analysis validation error: %s", e)
        return AnalyzeResponse(
            success=False,
            message="Analysis failed.",
            error=str(e),
        )
    except Exception as e:
        logger.exception("Analysis failed: %s", e)
        return AnalyzeResponse(
            success=False,
            message="Analysis failed.",
            error=str(e),
        )


@router.get("/results", response_model=ResultsResponse)
def get_results() -> ResultsResponse:
    """Return the latest analysis results (summary, report path, plot paths, XAI path)."""
    if not _analysis_store:
        return ResultsResponse(has_results=False)

    report_text = _analysis_store.get("report_text")
    if _analysis_store.get("report_path") and report_text is None:
        try:
            report_text = Path(_analysis_store["report_path"]).read_text(encoding="utf-8")
        except Exception:
            report_text = None

    return ResultsResponse(
        has_results=True,
        summary=_analysis_store.get("summary"),
        report_path=_analysis_store.get("report_path"),
        report_text=report_text,
        plot_paths=_analysis_store.get("plot_paths"),
        xai_path=_analysis_store.get("xai_path"),
    )


@router.get("/report")
def get_report() -> dict:
    """Return the generated behavioral report text and XAI explanation."""
    if not _analysis_store:
        raise HTTPException(status_code=404, detail="No analysis results available.")
    report_text = _analysis_store.get("report_text")
    if report_text is None and _analysis_store.get("report_path"):
        try:
            report_text = Path(_analysis_store["report_path"]).read_text(encoding="utf-8")
        except Exception:
            report_text = ""
    return {
        "report": report_text or "",
        "xai_summary": _analysis_store.get("xai_text", ""),
    }
