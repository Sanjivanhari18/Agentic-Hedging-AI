"""
Tools for accessing behavioral analysis results.

Tools: get_trade_summary, get_behavioral_patterns, get_anomalies.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Module-level store for analysis results (set by orchestrator/behavioral agent)
_analysis_results: Optional[Dict[str, Any]] = None


def set_analysis_results(results: Optional[Dict[str, Any]]) -> None:
    """Set the global analysis results used by tools."""
    global _analysis_results
    _analysis_results = results


def get_analysis_results() -> Optional[Dict[str, Any]]:
    """Get current analysis results."""
    return _analysis_results


def get_trade_summary() -> dict:
    """
    Get summary statistics from the latest behavioral analysis.

    Returns:
        Dict with total_trades, date_range, pnl (if available), clusters (if available),
        behavioral_probabilities (if available), error.
    """
    out: dict = {
        "total_trades": None,
        "date_range": None,
        "pnl": None,
        "clusters": None,
        "behavioral_probabilities": None,
        "error": None,
    }
    res = get_analysis_results()
    if not res:
        out["error"] = "No analysis results available. Run analysis first."
        return out
    try:
        features = res.get("features")
        if features is None:
            out["error"] = "No features in analysis results"
            return out
        import pandas as pd
        if not isinstance(features, pd.DataFrame):
            out["error"] = "features is not a DataFrame"
            return out
        out["total_trades"] = len(features)
        if "date" in features.columns:
            out["date_range"] = {
                "start": str(features["date"].min()),
                "end": str(features["date"].max()),
            }
        if "realized_pnl" in features.columns:
            pnl = features["realized_pnl"]
            out["pnl"] = {
                "total": float(pnl.sum()),
                "average": float(pnl.mean()),
                "win_rate": float((pnl > 0).mean()),
            }
        if "behavioral_cluster" in features.columns:
            out["clusters"] = {
                int(c): int((features["behavioral_cluster"] == c).sum())
                for c in features["behavioral_cluster"].dropna().unique()
            }
        prob = res.get("probabilistic") or {}
        if prob.get("behavioral_probabilities"):
            out["behavioral_probabilities"] = prob["behavioral_probabilities"]
    except Exception as e:
        logger.exception("get_trade_summary failed")
        out["error"] = str(e)
    return out


def get_behavioral_patterns() -> dict:
    """
    Get cluster descriptions and pattern metadata from analysis.

    Returns:
        Dict with cluster_descriptions, change_points, n_clusters, error.
    """
    out: dict = {
        "cluster_descriptions": None,
        "change_points": None,
        "n_clusters": None,
        "error": None,
    }
    res = get_analysis_results()
    if not res:
        out["error"] = "No analysis results available."
        return out
    try:
        patterns = res.get("patterns") or {}
        clusters = patterns.get("clusters") or {}
        analysis = clusters.get("analysis") or {}
        out["cluster_descriptions"] = analysis
        out["n_clusters"] = len(analysis) if analysis else None
        cp = patterns.get("change_points") or {}
        out["change_points"] = {
            "indices": cp.get("indices", []),
            "count": len(cp.get("indices", [])),
        }
    except Exception as e:
        logger.exception("get_behavioral_patterns failed")
        out["error"] = str(e)
    return out


def get_anomalies() -> dict:
    """
    Get anomalous trade information from the analysis.

    Returns:
        Dict with anomaly_indices, count, details (if available), error.
    """
    out: dict = {
        "anomaly_indices": [],
        "count": 0,
        "details": None,
        "error": None,
    }
    res = get_analysis_results()
    if not res:
        out["error"] = "No analysis results available."
        return out
    try:
        patterns = res.get("patterns") or {}
        anomalies = patterns.get("anomalies") or {}
        indices = anomalies.get("indices") or anomalies.get("anomaly_indices") or []
        if isinstance(indices, list):
            out["anomaly_indices"] = [int(i) for i in indices]
        else:
            out["anomaly_indices"] = list(indices) if hasattr(indices, "__iter__") else []
        out["count"] = len(out["anomaly_indices"])
        if "scores" in anomalies:
            out["details"] = {"scores": anomalies["scores"]}
    except Exception as e:
        logger.exception("get_anomalies failed")
        out["error"] = str(e)
    return out


try:
    from langchain_core.tools import tool
    HAS_LANGCHAIN_TOOLS = True
except ImportError:
    HAS_LANGCHAIN_TOOLS = False

if HAS_LANGCHAIN_TOOLS:

    @tool
    def get_trade_summary_tool() -> dict:
        """Get summary statistics from the latest behavioral trading analysis (trades, PnL, clusters)."""
        return get_trade_summary()

    @tool
    def get_behavioral_patterns_tool() -> dict:
        """Get behavioral cluster descriptions and pattern change points from the analysis."""
        return get_behavioral_patterns()

    @tool
    def get_anomalies_tool() -> dict:
        """Get anomalous trade indices and details from the behavioral analysis."""
        return get_anomalies()

    def get_analysis_tools() -> list:
        """Return list of LangChain tools for analysis results."""
        return [get_trade_summary_tool, get_behavioral_patterns_tool, get_anomalies_tool]
else:

    def get_analysis_tools() -> list:
        return []
