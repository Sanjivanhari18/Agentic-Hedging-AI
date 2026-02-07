"""Stock Recommender Agent - suggests stocks based on constraints."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, SystemMessage
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    ChatGroq = None
    HumanMessage = None
    SystemMessage = None

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    yf = None

# Placeholder for ML model (screening/ranking) - can be replaced with real model
_ML_MODEL_AVAILABLE = False


def _screen_stocks(
    sector: Optional[str] = None,
    risk_tolerance: Optional[str] = None,
    horizon: Optional[str] = None,
    budget: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Placeholder stock screening. Returns a short list of candidate symbols
    based on simple criteria. Replace with real screening/ML when available.
    """
    # Curated list for demo; in production use a screener API or ML model
    candidates = [
        {"symbol": "AAPL", "sector": "Technology", "risk": "low", "name": "Apple Inc."},
        {"symbol": "MSFT", "sector": "Technology", "risk": "low", "name": "Microsoft"},
        {"symbol": "GOOGL", "sector": "Technology", "risk": "medium", "name": "Alphabet"},
        {"symbol": "JPM", "sector": "Financials", "risk": "medium", "name": "JPMorgan Chase"},
        {"symbol": "V", "sector": "Financials", "risk": "low", "name": "Visa"},
        {"symbol": "JNJ", "sector": "Healthcare", "risk": "low", "name": "Johnson & Johnson"},
        {"symbol": "UNH", "sector": "Healthcare", "risk": "medium", "name": "UnitedHealth"},
        {"symbol": "XOM", "sector": "Energy", "risk": "medium", "name": "Exxon Mobil"},
        {"symbol": "PG", "sector": "Consumer Staples", "risk": "low", "name": "Procter & Gamble"},
        {"symbol": "SPY", "sector": "ETF", "risk": "low", "name": "S&P 500 ETF"},
    ]
    out = []
    sector_lower = (sector or "").lower()
    risk_lower = (risk_tolerance or "").lower()
    for c in candidates:
        if sector_lower and sector_lower not in c.get("sector", "").lower():
            continue
        if risk_lower and risk_lower not in c.get("risk", "").lower():
            continue
        out.append(c)
    return out[:10] if out else candidates[:5]


class StockRecommenderAgent:
    """Suggests stocks based on sector, risk tolerance, horizon, and budget."""

    def __init__(self, analysis_results: Optional[Dict] = None, api_key: str = ""):
        self.api_key = api_key or ""
        self.analysis_results = analysis_results or {}
        self.llm = None
        self._ml_model = None  # Placeholder for ML model
        self._setup()

    def _setup(self) -> None:
        """Initialize LLM and optional ML model."""
        if HAS_LANGCHAIN and self.api_key:
            try:
                self.llm = ChatGroq(
                    model="llama-3.1-70b-versatile",
                    api_key=self.api_key,
                    temperature=0.3,
                )
            except Exception as e:
                logger.warning("Could not initialize Recommender Groq LLM: %s", e)
                self.llm = None
        if _ML_MODEL_AVAILABLE:
            try:
                # self._ml_model = load_model(...)
                pass
            except Exception as e:
                logger.warning("ML model not loaded: %s", e)

    def recommend(
        self,
        sector: Optional[str] = None,
        risk_tolerance: Optional[str] = None,
        horizon: Optional[str] = None,
        budget: Optional[float] = None,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return personalized stock recommendations.
        Uses screening criteria; when ML model is available, can rank by model score.
        """
        screened = _screen_stocks(
            sector=sector,
            risk_tolerance=risk_tolerance,
            horizon=horizon,
            budget=budget,
        )
        # If ML model available, rank screened list here
        return screened[:top_n]

    def _parse_constraints(self, question: str) -> Dict[str, Any]:
        """Extract sector, risk, horizon, budget from natural language."""
        q = question.lower()
        constraints = {}
        sectors = [
            "technology", "financials", "healthcare", "energy",
            "consumer staples", "consumer discretionary", "industrials",
            "materials", "utilities", "real estate", "communications", "etf",
        ]
        for s in sectors:
            if s in q:
                constraints["sector"] = s
                break
        if "low risk" in q or "conservative" in q:
            constraints["risk_tolerance"] = "low"
        elif "high risk" in q or "aggressive" in q:
            constraints["risk_tolerance"] = "high"
        elif "medium risk" in q or "moderate" in q:
            constraints["risk_tolerance"] = "medium"
        if "short term" in q or "short horizon" in q:
            constraints["horizon"] = "short"
        elif "long term" in q or "long horizon" in q:
            constraints["horizon"] = "long"
        # Simple budget extraction (e.g. "5000" or "$5000")
        import re
        m = re.search(r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:k|K)?", q)
        if m:
            val = float(m.group(1).replace(",", ""))
            if "k" in q[m.start():m.end()].lower():
                val *= 1000
            constraints["budget"] = val
        return constraints

    def answer(self, question: str) -> str:
        """
        Answer recommendation requests using constraints and optional LLM.
        """
        if not question or not question.strip():
            return "Describe what you're looking for: e.g. sector, risk tolerance, time horizon, or budget."

        constraints = self._parse_constraints(question)
        recs = self.recommend(
            sector=constraints.get("sector"),
            risk_tolerance=constraints.get("risk_tolerance"),
            horizon=constraints.get("horizon"),
            budget=constraints.get("budget"),
            top_n=5,
        )

        context_lines = [
            "Constraints used: " + (str(constraints) if constraints else "none (default screening)."),
            "Recommendations:",
        ]
        for r in recs:
            context_lines.append(f"  - {r.get('symbol')} ({r.get('name')}): sector={r.get('sector')}, risk={r.get('risk')}")

        context = "\n".join(context_lines)

        if self.llm:
            try:
                sys_msg = (
                    "You are a stock recommendation assistant. Present the recommendations "
                    "in a friendly, concise way. Add a brief disclaimer that this is not financial advice."
                )
                messages = [
                    SystemMessage(content=sys_msg),
                    HumanMessage(content=f"Data:\n{context}\n\nUser asked: {question}"),
                ]
                resp = self.llm.invoke(messages)
                if hasattr(resp, "content") and resp.content:
                    return resp.content.strip()
            except Exception as e:
                logger.warning("Recommender LLM failed: %s", e)

        return (
            context + "\n\nThis is not financial advice. Consider doing your own research or "
            "consulting a financial advisor."
        )
