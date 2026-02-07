"""Stock Screener - screens and recommends stocks based on constraints."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:
    yf = None


class StockScreener:
    """Screens stocks based on user constraints and ML rankings."""

    # Default stock universes
    US_LARGE_CAPS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "JNJ",
        "V", "PG", "MA", "UNH", "HD", "DIS", "BAC", "XOM", "PFE", "CSCO",
        "NFLX", "ADBE", "CRM", "AMD", "INTC", "T", "VZ", "KO", "PEP", "MRK",
    ]

    INDIA_STOCKS = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "BAJFINANCE.NS",
        "TITAN.NS", "WIPRO.NS", "HCLTECH.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS",
    ]

    SECTOR_MAP = {
        "technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "ADBE", "CRM", "AMD", "INTC", "CSCO"],
        "healthcare": ["JNJ", "UNH", "PFE", "MRK"],
        "finance": ["JPM", "BAC", "V", "MA", "BRK-B"],
        "consumer": ["AMZN", "TSLA", "HD", "DIS", "KO", "PEP", "PG", "NFLX"],
        "energy": ["XOM"],
        "telecom": ["T", "VZ"],
    }

    def __init__(self, momentum_scorer=None, risk_classifier=None):
        self.momentum_scorer = momentum_scorer
        self.risk_classifier = risk_classifier

    def _filter_by_sector(self, symbols: List[str], sector: str) -> List[str]:
        """Return symbols that belong to the given sector."""
        if not sector:
            return list(symbols)
        sector = sector.lower().strip()
        if sector in self.SECTOR_MAP:
            sector_set = set(self.SECTOR_MAP[sector])
            return [s for s in symbols if s in sector_set]
        return list(symbols)

    def _filter_by_risk(self, symbols: List[str], risk_tolerance: str) -> List[str]:
        """Filter symbols by risk tolerance using risk_classifier if available."""
        if not risk_tolerance or not self.risk_classifier:
            return list(symbols)
        risk_tolerance = risk_tolerance.lower().strip()
        df = self.risk_classifier.classify_risk(symbols)
        if df.empty:
            return list(symbols)
        allowed = df[df["risk_level"] == risk_tolerance]["symbol"].tolist()
        return [s for s in symbols if s in allowed] if allowed else list(symbols)

    def _rank_stocks(self, symbols: List[str], horizon: str) -> pd.DataFrame:
        """Rank stocks by momentum (and optionally horizon). Returns DataFrame with score and rank."""
        if not symbols:
            return pd.DataFrame(columns=["symbol", "score", "rank"])
        if self.momentum_scorer:
            ranked = self.momentum_scorer.score_stocks(symbols)
            if not ranked.empty:
                return ranked
        # Fallback: assign equal score and rank by symbol order
        return pd.DataFrame({
            "symbol": symbols,
            "score": 0.5,
            "rank": range(1, len(symbols) + 1),
        })

    def get_stock_info(self, symbol: str) -> Dict:
        """Get basic info for a stock using yfinance."""
        if not yf:
            return {"symbol": symbol, "error": "yfinance not installed"}
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                "symbol": symbol,
                "shortName": info.get("shortName"),
                "longName": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "marketCap": info.get("marketCap"),
                "currentPrice": info.get("currentPrice"),
                "currency": info.get("currency"),
            }
        except Exception as e:
            logger.exception("Error getting info for %s: %s", symbol, e)
            return {"symbol": symbol, "error": str(e)}

    def screen(self, constraints: Dict) -> pd.DataFrame:
        """
        Screen stocks based on constraints:
        - sector: str or list
        - risk_tolerance: 'low', 'medium', 'high'
        - horizon: 'short' (<1mo), 'medium' (1-6mo), 'long' (>6mo)
        - budget: float (max investment)
        - market: 'us' or 'india'
        Returns: DataFrame with recommended stocks sorted by score
        """
        market = (constraints.get("market") or "us").lower()
        symbols = self.INDIA_STOCKS if market == "india" else self.US_LARGE_CAPS

        sector = constraints.get("sector")
        if sector is not None:
            if isinstance(sector, list):
                sector_set = set()
                for s in sector:
                    s = (s or "").lower().strip()
                    if s in self.SECTOR_MAP:
                        sector_set.update(self.SECTOR_MAP[s])
                symbols = [sym for sym in symbols if sym in sector_set] if sector_set else symbols
            else:
                symbols = self._filter_by_sector(symbols, str(sector))

        risk_tolerance = constraints.get("risk_tolerance")
        if risk_tolerance:
            symbols = self._filter_by_risk(symbols, risk_tolerance)

        if not symbols:
            return pd.DataFrame(columns=["symbol", "score", "rank"])

        horizon = constraints.get("horizon") or "medium"
        ranked = self._rank_stocks(symbols, horizon)
        if ranked.empty:
            return ranked

        budget = constraints.get("budget")
        if budget is not None and budget > 0 and yf:
            # Optionally limit number of recommendations by budget (simplified: top N by score)
            ranked = ranked.head(20)
        else:
            ranked = ranked.head(20)

        return ranked.sort_values("score", ascending=False).reset_index(drop=True)
