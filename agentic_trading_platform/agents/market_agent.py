"""Market Analysis Agent - provides technical analysis and market context."""

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

from .tools.market_tools import (
    get_stock_price,
    get_technical_indicators,
    get_market_overview,
    get_market_tools,
    HAS_YFINANCE,
)


class MarketAnalysisAgent:
    """Provides market data, technical indicators, and market condition explanations."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or ""
        self.llm = None
        self._setup()

    def _setup(self) -> None:
        """Initialize LLM."""
        if HAS_LANGCHAIN and self.api_key:
            try:
                self.llm = ChatGroq(
                    model="llama-3.1-70b-versatile",
                    api_key=self.api_key,
                    temperature=0.2,
                )
            except Exception as e:
                logger.warning("Could not initialize Market Agent Groq LLM: %s", e)
                self.llm = None

    def _format_price(self, data: dict) -> str:
        """Format get_stock_price result as readable text."""
        if data.get("error"):
            return f"Error for {data.get('symbol', '?')}: {data['error']}"
        lines = [
            f"Symbol: {data['symbol']}",
            f"Current price: {data.get('current_price')} {data.get('currency', 'USD')}",
            f"Day range: {data.get('day_low')} - {data.get('day_high')}",
            f"Volume: {data.get('volume')}",
            f"Previous close: {data.get('previous_close')}",
        ]
        return "\n".join(str(x) for x in lines)

    def _format_indicators(self, data: dict) -> str:
        """Format get_technical_indicators result."""
        if data.get("error"):
            return f"Error for {data.get('symbol', '?')}: {data['error']}"
        lines = [
            f"Symbol: {data['symbol']} (period: {data.get('period', '3mo')})",
            f"RSI(14): {data.get('last_rsi')}",
            f"MACD: {data.get('last_macd')} | Signal: {data.get('last_macd_signal')} | Hist: {data.get('last_macd_hist')}",
            f"EMA 20: {data.get('ema_20')} | EMA 50: {data.get('ema_50')}",
        ]
        return "\n".join(str(x) for x in lines)

    def _format_overview(self, data: dict) -> str:
        """Format get_market_overview result."""
        if data.get("error"):
            return f"Market overview error: {data['error']}"
        lines = []
        if data.get("sp500"):
            s = data["sp500"]
            lines.append(f"S&P 500: {s.get('price')} ({s.get('change_pct')}% change)")
        if data.get("vix") is not None:
            lines.append(f"VIX: {data['vix']}")
        if data.get("sectors"):
            lines.append("Sector performance:")
            for sec in data["sectors"][:8]:
                lines.append(f"  {sec.get('name')} ({sec.get('symbol')}): {sec.get('price')} ({sec.get('change_pct')}%)")
        return "\n".join(lines) if lines else "No market overview data."

    def get_price(self, symbol: str) -> str:
        """Get current price and basic info for a symbol."""
        if not HAS_YFINANCE:
            return "Market data requires yfinance. Install with: pip install yfinance"
        return self._format_price(get_stock_price(symbol))

    def get_indicators(self, symbol: str, period: str = "3mo") -> str:
        """Get technical indicators for a symbol."""
        if not HAS_YFINANCE:
            return "Market data requires yfinance. Install with: pip install yfinance"
        return self._format_indicators(get_technical_indicators(symbol, period))

    def get_overview(self) -> str:
        """Get broad market overview."""
        if not HAS_YFINANCE:
            return "Market data requires yfinance. Install with: pip install yfinance"
        return self._format_overview(get_market_overview())

    def _extract_symbols(self, query: str) -> List[str]:
        """Extract potential ticker symbols from query (simple heuristic)."""
        import re
        # Common tickers are 1-5 uppercase letters
        candidates = re.findall(r"\b([A-Z]{1,5})\b", query.upper())
        # Filter known indices/symbols
        known = {"RSI", "MACD", "EMA", "VIX", "SPY", "ETF", "USD", "API"}
        return [c for c in candidates if c not in known][:3]

    def answer(self, question: str) -> str:
        """
        Answer market-related questions using tools and optional LLM.
        Uses get_stock_price, get_technical_indicators, get_market_overview.
        """
        if not question or not question.strip():
            return "Ask a question about a stock symbol, technical indicators (RSI, MACD, EMA), or market overview."

        q = question.strip().lower()
        tool_results: List[str] = []

        # Market overview
        if any(k in q for k in ["market", "overview", "s&p", "vix", "sector", "broad"]):
            tool_results.append("Market overview:\n" + self.get_overview())

        # Extract symbol and run price/indicators if mentioned
        symbols = self._extract_symbols(question)
        if not symbols and any(k in q for k in ["price", "rsi", "macd", "ema", "technical", "indicator"]):
            symbols = ["SPY"]  # default to SPY for "market" technicals
        for sym in symbols[:2]:  # at most 2 symbols
            if any(k in q for k in ["price", "current", "quote", "trading"]):
                tool_results.append(f"{sym}:\n" + self.get_price(sym))
            if any(k in q for k in ["rsi", "macd", "ema", "technical", "indicator"]):
                tool_results.append(f"{sym} indicators:\n" + self.get_indicators(sym))

        context = "\n\n".join(tool_results) if tool_results else "No market data retrieved. Try asking for a symbol (e.g. AAPL) or 'market overview'."

        if self.llm and context:
            try:
                sys_msg = (
                    "You are a market analyst. Use the following data to answer the user. "
                    "Be concise. If data is missing or there was an error, say so."
                )
                messages = [
                    SystemMessage(content=sys_msg),
                    HumanMessage(content=f"Data:\n{context}\n\nQuestion: {question}"),
                ]
                resp = self.llm.invoke(messages)
                if hasattr(resp, "content") and resp.content:
                    return resp.content.strip()
            except Exception as e:
                logger.warning("Market Agent LLM failed: %s", e)

        return context
