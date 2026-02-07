"""
LangChain-compatible tools wrapping yfinance for market data.

Tools: get_stock_price, get_technical_indicators, get_market_overview.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    yf = None

# LangChain tool decorator / BaseTool - optional
try:
    from langchain_core.tools import tool
    HAS_LANGCHAIN_TOOLS = True
except ImportError:
    HAS_LANGCHAIN_TOOLS = False


def _get_ticker(symbol: str):
    """Return yfinance Ticker; raises if yfinance not available."""
    if not HAS_YFINANCE:
        raise RuntimeError("yfinance is not installed. Install with: pip install yfinance")
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("symbol is required")
    return yf.Ticker(symbol)


def get_stock_price(symbol: str) -> dict:
    """
    Get current price and basic info for a stock symbol.

    Args:
        symbol: Stock ticker (e.g. AAPL, MSFT).

    Returns:
        Dict with keys: symbol, current_price, currency, day_high, day_low,
        volume, previous_close, open, error (if any).
    """
    out: dict = {
        "symbol": symbol,
        "current_price": None,
        "currency": None,
        "day_high": None,
        "day_low": None,
        "volume": None,
        "previous_close": None,
        "open": None,
        "error": None,
    }
    try:
        t = _get_ticker(symbol)
        info = t.info
        if not info:
            out["error"] = "No info returned for symbol"
            return out
        out["current_price"] = info.get("currentPrice") or info.get("regularMarketPrice")
        out["currency"] = info.get("currency", "USD")
        out["day_high"] = info.get("dayHigh") or info.get("regularMarketDayHigh")
        out["day_low"] = info.get("dayLow") or info.get("regularMarketDayLow")
        out["volume"] = info.get("volume") or info.get("regularMarketVolume")
        out["previous_close"] = info.get("previousClose") or info.get("regularMarketPreviousClose")
        out["open"] = info.get("open") or info.get("regularMarketOpen")
        if out["current_price"] is None and out["previous_close"] is not None:
            out["current_price"] = out["previous_close"]
    except Exception as e:
        logger.exception("get_stock_price failed for %s", symbol)
        out["error"] = str(e)
    return out


def get_technical_indicators(symbol: str, period: str = "3mo") -> dict:
    """
    Get technical indicators (RSI, MACD, EMA) for a symbol.

    Args:
        symbol: Stock ticker.
        period: yfinance period (e.g. 1mo, 3mo, 6mo, 1y).

    Returns:
        Dict with symbol, period, last_rsi, last_macd, last_macd_signal,
        last_macd_hist, ema_20, ema_50, error.
    """
    out: dict = {
        "symbol": symbol,
        "period": period,
        "last_rsi": None,
        "last_macd": None,
        "last_macd_signal": None,
        "last_macd_hist": None,
        "ema_20": None,
        "ema_50": None,
        "error": None,
    }
    try:
        t = _get_ticker(symbol)
        hist = t.history(period=period)
        if hist is None or hist.empty or len(hist) < 50:
            out["error"] = "Insufficient history for indicators"
            return out

        close = hist["Close"]

        # RSI(14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        out["last_rsi"] = float(rsi.iloc[-1]) if not rsi.empty else None

        # EMA 20, 50
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        out["ema_20"] = float(ema20.iloc[-1]) if not ema20.empty else None
        out["ema_50"] = float(ema50.iloc[-1]) if not ema50.empty else None

        # MACD(12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist_line = macd_line - signal_line
        out["last_macd"] = float(macd_line.iloc[-1]) if not macd_line.empty else None
        out["last_macd_signal"] = float(signal_line.iloc[-1]) if not signal_line.empty else None
        out["last_macd_hist"] = float(hist_line.iloc[-1]) if not hist_line.empty else None

    except Exception as e:
        logger.exception("get_technical_indicators failed for %s", symbol)
        out["error"] = str(e)
    return out


def get_market_overview() -> dict:
    """
    Get broad market overview: S&P 500 (^GSPC), VIX (^VIX), and sector ETFs.

    Returns:
        Dict with sp500, vix, sectors (list of {symbol, name, price, change_pct}), error.
    """
    out: dict = {
        "sp500": None,
        "vix": None,
        "sectors": [],
        "error": None,
    }
    if not HAS_YFINANCE:
        out["error"] = "yfinance is not installed"
        return out

    sector_etfs = [
        ("XLK", "Technology"),
        ("XLF", "Financials"),
        ("XLV", "Healthcare"),
        ("XLE", "Energy"),
        ("XLY", "Consumer Discretionary"),
        ("XLP", "Consumer Staples"),
        ("XLI", "Industrials"),
        ("XLB", "Materials"),
        ("XLRE", "Real Estate"),
        ("XLC", "Communications"),
        ("XLU", "Utilities"),
    ]

    try:
        sp = yf.Ticker("^GSPC")
        sp_hist = sp.history(period="5d")
        if sp_hist is not None and not sp_hist.empty:
            last = sp_hist["Close"].iloc[-1]
            prev = sp_hist["Close"].iloc[-2] if len(sp_hist) >= 2 else last
            ch = ((last - prev) / prev * 100) if prev else 0
            out["sp500"] = {"price": float(last), "change_pct": round(ch, 2)}

        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="5d")
        if vix_hist is not None and not vix_hist.empty:
            out["vix"] = float(vix_hist["Close"].iloc[-1])

        for sym, name in sector_etfs:
            try:
                t = yf.Ticker(sym)
                h = t.history(period="5d")
                if h is not None and not h.empty and len(h) >= 2:
                    last_p = float(h["Close"].iloc[-1])
                    prev_p = float(h["Close"].iloc[-2])
                    ch_pct = (last_p - prev_p) / prev_p * 100
                    out["sectors"].append({
                        "symbol": sym,
                        "name": name,
                        "price": last_p,
                        "change_pct": round(ch_pct, 2),
                    })
            except Exception:
                continue

    except Exception as e:
        logger.exception("get_market_overview failed")
        out["error"] = str(e)
    return out


# LangChain @tool wrappers for agent use
if HAS_LANGCHAIN_TOOLS:

    @tool
    def get_stock_price_tool(symbol: str) -> dict:
        """Get current stock price and basic info for a given ticker symbol (e.g. AAPL, MSFT)."""
        return get_stock_price(symbol)

    @tool
    def get_technical_indicators_tool(symbol: str, period: str = "3mo") -> dict:
        """Get technical indicators (RSI, MACD, EMA 20/50) for a stock symbol."""
        return get_technical_indicators(symbol, period)

    @tool
    def get_market_overview_tool() -> dict:
        """Get market overview: S&P 500 level, VIX, and sector ETF performance."""
        return get_market_overview()

    def get_market_tools() -> list:
        """Return list of LangChain tools for market data."""
        return [get_stock_price_tool, get_technical_indicators_tool, get_market_overview_tool]
else:

    def get_market_tools() -> list:
        """Return empty list when LangChain tools not available."""
        return []
