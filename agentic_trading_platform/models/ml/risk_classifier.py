"""Risk Classifier - classifies stocks by risk level."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

MARKET_INDEX = "^GSPC"  # S&P 500 for beta


class RiskClassifier:
    """Classifies stocks into risk categories (low/medium/high)."""

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _compute_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Compute beta relative to S&P 500."""
        if returns.empty or market_returns.empty or returns.isna().all():
            return np.nan
        common = returns.align(market_returns, join="inner")
        r, m = common[0].dropna(), common[1].dropna()
        if len(r) < 2 or len(m) < 2:
            return np.nan
        cov = np.cov(r, m)
        if cov[1, 1] == 0:
            return np.nan
        return float(cov[0, 1] / cov[1, 1])

    def _compute_max_drawdown(self, prices: pd.Series) -> float:
        """Compute maximum drawdown."""
        if prices.empty or len(prices) < 2:
            return np.nan
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax.replace(0, np.nan)
        return float(drawdown.min())

    def _compute_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Compute Value at Risk (negative of quantile so positive = loss)."""
        if returns.empty or returns.isna().all():
            return np.nan
        r = returns.dropna()
        if r.empty:
            return np.nan
        q = 1 - confidence
        return float(-r.quantile(q))

    def compute_risk_features(self, symbol: str, period: str = "1y") -> Optional[Dict]:
        """Compute risk features: volatility, beta, max drawdown, VaR."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if df is None or df.empty or len(df) < 20:
                logger.warning("Insufficient data for %s", symbol)
                return None
            df = df.sort_index()
            close = df["Close"]
            returns = close.pct_change().dropna()
            if returns.empty:
                return None

            # Annualized volatility (252 trading days)
            volatility = float(returns.std() * np.sqrt(252))

            # Beta vs S&P 500
            market = yf.Ticker(MARKET_INDEX)
            market_hist = market.history(period=period)
            if market_hist is not None and not market_hist.empty:
                market_returns = market_hist["Close"].sort_index().pct_change().dropna()
                beta = self._compute_beta(returns, market_returns)
            else:
                beta = np.nan

            max_drawdown = self._compute_max_drawdown(close)
            var_95 = self._compute_var(returns, 0.95)

            return {
                "symbol": symbol,
                "volatility": volatility,
                "beta": beta,
                "max_drawdown": max_drawdown,
                "var_95": var_95,
            }
        except Exception as e:
            logger.exception("Error computing risk features for %s: %s", symbol, e)
            return None

    def classify_risk(self, symbols: List[str]) -> pd.DataFrame:
        """Classify risk level for a list of stocks."""
        rows = []
        for symbol in symbols:
            feats = self.compute_risk_features(symbol)
            if feats is None:
                continue
            rows.append(feats)

        if not rows:
            return pd.DataFrame(columns=["symbol", "risk_level", "volatility", "beta", "max_drawdown"])

        df = pd.DataFrame(rows)
        vol = df["volatility"].replace([np.inf, -np.inf], np.nan).dropna()
        if vol.empty:
            df["risk_level"] = "medium"
            return df[["symbol", "risk_level", "volatility", "beta", "max_drawdown"]]

        # Classify by volatility terciles: low = bottom third, medium = middle, high = top third
        p33 = vol.quantile(1 / 3)
        p67 = vol.quantile(2 / 3)
        def level(v):
            if np.isnan(v) or v <= p33:
                return "low"
            if v <= p67:
                return "medium"
            return "high"

        df["risk_level"] = df["volatility"].apply(level)
        return df[["symbol", "risk_level", "volatility", "beta", "max_drawdown"]]
