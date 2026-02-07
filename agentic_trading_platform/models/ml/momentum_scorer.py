"""Momentum Scorer - ranks stocks by momentum signals."""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram = MACD line - Signal line."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _rate_of_change(series: pd.Series, period: int) -> pd.Series:
    """Rate of change: (current - period_ago) / period_ago."""
    return (series - series.shift(period)) / series.shift(period).replace(0, np.nan)


class MomentumScorer:
    """Ranks stocks by momentum using technical indicators."""

    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def compute_momentum_features(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Fetch data and compute momentum features for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if df is None or df.empty or len(df) < 50:
                logger.warning("Insufficient data for %s", symbol)
                return None
            df = df.sort_index()
            close = df["Close"]
            volume = df["Volume"]

            # RSI(14)
            rsi = _rsi(close, 14)

            # MACD histogram
            macd_hist = _macd_histogram(close)

            # EMAs and price vs EMA ratios (as % above/below)
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            price_vs_ema20 = (close - ema20) / ema20.replace(0, np.nan)
            price_vs_ema50 = (close - ema50) / ema50.replace(0, np.nan)

            # Rate of change 10-day, 20-day
            roc10 = _rate_of_change(close, 10)
            roc20 = _rate_of_change(close, 20)

            # Volume z-score (volume trend)
            vol_mean = volume.rolling(20, min_periods=1).mean()
            vol_std = volume.rolling(20, min_periods=1).std().replace(0, np.nan)
            volume_z = (volume - vol_mean) / vol_std

            features = pd.DataFrame(
                {
                    "rsi": rsi,
                    "macd_hist": macd_hist,
                    "price_vs_ema20": price_vs_ema20,
                    "price_vs_ema50": price_vs_ema50,
                    "roc10": roc10,
                    "roc20": roc20,
                    "volume_z": volume_z,
                },
                index=df.index,
            )
            features = features.dropna(how="all").ffill().bfill()
            return features
        except Exception as e:
            logger.exception("Error computing momentum features for %s: %s", symbol, e)
            return None

    def score_stocks(self, symbols: List[str]) -> pd.DataFrame:
        """Score and rank a list of stocks by momentum."""
        rows = []
        for symbol in symbols:
            feats = self.compute_momentum_features(symbol)
            if feats is None or feats.empty:
                continue
            # Use latest row for scoring
            last = feats.iloc[-1]
            # Weights: RSI (neutral 50), MACD+, EMAs+, ROC+, volume trend
            rsi_norm = 1 - abs(last["rsi"] - 50) / 50 if not np.isnan(last["rsi"]) else 0.5
            macd_norm = np.clip((last["macd_hist"] + 2) / 4, 0, 1) if not np.isnan(last["macd_hist"]) else 0.5
            ema20_norm = np.clip((last["price_vs_ema20"] + 0.1) / 0.2, 0, 1) if not np.isnan(last["price_vs_ema20"]) else 0.5
            ema50_norm = np.clip((last["price_vs_ema50"] + 0.1) / 0.2, 0, 1) if not np.isnan(last["price_vs_ema50"]) else 0.5
            roc10_norm = np.clip((last["roc10"] + 0.2) / 0.4, 0, 1) if not np.isnan(last["roc10"]) else 0.5
            roc20_norm = np.clip((last["roc20"] + 0.2) / 0.4, 0, 1) if not np.isnan(last["roc20"]) else 0.5
            vol_norm = np.clip((last["volume_z"] + 2) / 4, 0, 1) if not np.isnan(last["volume_z"]) else 0.5

            weights = [0.15, 0.2, 0.15, 0.15, 0.15, 0.1, 0.1]
            score = (
                rsi_norm * weights[0]
                + macd_norm * weights[1]
                + ema20_norm * weights[2]
                + ema50_norm * weights[3]
                + roc10_norm * weights[4]
                + roc20_norm * weights[5]
                + vol_norm * weights[6]
            )
            rows.append({"symbol": symbol, "score": float(score)})

        if not rows:
            return pd.DataFrame(columns=["symbol", "score", "rank"])

        out = pd.DataFrame(rows)
        out["rank"] = out["score"].rank(ascending=False, method="min").astype(int)
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
        return out[["symbol", "score", "rank"]]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the momentum model on historical data."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict_momentum(self, X: np.ndarray) -> np.ndarray:
        """Predict momentum direction (up/down/sideways)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
