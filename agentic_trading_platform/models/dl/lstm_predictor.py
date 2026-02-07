"""LSTM Trend Predictor - predicts 5-day trend direction."""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import yfinance as yf
except ImportError:
    yf = None


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


class LSTMTrendPredictor:
    """Predicts 5-day trend direction using LSTM neural network."""

    TREND_UP = 2
    TREND_DOWN = 0
    TREND_SIDEWAYS = 1
    THRESHOLD = 0.01  # 1% for up/down

    def __init__(self, sequence_length: int = 30, n_features: int = 10):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.is_trained = False

        if HAS_TENSORFLOW:
            self.model = self._build_model()

    def _build_model(self) -> "Optional[Sequential]":
        """Build LSTM model: 2 layers, 64 units."""
        if not HAS_TENSORFLOW:
            return None
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(3, activation="softmax"),  # Down, Sideways, Up
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def prepare_data(self, symbol: str, period: str = "2y") -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from yfinance: features and one-hot labels from 5-day forward return."""
        if not yf:
            logger.warning("yfinance not installed")
            return np.array([]).reshape(0, self.sequence_length, self.n_features), np.array([]).reshape(0, 3)
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if df is None or df.empty or len(df) < self.sequence_length + 10:
                return np.array([]).reshape(0, self.sequence_length, self.n_features), np.array([]).reshape(0, 3)
            df = df.sort_index()
            open_ = df["Open"]
            high = df["High"]
            low = df["Low"]
            close = df["Close"]
            volume = df["Volume"]

            # Returns (normalized)
            ret = close.pct_change().fillna(0)
            open_ret = open_.pct_change().fillna(0)
            high_ret = high.pct_change().fillna(0)
            low_ret = low.pct_change().fillna(0)

            # Volume z-score
            vol_mean = volume.rolling(20, min_periods=1).mean()
            vol_std = volume.rolling(20, min_periods=1).std().replace(0, np.nan).fillna(1)
            volume_z = ((volume - vol_mean) / vol_std).fillna(0)

            rsi = _rsi(close, 14).fillna(50) / 100.0 - 0.5
            macd = _macd_histogram(close).fillna(0)
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema_dist = ((close - ema20) / ema20.replace(0, np.nan)).fillna(0)
            atr = _atr(high, low, close, 14)
            atr_norm = (atr / close.replace(0, np.nan)).fillna(0)
            volatility = ret.rolling(20, min_periods=1).std().fillna(0) * np.sqrt(252)

            # 10 features
            features = pd.DataFrame({
                "open_ret": open_ret,
                "high_ret": high_ret,
                "low_ret": low_ret,
                "close_ret": ret,
                "volume_z": volume_z,
                "rsi": rsi,
                "macd": macd,
                "ema_dist": ema_dist,
                "atr": atr_norm,
                "volatility": volatility,
            }, index=df.index)
            features = features.ffill().bfill().fillna(0)

            # 5-day forward return for labels (aligned to features)
            fwd = (close.shift(-5) / close - 1).reindex(features.index)
            valid_idx = fwd.dropna().index
            features = features.loc[features.index.intersection(valid_idx)]
            fwd = fwd.loc[features.index]

            # Labels: 0=Down (<-1%), 1=Sideways, 2=Up (>1%)
            labels = np.where(
                fwd > self.THRESHOLD, self.TREND_UP,
                np.where(fwd < -self.THRESHOLD, self.TREND_DOWN, self.TREND_SIDEWAYS),
            )
            labels = labels.astype(int)

            # Sliding windows: label = 5-day return from last day in window
            X_list, y_list = [], []
            for i in range(len(features) - self.sequence_length):
                X_list.append(features.iloc[i : i + self.sequence_length].values)
                y_list.append(labels[i + self.sequence_length - 1])
            if not X_list:
                return np.array([]).reshape(0, self.sequence_length, self.n_features), np.array([]).reshape(0, 3)

            X = np.array(X_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.int32)
            # One-hot encode
            y_onehot = np.zeros((len(y), 3), dtype=np.float32)
            y_onehot[np.arange(len(y)), y] = 1
            return X, y_onehot
        except Exception as e:
            logger.exception("prepare_data failed for %s: %s", symbol, e)
            return np.array([]).reshape(0, self.sequence_length, self.n_features), np.array([]).reshape(0, 3)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, validation_split: float = 0.2):
        """Train the model."""
        if not HAS_TENSORFLOW or self.model is None:
            logger.warning("TensorFlow not available or model not built")
            return
        if X.size == 0 or y.size == 0:
            logger.warning("Empty training data")
            return
        early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        self.model.fit(X, y, epochs=epochs, validation_split=validation_split, callbacks=[early], verbose=0)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict trend direction. Returns probabilities for [Down, Sideways, Up]."""
        if not HAS_TENSORFLOW or self.model is None:
            return np.tile([1 / 3, 1 / 3, 1 / 3], (len(X), 1)).astype(np.float32) if len(X) else np.array([]).reshape(0, 3)
        return self.model.predict(X, verbose=0)

    def predict_symbol(self, symbol: str) -> Dict:
        """Predict trend for a specific symbol using latest data."""
        out = {"symbol": symbol, "prob_down": None, "prob_sideways": None, "prob_up": None, "trend": None, "error": None}
        if not yf:
            out["error"] = "yfinance not installed"
            return out
        X, _ = self.prepare_data(symbol, period="1y")
        if X is None or len(X) == 0:
            out["error"] = "Insufficient data"
            return out
        probs = self.predict(X[-1:])
        if probs.size == 0:
            return out
        p = probs[0]
        out["prob_down"], out["prob_sideways"], out["prob_up"] = float(p[0]), float(p[1]), float(p[2])
        out["trend"] = ["down", "sideways", "up"][int(np.argmax(p))]
        return out

    def save_weights(self, path: str):
        """Save model weights."""
        if HAS_TENSORFLOW and self.model is not None:
            self.model.save_weights(path)

    def load_weights(self, path: str):
        """Load model weights."""
        if HAS_TENSORFLOW and self.model is not None:
            self.model.load_weights(path)
            self.is_trained = True
