"""
Market data fetching and indicator calculation from OHLCV only.

IMPORTANT: All indicators (RSI, MACD, EMA, ATR, etc.) are computed internally
from OHLCV data only. There are NO external indicator databases or 
pre-calculated values used. The system:
1. Fetches raw OHLCV data (Open, High, Low, Close, Volume)
2. Computes all indicators from scratch using standard formulas
3. Provides complete transparency and control over all calculations

No external indicator libraries (like pandas-ta) are used.
All calculations are implemented directly from OHLCV data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    Fetches OHLCV data and computes market indicators.
    
    All indicators are computed internally from OHLCV data only.
    No external indicator databases or pre-calculated values are used.
    
    Computed Indicators:
    - EMA(20), EMA(50) from close prices
    - RSI(14) from close price deltas
    - MACD(12, 26, 9) from close prices
    - ATR(14) from High, Low, Close
    - Volatility (rolling std of returns)
    - Volume Z-score
    """
    
    def __init__(self):
        self.cache: Dict[str, pd.DataFrame] = {}
    
    def fetch_ohlcv(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Returns DataFrame with columns: Open, High, Low, Close, Volume
        """
        cache_key = f"{symbol}_{start_date}_{end_date}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {symbol} between {start_date} and {end_date}")
        
        # Ensure we have required OHLCV columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")
        
        self.cache[cache_key] = df
        return df
    
    def compute_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Compute Exponential Moving Average (EMA) from price series.
        
        Args:
            prices: Price series (typically Close prices)
            period: EMA period (e.g., 20, 50)
            
        Returns:
            EMA series
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    def compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Compute Relative Strength Index (RSI) from close price deltas.
        
        RSI = 100 - (100 / (1 + RS))
        where RS = average gain / average loss over period
        
        Args:
            prices: Close prices
            period: RSI period (default 14)
            
        Returns:
            RSI series (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Compute Average True Range (ATR) - volatility metric.
        
        True Range = max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
        ATR = moving average of True Range
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period (default 14)
            
        Returns:
            ATR series
        """
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def compute_volatility(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Compute rolling standard deviation of returns (volatility metric).
        
        Args:
            prices: Close prices
            window: Rolling window size (default 14)
            
        Returns:
            Volatility series (standard deviation of returns)
        """
        returns = prices.pct_change()
        volatility = returns.rolling(window=window).std()
        
        return volatility
    
    def compute_volume_zscore(self, volume: pd.Series, window: int = 20) -> pd.Series:
        """
        Compute volume Z-score (standardized volume).
        
        Z-score = (volume - rolling_mean) / rolling_std
        
        Args:
            volume: Volume series
            window: Rolling window size (default 20)
            
        Returns:
            Volume Z-score series
        """
        rolling_mean = volume.rolling(window=window).mean()
        rolling_std = volume.rolling(window=window).std()
        
        zscore = (volume - rolling_mean) / (rolling_std + 1e-10)  # Avoid division by zero
        
        return zscore
    
    def compute_macd(self, prices: pd.Series, fast_period: int = 12, 
                     slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Compute MACD (Moving Average Convergence Divergence) from close prices.
        
        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(signal_period) of MACD Line
        Histogram = MACD Line - Signal Line
        
        Args:
            prices: Close prices
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)
            
        Returns:
            DataFrame with columns: macd_line, macd_signal, macd_histogram
        """
        # Compute fast and slow EMAs
        ema_fast = self.compute_ema(prices, period=fast_period)
        ema_slow = self.compute_ema(prices, period=slow_period)
        
        # MACD line = fast EMA - slow EMA
        macd_line = ema_fast - ema_slow
        
        # Signal line = EMA of MACD line
        macd_signal = self.compute_ema(macd_line, period=signal_period)
        
        # Histogram = MACD line - Signal line
        macd_histogram = macd_line - macd_signal
        
        # Return as DataFrame
        result = pd.DataFrame({
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        })
        
        return result
    
    def calculate_market_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all market indicators from OHLCV data.
        
        Market Context Features (ALL computed internally from OHLCV only):
        - EMA(20) and EMA(50) from close prices
        - EMA slope (rate of change)
        - RSI(14) from close price deltas
        - MACD (12, 26, 9) - MACD line, signal line, and histogram
        - ATR(14) - volatility metric
        - Rolling volatility (std of returns)
        - Raw volume
        - Volume Z-score
        
        Note: All indicators are computed from OHLCV data only.
        No external indicator databases or pre-calculated values are used.
        
        Args:
            df: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)
            
        Returns:
            DataFrame with added indicator columns
        """
        result = df.copy()
        
        # EMA(20) and EMA(50) from close prices
        result['ema_20'] = self.compute_ema(result['Close'], period=20)
        result['ema_50'] = self.compute_ema(result['Close'], period=50)
        
        # EMA slope (rate of change) - percentage change over 1 period
        result['ema_20_slope'] = result['ema_20'].pct_change()
        result['ema_50_slope'] = result['ema_50'].pct_change()
        
        # RSI(14) computed from close price deltas
        result['rsi_14'] = self.compute_rsi(result['Close'], period=14)
        
        # MACD (12, 26, 9) computed from close prices
        macd_data = self.compute_macd(result['Close'], fast_period=12, slow_period=26, signal_period=9)
        result['macd_line'] = macd_data['macd_line']
        result['macd_signal'] = macd_data['macd_signal']
        result['macd_histogram'] = macd_data['macd_histogram']
        
        # Volatility metrics
        result['atr_14'] = self.compute_atr(result['High'], result['Low'], result['Close'], period=14)
        result['volatility_rolling_std'] = self.compute_volatility(result['Close'], window=14)
        
        # Volume features
        result['volume_raw'] = result['Volume']
        result['volume_zscore'] = self.compute_volume_zscore(result['Volume'], window=20)
        
        # Fill NaN values (forward fill then backward fill)
        result = result.bfill().ffill()
        
        return result
    
    def label_market_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label market regimes based on analytical rules (not predictive).
        
        Trend Regime:
        - uptrend: EMA(20) > EMA(50) AND EMA(20) slope > 0
        - downtrend: EMA(20) < EMA(50) AND EMA(20) slope < 0
        - sideways: otherwise
        
        Volatility Regime:
        - low: volatility in bottom 33rd percentile
        - medium: volatility in middle 33rd percentile
        - high: volatility in top 33rd percentile
        
        Args:
            df: DataFrame with market indicators
            
        Returns:
            DataFrame with regime label columns
        """
        result = df.copy()
        
        # Trend regime labels
        ema_above = result['ema_20'] > result['ema_50']
        ema_slope_positive = result['ema_20_slope'] > 0
        ema_slope_negative = result['ema_20_slope'] < 0
        
        result['trend_regime'] = 'sideways'  # Default
        result.loc[ema_above & ema_slope_positive, 'trend_regime'] = 'uptrend'
        result.loc[~ema_above & ema_slope_negative, 'trend_regime'] = 'downtrend'
        
        # Volatility regime labels (based on rolling std of returns)
        vol_col = 'volatility_rolling_std'
        if vol_col in result.columns:
            vol_33 = result[vol_col].quantile(0.33)
            vol_67 = result[vol_col].quantile(0.67)
            
            result['volatility_regime'] = 'medium'  # Default
            result.loc[result[vol_col] <= vol_33, 'volatility_regime'] = 'low'
            result.loc[result[vol_col] >= vol_67, 'volatility_regime'] = 'high'
        
        return result
    
    def enrich_trades_with_market_data(
        self, 
        trades: pd.DataFrame, 
        symbol: Optional[str] = None,
        date_col: str = 'date',
        symbol_col: str = 'symbol'
    ) -> pd.DataFrame:
        """
        Enrich trade data with market context at trade time.
        
        Now supports multiple symbols automatically:
        - If symbol is provided, uses that symbol for all trades (backward compatibility)
        - If symbol is None, automatically detects all unique symbols in trades
          and fetches OHLCV for each symbol separately
        
        Args:
            trades: DataFrame with trade data
            symbol: Stock symbol (optional - if None, uses symbol column from trades)
            date_col: Name of date column in trades
            symbol_col: Name of symbol column in trades
            
        Returns:
            Enriched trades DataFrame with market indicators and regime labels
        """
        trades = trades.copy()
        trades[date_col] = pd.to_datetime(trades[date_col])
        
        # Check if symbol column exists
        if symbol_col not in trades.columns:
            if symbol is None:
                raise ValueError(f"Symbol column '{symbol_col}' not found in trades. Either provide symbol parameter or ensure trades have a '{symbol_col}' column.")
            # If symbol provided but no symbol column, use provided symbol for all
            logger.info(f"Using provided symbol '{symbol}' for all trades (no symbol column found)")
            unique_symbols = [symbol]
            trades[symbol_col] = symbol
        else:
            # Get unique symbols from trades
            unique_symbols = trades[symbol_col].unique().tolist()
            if symbol is not None:
                # If symbol provided, use it for all trades (backward compatibility)
                logger.info(f"Using provided symbol '{symbol}' for all trades (ignoring symbol column)")
                unique_symbols = [symbol]
                trades[symbol_col] = symbol
            else:
                logger.info(f"Detected {len(unique_symbols)} unique symbols: {unique_symbols}")
        
        # Get date range
        start_date = trades[date_col].min() - pd.Timedelta(days=60)  # Need more history for EMA(50)
        end_date = trades[date_col].max() + pd.Timedelta(days=1)
        
        # Process each symbol separately
        enriched_trades_list = []
        
        for sym in unique_symbols:
            logger.info(f"Processing symbol: {sym}")
            
            # Get trades for this symbol
            symbol_trades = trades[trades[symbol_col] == sym].copy()
            
            if len(symbol_trades) == 0:
                continue
            
            try:
                # Fetch OHLCV data for this symbol
                market_data = self.fetch_ohlcv(sym, str(start_date.date()), str(end_date.date()))
                
                # Calculate indicators from OHLCV only
                market_data = self.calculate_market_indicators(market_data)
                
                # Label market regimes
                market_data = self.label_market_regimes(market_data)
                
                # Prepare for merge
                symbol_trades = symbol_trades.sort_values(date_col)
                market_data = market_data.reset_index()
                market_data['Date'] = pd.to_datetime(market_data['Date'])
                
                # Ensure both date columns are timezone-naive for compatibility
                try:
                    if isinstance(symbol_trades[date_col].dtype, pd.DatetimeTZDtype):
                        symbol_trades[date_col] = symbol_trades[date_col].dt.tz_localize(None)
                    elif len(symbol_trades) > 0 and hasattr(symbol_trades[date_col].iloc[0], 'tz') and symbol_trades[date_col].iloc[0].tz is not None:
                        symbol_trades[date_col] = symbol_trades[date_col].dt.tz_localize(None)
                except (AttributeError, TypeError):
                    pass
                
                try:
                    if isinstance(market_data['Date'].dtype, pd.DatetimeTZDtype):
                        market_data['Date'] = market_data['Date'].dt.tz_localize(None)
                    elif len(market_data) > 0 and hasattr(market_data['Date'].iloc[0], 'tz') and market_data['Date'].iloc[0].tz is not None:
                        market_data['Date'] = market_data['Date'].dt.tz_localize(None)
                except (AttributeError, TypeError):
                    pass
                
                # Select columns to merge (market context features - ALL computed from OHLCV)
                market_cols = [
                    'Date', 
                    'ema_20', 'ema_50', 'ema_20_slope', 'ema_50_slope',
                    'rsi_14',
                    'macd_line', 'macd_signal', 'macd_histogram',
                    'atr_14', 'volatility_rolling_std',
                    'volume_raw', 'volume_zscore',
                    'trend_regime', 'volatility_regime',
                    'Close'  # Keep close price for reference
                ]
                
                # Merge on nearest date
                symbol_enriched = pd.merge_asof(
                    symbol_trades.sort_values(date_col),
                    market_data[market_cols],
                    left_on=date_col,
                    right_on='Date',
                    direction='nearest'
                )
                
                enriched_trades_list.append(symbol_enriched)
                logger.info(f"Successfully enriched {len(symbol_enriched)} trades for {sym}")
                
            except Exception as e:
                logger.warning(f"Failed to enrich trades for symbol {sym}: {e}. Skipping this symbol.")
                # Add trades without enrichment (with NaN indicators)
                enriched_trades_list.append(symbol_trades)
        
        # Combine all enriched trades
        if enriched_trades_list:
            enriched_trades = pd.concat(enriched_trades_list, ignore_index=True)
            enriched_trades = enriched_trades.sort_values(date_col).reset_index(drop=True)
            return enriched_trades
        else:
            raise ValueError("No trades could be enriched with market data")
