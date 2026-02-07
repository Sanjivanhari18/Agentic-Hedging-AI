"""
Behavioral feature engineering - converts trades into behavior signals.

This module creates two distinct feature categories:
1. Market Context Features: Derived from OHLCV data (computed in market_data.py)
2. Behavioral Features: Derived from trading actions and decisions

All features are designed for behavioral analysis, not prediction or optimization.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BehavioralFeatureEngineer:
    """
    Engineers behavioral features from trade data.
    
    Features are clearly separated into:
    - Market-derived features (from OHLCV indicators)
    - Behavior-derived features (from trading decisions)
    """
    
    def __init__(self):
        self.features: Optional[pd.DataFrame] = None
    
    def engineer_features(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral features from enriched trade data.
        
        Feature Categories:
        1. Market Context Features (already computed in market_data.py):
           - EMA(20), EMA(50), EMA slopes
           - RSI(14)
           - ATR(14), volatility metrics
           - Volume Z-score
           - Market regime labels
        
        2. Behavioral Features (computed here):
           - Entry/exit price relative to EMA
           - Holding duration vs volatility
           - Position size normalized by volatility
           - Trade frequency per rolling window
           - Time gap between consecutive trades
        
        Args:
            trades: DataFrame with trade data enriched with market indicators
            
        Returns:
            DataFrame with all behavioral features
        """
        df = trades.copy()
        
        # Validate required market context columns
        required_market_cols = ['ema_20', 'ema_50', 'atr_14', 'volatility_rolling_std']
        missing = set(required_market_cols) - set(df.columns)
        if missing:
            logger.warning(f"Missing market data columns: {missing}. Some features may be unavailable.")
            # Fill with defaults to allow partial feature engineering
            for col in missing:
                df[col] = 0.0
        
        # Behavioral Feature Engineering
        df = self._add_entry_exit_ema_features(df)
        df = self._add_holding_duration_features(df)
        df = self._add_position_sizing_features(df)
        df = self._add_trade_frequency_features(df)
        df = self._add_time_gap_features(df)
        df = self._add_post_loss_features(df)
        
        self.features = df
        return df
    
    def _add_entry_exit_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add behavioral features: Entry and exit price relative to EMA.
        
        Features:
        - entry_price_distance_from_ema20: Distance of entry price from EMA(20)
        - entry_price_distance_from_ema50: Distance of entry price from EMA(50)
        - entry_price_above_ema20: Binary indicator if entry above EMA(20)
        - entry_price_above_ema50: Binary indicator if entry above EMA(50)
        - exit_price_distance_from_ema20: Distance of exit price from EMA(20) (for sells)
        - exit_price_distance_from_ema50: Distance of exit price from EMA(50) (for sells)
        - exit_price_above_ema20: Binary indicator if exit above EMA(20) (for sells)
        - exit_price_above_ema50: Binary indicator if exit above EMA(50) (for sells)
        
        These features capture decision-making behavior relative to market trends.
        """
        df = df.copy()
        
        # Entry price features (for all trades)
        if 'price' in df.columns and 'ema_20' in df.columns:
            # Distance as percentage: (price - ema) / ema
            df['entry_price_distance_from_ema20'] = (
                (df['price'] - df['ema_20']) / (df['ema_20'] + 1e-6)
            )
            df['entry_price_above_ema20'] = (df['price'] > df['ema_20']).astype(int)
        
        if 'price' in df.columns and 'ema_50' in df.columns:
            df['entry_price_distance_from_ema50'] = (
                (df['price'] - df['ema_50']) / (df['ema_50'] + 1e-6)
            )
            df['entry_price_above_ema50'] = (df['price'] > df['ema_50']).astype(int)
        
        # Exit price features (for sell trades only)
        sell_mask = df['side'] == 'sell'
        
        if 'price' in df.columns and 'ema_20' in df.columns:
            df['exit_price_distance_from_ema20'] = 0.0
            df.loc[sell_mask, 'exit_price_distance_from_ema20'] = (
                (df.loc[sell_mask, 'price'] - df.loc[sell_mask, 'ema_20']) / 
                (df.loc[sell_mask, 'ema_20'] + 1e-6)
            )
            df['exit_price_above_ema20'] = 0
            df.loc[sell_mask, 'exit_price_above_ema20'] = (
                (df.loc[sell_mask, 'price'] > df.loc[sell_mask, 'ema_20']).astype(int)
            )
        
        if 'price' in df.columns and 'ema_50' in df.columns:
            df['exit_price_distance_from_ema50'] = 0.0
            df.loc[sell_mask, 'exit_price_distance_from_ema50'] = (
                (df.loc[sell_mask, 'price'] - df.loc[sell_mask, 'ema_50']) / 
                (df.loc[sell_mask, 'ema_50'] + 1e-6)
            )
            df['exit_price_above_ema50'] = 0
            df.loc[sell_mask, 'exit_price_above_ema50'] = (
                (df.loc[sell_mask, 'price'] > df.loc[sell_mask, 'ema_50']).astype(int)
            )
        
        return df
    
    def _add_holding_duration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add behavioral features: Holding duration relative to volatility.
        
        Features:
        - holding_duration_days: Number of days position was held
        - holding_duration_vs_volatility: Holding duration normalized by volatility
        - holding_duration_vs_atr: Holding duration normalized by ATR
        
        These features capture patience/discipline behavior in different volatility conditions.
        """
        df = df.copy()
        
        # Ensure holding_duration exists and is numeric
        if 'holding_duration' in df.columns:
            df['holding_duration'] = df['holding_duration'].fillna(0).astype(float)
            df['holding_duration_days'] = df['holding_duration']
        else:
            df['holding_duration_days'] = 0.0
            df['holding_duration'] = 0.0
        
        # Holding duration vs volatility (risk-adjusted holding period)
        if 'volatility_rolling_std' in df.columns:
            df['holding_duration_vs_volatility'] = (
                df['holding_duration'] / (df['volatility_rolling_std'] + 1e-6)
            )
        
        if 'atr_14' in df.columns:
            df['holding_duration_vs_atr'] = (
                df['holding_duration'] / (df['atr_14'] + 1e-6)
            )
        
        return df
    
    def _add_position_sizing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add behavioral features: Position size normalized by volatility.
        
        Features:
        - position_size_dollar_value: Dollar value of position (quantity * price)
        - position_size_normalized_by_volatility: Position size / (volatility * quantity)
        - position_size_normalized_by_atr: Position size / (ATR * quantity)
        
        These features capture risk-adjusted position sizing behavior.
        """
        df = df.copy()
        
        # Position size in dollar terms
        if 'quantity' in df.columns and 'price' in df.columns:
            df['position_size_dollar_value'] = df['quantity'] * df['price']
        else:
            df['position_size_dollar_value'] = 0.0
        
        # Risk-adjusted position sizing
        if 'volatility_rolling_std' in df.columns and 'quantity' in df.columns:
            df['position_size_normalized_by_volatility'] = (
                df['position_size_dollar_value'] / 
                ((df['volatility_rolling_std'] * df['quantity']) + 1e-6)
            )
        
        if 'atr_14' in df.columns and 'quantity' in df.columns:
            df['position_size_normalized_by_atr'] = (
                df['position_size_dollar_value'] / 
                ((df['atr_14'] * df['quantity']) + 1e-6)
            )
        
        return df
    
    def _add_trade_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add behavioral features: Trade frequency per rolling window.
        
        Features:
        - trades_per_day: Number of trades executed on the same day
        - trades_per_rolling_7days: Number of trades in rolling 7-day window
        - trades_per_rolling_30days: Number of trades in rolling 30-day window
        
        These features capture trading frequency behavior over time.
        """
        df = df.copy()
        
        # Ensure date column exists
        if 'date' not in df.columns:
            logger.warning("'date' column not found. Skipping trade frequency features.")
            return df
        
        # Trades per day
        df['date_only'] = df['date'].dt.date
        trades_per_day = df.groupby('date_only').size()
        df['trades_per_day'] = df['date_only'].map(trades_per_day).fillna(0).astype(int)
        
        # Rolling window trade frequency
        df = df.sort_values('date').reset_index(drop=True)
        
        # Trades per rolling 7-day window
        df['trades_per_rolling_7days'] = 0
        for idx in range(len(df)):
            window_start = df.loc[idx, 'date'] - pd.Timedelta(days=7)
            window_trades = df[(df['date'] >= window_start) & (df['date'] <= df.loc[idx, 'date'])]
            df.loc[idx, 'trades_per_rolling_7days'] = len(window_trades)
        
        # Trades per rolling 30-day window
        df['trades_per_rolling_30days'] = 0
        for idx in range(len(df)):
            window_start = df.loc[idx, 'date'] - pd.Timedelta(days=30)
            window_trades = df[(df['date'] >= window_start) & (df['date'] <= df.loc[idx, 'date'])]
            df.loc[idx, 'trades_per_rolling_30days'] = len(window_trades)
        
        return df
    
    def _add_time_gap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add behavioral features: Time gap between consecutive trades.
        
        Features:
        - time_gap_hours_since_last_trade: Hours since previous trade
        - time_gap_days_since_last_trade: Days since previous trade
        
        These features capture timing behavior and potential emotional reactions.
        """
        df = df.copy()
        
        # Ensure date column exists
        if 'date' not in df.columns:
            logger.warning("'date' column not found. Skipping time gap features.")
            return df
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Time gap in hours
        df['time_gap_hours_since_last_trade'] = (
            df['date'].diff().dt.total_seconds() / 3600
        ).fillna(0)
        
        # Time gap in days
        df['time_gap_days_since_last_trade'] = (
            df['date'].diff().dt.days
        ).fillna(0)
        
        return df
    
    def _add_post_loss_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add behavioral features: Post-loss trading behavior.
        
        Features:
        - is_loss: Binary indicator if trade resulted in a loss (realized_pnl < 0)
        - trades_after_loss: Number of trades executed within 48 hours after a loss
        - size_after_loss: Position size after a loss (for revenge trading detection)
        
        These features capture emotional reactions to losses (revenge trading, risk escalation).
        """
        df = df.copy()
        
        # Ensure date column exists
        if 'date' not in df.columns:
            logger.warning("'date' column not found. Skipping post-loss features.")
            df['is_loss'] = 0
            df['trades_after_loss'] = 0
            df['size_after_loss'] = 0.0
            return df
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Identify loss trades
        if 'realized_pnl' in df.columns:
            df['is_loss'] = (df['realized_pnl'] < 0).astype(int)
        else:
            df['is_loss'] = 0
        
        # Count trades after loss (within 48 hours)
        df['trades_after_loss'] = 0
        for idx in range(len(df)):
            if idx > 0 and df.loc[idx - 1, 'is_loss'] == 1:
                # Previous trade was a loss, count trades in next 48 hours
                loss_date = df.loc[idx - 1, 'date']
                window_end = loss_date + pd.Timedelta(hours=48)
                future_trades = df[(df['date'] > loss_date) & (df['date'] <= window_end)]
                df.loc[idx, 'trades_after_loss'] = len(future_trades)
        
        # Position size after loss (for risk escalation detection)
        df['size_after_loss'] = 0.0
        if 'position_size_normalized_by_volatility' in df.columns:
            for idx in range(len(df)):
                if idx > 0 and df.loc[idx - 1, 'is_loss'] == 1:
                    # This trade occurred after a loss
                    df.loc[idx, 'size_after_loss'] = df.loc[idx, 'position_size_normalized_by_volatility']
        
        return df
    
    def get_features(self) -> pd.DataFrame:
        """Get engineered features."""
        if self.features is None:
            raise ValueError("No features available. Call engineer_features() first.")
        return self.features
    
    def get_feature_summary(self) -> dict:
        """
        Get summary of all features by category.
        
        Returns:
            Dictionary with feature categories and their feature names
        """
        if self.features is None:
            raise ValueError("No features available. Call engineer_features() first.")
        
        # Market context features (from market_data.py)
        market_features = [
            'ema_20', 'ema_50', 'ema_20_slope', 'ema_50_slope',
            'rsi_14',
            'atr_14', 'volatility_rolling_std',
            'volume_raw', 'volume_zscore',
            'trend_regime', 'volatility_regime'
        ]
        
        # Behavioral features (computed here)
        behavioral_features = [
            'entry_price_distance_from_ema20', 'entry_price_distance_from_ema50',
            'entry_price_above_ema20', 'entry_price_above_ema50',
            'exit_price_distance_from_ema20', 'exit_price_distance_from_ema50',
            'exit_price_above_ema20', 'exit_price_above_ema50',
            'holding_duration_days', 'holding_duration_vs_volatility', 'holding_duration_vs_atr',
            'position_size_dollar_value', 'position_size_normalized_by_volatility', 
            'position_size_normalized_by_atr',
            'trades_per_day', 'trades_per_rolling_7days', 'trades_per_rolling_30days',
            'time_gap_hours_since_last_trade', 'time_gap_days_since_last_trade'
        ]
        
        # Filter to only features that exist in the DataFrame
        available_market = [f for f in market_features if f in self.features.columns]
        available_behavioral = [f for f in behavioral_features if f in self.features.columns]
        
        return {
            'market_context_features': available_market,
            'behavioral_features': available_behavioral,
            'total_features': len(available_market) + len(available_behavioral)
        }
