"""Baseline construction using statistical methods."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaselineConstructor:
    """Constructs personalized behavioral baselines using statistics."""
    
    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Rolling window size for baseline calculation
        """
        self.window_size = window_size
        self.baselines: Optional[Dict[str, Dict]] = None
    
    def construct_baselines(self, features: pd.DataFrame) -> Dict[str, Dict]:
        """
        Construct personalized behavioral baselines.
        
        Returns:
            Dictionary with baseline statistics for each feature
        """
        baselines = {}
        
        # Key behavioral features to baseline
        baseline_features = [
            'trades_per_day',
            'trades_per_rolling_7days',
            'trades_per_rolling_30days',
            'position_size_dollar_value',
            'position_size_normalized_by_volatility',
            'holding_duration_days',
            'holding_duration_vs_volatility',
            'time_gap_hours_since_last_trade',
            'time_gap_days_since_last_trade'
        ]
        
        # Regime-conditioned baselines (using new regime labels)
        regimes = ['volatility_regime', 'trend_regime']
        
        for feature in baseline_features:
            if feature not in features.columns:
                continue
            
            # Filter out NaN and infinite values for robust statistics
            feature_data = features[feature].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(feature_data) == 0:
                logger.warning(f"No valid data for feature {feature}, skipping baseline.")
                continue
            
            # Overall baseline (use median as robust central tendency)
            baselines[feature] = {
                'mean': float(feature_data.mean()),
                'median': float(feature_data.median()),
                'std': float(feature_data.std()) if len(feature_data) > 1 else 0.0,
                'q25': float(feature_data.quantile(0.25)),
                'q75': float(feature_data.quantile(0.75)),
                'rolling_mean': self._rolling_stats(features, feature, 'mean'),
                'rolling_std': self._rolling_stats(features, feature, 'std')
            }
            
            # Regime-conditioned baselines
            for regime in regimes:
                if regime not in features.columns:
                    continue
                
                regime_mask = features[regime] == 1
                if regime_mask.sum() > 0:
                    regime_data = features.loc[regime_mask, feature].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(regime_data) > 0:
                        baselines[f"{feature}_{regime}"] = {
                            'mean': float(regime_data.mean()),
                            'median': float(regime_data.median()),
                            'std': float(regime_data.std()) if len(regime_data) > 1 else 0.0,
                            'q25': float(regime_data.quantile(0.25)),
                            'q75': float(regime_data.quantile(0.75))
                        }
        
        self.baselines = baselines
        return baselines
    
    def _rolling_stats(self, df: pd.DataFrame, feature: str, stat: str) -> pd.Series:
        """Calculate rolling statistics."""
        if stat == 'mean':
            return df[feature].rolling(window=self.window_size, min_periods=1).mean()
        elif stat == 'std':
            return df[feature].rolling(window=self.window_size, min_periods=1).std()
        else:
            return pd.Series()
    
    def calculate_deviations(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate deviations from baseline for each trade.
        
        Returns:
            DataFrame with deviation scores
        """
        if self.baselines is None:
            raise ValueError("No baselines available. Call construct_baselines() first.")
        
        df = features.copy()
        
        # Calculate z-scores for key features (using robust median-based z-score)
        for feature in ['trades_per_day', 'position_size_ratio', 'holding_duration', 
                       'time_gap_hours', 'trade_value']:
            if feature in df.columns and feature in self.baselines:
                baseline = self.baselines[feature]
                # Use median and IQR for more robust z-scores (less sensitive to outliers)
                iqr = baseline['q75'] - baseline['q25']
                # Use median absolute deviation (MAD) if std is too small
                mad = baseline['std'] if baseline['std'] > 1e-6 else (iqr / 1.349) if iqr > 1e-6 else 1.0
                df[f'{feature}_zscore'] = (
                    (df[feature] - baseline['median']) / (mad + 1e-6)
                )
                df[f'{feature}_deviation'] = abs(df[f'{feature}_zscore'])
                # Handle NaN and infinite values
                df[f'{feature}_zscore'] = df[f'{feature}_zscore'].replace([np.inf, -np.inf], np.nan).fillna(0)
                df[f'{feature}_deviation'] = df[f'{feature}_deviation'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Regime-conditioned deviations
        for regime in ['high_volatility_regime', 'low_volatility_regime']:
            if regime not in df.columns:
                continue
            
            regime_mask = df[regime] == 1
            for feature in ['trades_per_day', 'position_size_ratio', 'holding_duration']:
                baseline_key = f"{feature}_{regime}"
                if baseline_key in self.baselines and feature in df.columns:
                    baseline = self.baselines[baseline_key]
                    df.loc[regime_mask, f'{feature}_{regime}_zscore'] = (
                        (df.loc[regime_mask, feature] - baseline['mean']) / 
                        (baseline['std'] + 1e-6)
                    )
        
        # Overall deviation score (composite)
        deviation_cols = [col for col in df.columns if col.endswith('_deviation')]
        if deviation_cols:
            df['overall_deviation_score'] = df[deviation_cols].mean(axis=1)
        
        return df
    
    def get_baselines(self) -> Dict[str, Dict]:
        """Get constructed baselines."""
        if self.baselines is None:
            raise ValueError("No baselines available. Call construct_baselines() first.")
        return self.baselines

