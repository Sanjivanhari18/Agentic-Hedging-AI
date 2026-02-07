"""
Behavioral Stability / Consistency Score Analyzer.

This module calculates a behavioral stability index that measures how consistent
a trader's behavior is over time. This is a non-judgmental metric that does not
measure skill or profitability - only consistency of behavior.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BehavioralStabilityAnalyzer:
    """
    Analyzes behavioral stability/consistency over time.
    
    Measures variance of:
    - Trade frequency
    - Position size (volatility-adjusted)
    - Holding duration
    
    Across rolling windows, normalized to a single "behavioral stability index".
    """
    
    def __init__(self, window_size: int = 30, min_periods: int = 10):
        """
        Args:
            window_size: Rolling window size for stability calculation
            min_periods: Minimum periods required for valid calculation
        """
        self.window_size = window_size
        self.min_periods = min_periods
        self.stability_score: Optional[float] = None
        self.stability_components: Optional[Dict] = None
    
    def calculate_stability_score(self, features: pd.DataFrame) -> Dict:
        """
        Calculate behavioral stability/consistency score.
        
        This score measures how stable a trader's behavior is over time by
        analyzing variance in:
        1. Trade frequency (trades per day)
        2. Position size (volatility-adjusted)
        3. Holding duration
        
        The score is normalized to 0-100, where:
        - Higher scores (70-100) = More consistent behavior
        - Lower scores (0-30) = More variable behavior
        - Medium scores (30-70) = Moderate consistency
        
        IMPORTANT: This score does NOT measure skill or profitability.
        It only measures consistency of behavior patterns.
        
        Args:
            features: DataFrame with behavioral features
            
        Returns:
            Dictionary containing:
            - stability_score: Overall stability index (0-100)
            - components: Individual component scores and variances
            - interpretation: Human-readable interpretation
        """
        if 'date' not in features.columns:
            logger.warning("'date' column not found. Cannot calculate stability score.")
            return self._empty_result()
        
        # Sort by date
        df = features.sort_values('date').reset_index(drop=True)
        
        # Calculate rolling variances for each component
        components = {}
        
        # 1. Trade Frequency Stability
        trade_freq_stability = self._calculate_component_stability(
            df, 'trades_per_day', 'Trade Frequency'
        )
        components['trade_frequency'] = trade_freq_stability
        
        # 2. Position Size Stability (volatility-adjusted)
        position_size_stability = self._calculate_component_stability(
            df, 'position_size_normalized_by_volatility', 'Position Size (Volatility-Adjusted)'
        )
        components['position_size'] = position_size_stability
        
        # 3. Holding Duration Stability
        holding_duration_stability = self._calculate_component_stability(
            df, 'holding_duration_days', 'Holding Duration'
        )
        components['holding_duration'] = holding_duration_stability
        
        # Normalize to 0-100 scale
        # Lower variance = higher stability = higher score
        component_scores = []
        for comp_name, comp_data in components.items():
            if comp_data['variance'] is not None:
                # Use coefficient of variation (CV) for normalization
                # CV = std / mean, but we want to handle cases where mean is near 0
                cv = comp_data['coefficient_of_variation']
                if cv is not None and not np.isnan(cv) and cv > 0:
                    # Convert CV to stability score: lower CV = higher stability
                    # Use inverse relationship: stability = 100 / (1 + CV)
                    # This maps: CV=0 -> 100, CV=1 -> 50, CV=2 -> 33, etc.
                    stability_component = 100 / (1 + cv)
                    # Clamp to 0-100
                    stability_component = max(0, min(100, stability_component))
                    component_scores.append(stability_component)
                    comp_data['stability_score'] = stability_component
        
        # Overall stability score: average of component scores
        if component_scores:
            overall_stability = np.mean(component_scores)
        else:
            overall_stability = 50.0  # Default to medium if no valid components
        
        self.stability_score = overall_stability
        self.stability_components = components
        
        # Generate interpretation
        interpretation = self._generate_interpretation(overall_stability, components)
        
        return {
            'stability_score': float(overall_stability),
            'components': components,
            'interpretation': interpretation,
            'note': "This score does not measure skill or profitability — only consistency of behavior."
        }
    
    def _calculate_component_stability(self, df: pd.DataFrame, 
                                      feature: str, 
                                      feature_name: str) -> Dict:
        """
        Calculate stability metrics for a single behavioral component.
        
        Args:
            df: Features DataFrame
            feature: Feature column name
            feature_name: Human-readable feature name
            
        Returns:
            Dictionary with stability metrics
        """
        if feature not in df.columns:
            return {
                'feature_name': feature_name,
                'variance': None,
                'coefficient_of_variation': None,
                'rolling_variance': None,
                'stability_score': None
            }
        
        # Remove NaN and infinite values
        feature_data = df[feature].replace([np.inf, -np.inf], np.nan)
        valid_data = feature_data.dropna()
        
        if len(valid_data) < self.min_periods:
            return {
                'feature_name': feature_name,
                'variance': None,
                'coefficient_of_variation': None,
                'rolling_variance': None,
                'stability_score': None
            }
        
        # Overall variance
        overall_variance = valid_data.var()
        overall_mean = valid_data.mean()
        
        # Coefficient of variation (normalized variance)
        cv = overall_variance / (abs(overall_mean) + 1e-6) if abs(overall_mean) > 1e-6 else None
        
        # Rolling variance (to see how variance changes over time)
        rolling_var = feature_data.rolling(
            window=self.window_size, 
            min_periods=self.min_periods
        ).var()
        
        # Average rolling variance
        avg_rolling_var = rolling_var.mean() if len(rolling_var.dropna()) > 0 else None
        
        return {
            'feature_name': feature_name,
            'variance': float(overall_variance) if not np.isnan(overall_variance) else None,
            'mean': float(overall_mean) if not np.isnan(overall_mean) else None,
            'coefficient_of_variation': float(cv) if cv is not None and not np.isnan(cv) else None,
            'rolling_variance': float(avg_rolling_var) if avg_rolling_var is not None and not np.isnan(avg_rolling_var) else None,
            'stability_score': None  # Will be calculated in main method
        }
    
    def _generate_interpretation(self, score: float, components: Dict) -> str:
        """
        Generate human-readable interpretation of stability score.
        
        Args:
            score: Overall stability score (0-100)
            components: Component stability data
            
        Returns:
            Interpretation string
        """
        # Overall interpretation
        if score >= 70:
            overall_desc = "highly consistent"
            overall_detail = "Your trading behavior shows strong consistency over time."
        elif score >= 50:
            overall_desc = "moderately consistent"
            overall_detail = "Your trading behavior shows moderate consistency with some variation."
        elif score >= 30:
            overall_desc = "somewhat variable"
            overall_detail = "Your trading behavior shows noticeable variation over time."
        else:
            overall_desc = "highly variable"
            overall_detail = "Your trading behavior shows significant variation over time."
        
        interpretation = f"Behavioral Stability Score: {score:.1f}/100\n"
        interpretation += f"Overall Assessment: {overall_desc.capitalize()}\n\n"
        interpretation += overall_detail + "\n\n"
        
        # Component breakdown
        interpretation += "Component Analysis:\n"
        for comp_name, comp_data in components.items():
            if comp_data['stability_score'] is not None:
                comp_score = comp_data['stability_score']
                comp_name_readable = comp_data['feature_name']
                
                if comp_score >= 70:
                    comp_desc = "very stable"
                elif comp_score >= 50:
                    comp_desc = "moderately stable"
                elif comp_score >= 30:
                    comp_desc = "somewhat variable"
                else:
                    comp_desc = "highly variable"
                
                interpretation += f"  • {comp_name_readable}: {comp_score:.1f}/100 ({comp_desc})\n"
        
        interpretation += "\n"
        interpretation += "Note: This score measures consistency of behavior patterns, "
        interpretation += "not trading skill or profitability. A consistent trader may be "
        interpretation += "consistently profitable or consistently unprofitable."
        
        return interpretation
    
    def _empty_result(self) -> Dict:
        """Return empty result structure when calculation is not possible."""
        return {
            'stability_score': None,
            'components': {},
            'interpretation': "Insufficient data to calculate behavioral stability score.",
            'note': "This score does not measure skill or profitability — only consistency of behavior."
        }
    
    def get_stability_score(self) -> Optional[float]:
        """Get the calculated stability score."""
        return self.stability_score
    
    def get_stability_components(self) -> Optional[Dict]:
        """Get the component stability data."""
        return self.stability_components
