"""Explainable AI components for behavioral analysis."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class BehavioralExplainer:
    """Provides explainable insights for behavioral analysis."""
    
    def __init__(self):
        self.explanations: Dict = {}
    
    def explain_trade(self, trade_idx: int, features: pd.DataFrame, 
                     baselines: Dict, clusters: Optional[np.ndarray] = None) -> str:
        """
        Generate natural language explanation for a specific trade.
        
        Args:
            trade_idx: Index of trade to explain
            features: DataFrame with trade features
            baselines: Baseline statistics dictionary
            clusters: Optional cluster labels
            
        Returns:
            Natural language explanation
        """
        if trade_idx >= len(features):
            return "Trade index out of range."
        
        trade = features.iloc[trade_idx]
        explanations = []
        
        # Market context
        if 'rsi_14' in trade:
            rsi = trade['rsi_14']
            if not pd.isna(rsi):
                if rsi < 30:
                    explanations.append(f"Entered during oversold conditions (RSI: {rsi:.1f})")
                elif rsi > 70:
                    explanations.append(f"Entered during overbought conditions (RSI: {rsi:.1f})")
                else:
                    explanations.append(f"Entered during neutral RSI conditions (RSI: {rsi:.1f})")
        
        # Market regime context
        if 'trend_regime' in trade:
            explanations.append(f"Market trend regime: {trade['trend_regime']}")
        if 'volatility_regime' in trade:
            explanations.append(f"Market volatility regime: {trade['volatility_regime']}")
        
        # Position size deviation
        if 'position_size_normalized_by_volatility' in trade and 'position_size_normalized_by_volatility' in baselines:
            baseline = baselines['position_size_normalized_by_volatility']
            size_norm = trade['position_size_normalized_by_volatility']
            if not pd.isna(size_norm):
                zscore = (size_norm - baseline['mean']) / (baseline['std'] + 1e-6)
                
                if abs(zscore) > 2:
                    explanations.append(
                        f"Risk-adjusted position size was {'much larger' if zscore > 0 else 'much smaller'} "
                        f"than typical ({zscore:.2f} standard deviations from baseline)"
                    )
                elif abs(zscore) > 1:
                    explanations.append(
                        f"Risk-adjusted position size was {'larger' if zscore > 0 else 'smaller'} "
                        f"than typical ({zscore:.2f} standard deviations from baseline)"
                    )
        
        # Holding duration
        if 'holding_duration_days' in trade and 'holding_duration_days' in baselines:
            baseline = baselines['holding_duration_days']
            duration = trade['holding_duration_days']
            if not pd.isna(duration):
                if duration > baseline['q75']:
                    explanations.append(f"Held for longer than usual ({duration:.1f} days vs median {baseline['median']:.1f} days)")
                elif duration < baseline['q25']:
                    explanations.append(f"Held for shorter than usual ({duration:.1f} days vs median {baseline['median']:.1f} days)")
        
        # Sequence context
        if 'trades_after_loss' in trade and trade['trades_after_loss'] > 0:
            explanations.append(
                f"This trade occurred after a loss, with {trade['trades_after_loss']} trades "
                f"executed within 48 hours (potential revenge trading pattern)"
            )
        
        # Entry/exit relative to EMA
        if 'entry_price_distance_from_ema20' in trade:
            ema_dist = trade['entry_price_distance_from_ema20']
            if not pd.isna(ema_dist):
                if ema_dist > 0.02:  # 2% above EMA
                    explanations.append(f"Entered {ema_dist*100:.1f}% above EMA(20) - chasing trend")
                elif ema_dist < -0.02:  # 2% below EMA
                    explanations.append(f"Entered {abs(ema_dist)*100:.1f}% below EMA(20) - contrarian entry")
        
        # Cluster assignment
        if clusters is not None and trade_idx < len(clusters):
            cluster_id = clusters[trade_idx]
            explanations.append(f"Belongs to behavioral cluster {cluster_id}")
        
        # Probability-enriched statements (when data is available)
        prob_lines = self._get_probability_enriched_lines(trade, features, baselines)
        explanations.extend(prob_lines)
        
        # Combine explanations
        if explanations:
            explanation = f"Trade on {trade['date']} ({trade.get('symbol', 'N/A')}):\n"
            explanation += "\n".join(f"  • {exp}" for exp in explanations)
        else:
            explanation = f"Trade on {trade['date']}: No significant deviations detected."
        
        return explanation
    
    def _get_probability_enriched_lines(self, trade: pd.Series, features: pd.DataFrame,
                                        baselines: Dict) -> List[str]:
        """Build probability-enriched explanation lines when data is available."""
        lines = []
        if len(features) < 3:
            return lines
        if 'realized_pnl' in features.columns and 'holding_duration_days' in features.columns and 'holding_duration_days' in baselines:
            winners = features[features['realized_pnl'] > 0]
            if len(winners) >= 3:
                median_d = baselines['holding_duration_days'].get('median', features['holding_duration_days'].median())
                hold_longer = (winners['holding_duration_days'] > median_d).mean()
                if not pd.isna(hold_longer) and 0 < hold_longer < 1:
                    trade_pnl = trade.get('realized_pnl', np.nan)
                    trade_dur = trade.get('holding_duration_days', np.nan)
                    if not pd.isna(trade_pnl) and trade_pnl > 0 and not pd.isna(trade_dur) and trade_dur > median_d:
                        lines.append(f"There is a {hold_longer*100:.0f}% probability you hold longer after winning trades.")
        if 'position_size_normalized_by_volatility' in features.columns and 'position_size_normalized_by_volatility' in baselines:
            bl = baselines['position_size_normalized_by_volatility']
            mean_bl = bl.get('mean', features['position_size_normalized_by_volatility'].mean())
            above = (features['position_size_normalized_by_volatility'] > mean_bl).mean()
            if not pd.isna(above) and 0 < above < 1:
                size_norm = trade.get('position_size_normalized_by_volatility', np.nan)
                if not pd.isna(size_norm) and size_norm > mean_bl:
                    lines.append(f"There is a {above*100:.0f}% probability you size above baseline in your typical trades.")
        return lines
    
    def explain_cluster(self, cluster_id: int, cluster_analysis: Dict) -> str:
        """Explain what a behavioral cluster represents."""
        cluster_key = f'cluster_{cluster_id}'
        if cluster_key not in cluster_analysis:
            return f"Cluster {cluster_id} not found."
        
        info = cluster_analysis[cluster_key]
        
        explanation = f"Behavioral Cluster {cluster_id}:\n"
        explanation += f"  • Size: {info['size']} trades\n"
        explanation += f"  • Average trades per day: {info['avg_trades_per_day']:.2f}\n"
        explanation += f"  • Average position size ratio: {info['avg_position_size']:.4f}\n"
        explanation += f"  • Average holding duration: {info['avg_holding_duration']:.1f} days\n"
        explanation += f"  • Average P&L: ${info['avg_pnl']:.2f}\n"
        explanation += f"  • Win rate: {info['win_rate']:.1%}\n"
        
        # Behavioral interpretation
        if info['avg_trades_per_day'] > 3:
            explanation += "\nInterpretation: High-frequency trading mode"
        elif info['avg_position_size'] > 0.1:
            explanation += "\nInterpretation: Aggressive position sizing"
        elif info['avg_holding_duration'] < 2:
            explanation += "\nInterpretation: Short-term trading focus"
        else:
            explanation += "\nInterpretation: Moderate, disciplined trading"
        
        return explanation
    
    def explain_change_point(self, change_point_idx: int, segments: List[Dict], 
                            features: Optional[pd.DataFrame] = None) -> str:
        """
        Explain a detected change point with "What changed?" summary.
        
        Args:
            change_point_idx: Index of the change point
            segments: List of segment dictionaries
            features: Optional DataFrame with features for detailed analysis
            
        Returns:
            Explanation string with top 3 features that changed most
        """
        # Find segment containing this change point
        for i, segment in enumerate(segments):
            if segment['end_idx'] == change_point_idx:
                prev_segment = segments[i-1] if i > 0 else None
                next_segment = segments[i] if i < len(segments) else None
                
                if prev_segment and next_segment:
                    explanation = f"Behavioral Change Point Detected:\n"
                    explanation += f"  • Date: {segment.get('start_date', 'N/A')}\n"
                    explanation += f"  • Previous period avg trades/day: {prev_segment['avg_trades_per_day']:.2f}\n"
                    explanation += f"  • New period avg trades/day: {next_segment['avg_trades_per_day']:.2f}\n"
                    explanation += f"  • Previous period avg P&L: ${prev_segment['avg_pnl']:.2f}\n"
                    explanation += f"  • New period avg P&L: ${next_segment['avg_pnl']:.2f}\n"
                    
                    # "What changed?" Summary - Top 3 features that shifted most
                    if features is not None:
                        what_changed = self._analyze_change_point_features(
                            features, prev_segment, next_segment
                        )
                        if what_changed:
                            explanation += "\n" + "=" * 60 + "\n"
                            explanation += "WHAT CHANGED? - Top 3 Feature Shifts:\n"
                            explanation += "=" * 60 + "\n"
                            for idx, change_info in enumerate(what_changed[:3], 1):
                                explanation += f"\n{idx}. {change_info['feature_name']}:\n"
                                explanation += f"   Before: {change_info['before_value']:.2f}\n"
                                explanation += f"   After:  {change_info['after_value']:.2f}\n"
                                explanation += f"   Change: {change_info['change_description']}\n"
                    
                    # Interpretation
                    if abs(next_segment['avg_trades_per_day'] - prev_segment['avg_trades_per_day']) > 1:
                        explanation += "\n\nInterpretation: Significant change in trading frequency detected."
                    if abs(next_segment['avg_pnl'] - prev_segment['avg_pnl']) > 100:
                        explanation += "\nInterpretation: Significant change in performance detected."
                    
                    return explanation
        
        return f"Change point at index {change_point_idx}: No detailed analysis available."
    
    def _analyze_change_point_features(self, features: pd.DataFrame,
                                       prev_segment: Dict,
                                       next_segment: Dict) -> List[Dict]:
        """
        Analyze which features changed most between segments.
        
        Args:
            features: DataFrame with all features
            prev_segment: Previous segment dictionary
            next_segment: Next segment dictionary
            
        Returns:
            List of change information dictionaries, sorted by magnitude of change
        """
        if 'start_idx' not in prev_segment or 'end_idx' not in prev_segment:
            return []
        if 'start_idx' not in next_segment or 'end_idx' not in next_segment:
            return []
        
        prev_start = prev_segment['start_idx']
        prev_end = prev_segment['end_idx']
        next_start = next_segment['start_idx']
        next_end = next_segment['end_idx']
        
        # Get data for each segment
        prev_data = features.iloc[prev_start:prev_end]
        next_data = features.iloc[next_start:next_end]
        
        if len(prev_data) == 0 or len(next_data) == 0:
            return []
        
        # Key behavioral features to analyze
        key_features = [
            ('trades_per_day', 'Trade Frequency'),
            ('position_size_normalized_by_volatility', 'Position Size (Volatility-Adjusted)'),
            ('holding_duration_days', 'Holding Duration'),
            ('entry_price_distance_from_ema20', 'Entry Price Distance from EMA20'),
            ('time_gap_hours_since_last_trade', 'Time Gap Between Trades'),
            ('trades_after_loss', 'Trades After Loss'),
            ('position_size_dollar_value', 'Position Size (Dollar Value)')
        ]
        
        changes = []
        
        for feature_col, feature_name in key_features:
            if feature_col not in features.columns:
                continue
            
            # Calculate means for each segment
            prev_mean = prev_data[feature_col].replace([np.inf, -np.inf], np.nan).mean()
            next_mean = next_data[feature_col].replace([np.inf, -np.inf], np.nan).mean()
            
            if pd.isna(prev_mean) or pd.isna(next_mean):
                continue
            
            # Calculate change
            if abs(prev_mean) > 1e-6:
                pct_change = ((next_mean - prev_mean) / abs(prev_mean)) * 100
                abs_change = next_mean - prev_mean
            else:
                # Handle case where prev_mean is near zero
                pct_change = 100 if abs(next_mean) > 1e-6 else 0
                abs_change = next_mean - prev_mean
            
            # Only include significant changes (>10% or meaningful absolute change)
            if abs(pct_change) > 10 or abs(abs_change) > 0.1:
                # Create human-readable change description
                if pct_change > 0:
                    change_desc = f"Increased by {abs(pct_change):.1f}% ({abs_change:+.2f})"
                else:
                    change_desc = f"Decreased by {abs(pct_change):.1f}% ({abs_change:+.2f})"
                
                changes.append({
                    'feature_name': feature_name,
                    'feature_col': feature_col,
                    'before_value': prev_mean,
                    'after_value': next_mean,
                    'change_pct': pct_change,
                    'change_abs': abs_change,
                    'change_description': change_desc,
                    'change_magnitude': abs(pct_change)  # For sorting
                })
        
        # Sort by magnitude of change (descending)
        changes.sort(key=lambda x: x['change_magnitude'], reverse=True)
        
        return changes
    
    def generate_feature_attribution(self, trade_idx: int, features: pd.DataFrame, 
                                    cluster_centers: np.ndarray, cluster_id: int) -> str:
        """Explain why a trade belongs to a specific cluster."""
        if trade_idx >= len(features):
            return "Trade index out of range."
        
        trade = features.iloc[trade_idx]
        feature_cols = [col for col in features.columns 
                       if col not in ['date', 'symbol', 'side', 'price', 'quantity'] 
                       and features[col].dtype in [np.float64, np.int64]]
        
        # Calculate distances to cluster centers
        trade_features = trade[feature_cols].fillna(0).values
        center = cluster_centers[cluster_id]
        
        # Feature contributions (difference from cluster center)
        contributions = {}
        for i, col in enumerate(feature_cols):
            if i < len(trade_features) and i < len(center):
                contributions[col] = abs(trade_features[i] - center[i])
        
        # Sort by contribution
        top_contributors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:5]
        
        explanation = f"Trade belongs to Cluster {cluster_id} primarily due to:\n"
        for feature, contribution in top_contributors:
            trade_val = trade[feature] if feature in trade else 0
            center_val = center[feature_cols.index(feature)] if feature_cols.index(feature) < len(center) else 0
            explanation += f"  • {feature}: {trade_val:.2f} (cluster center: {center_val:.2f})\n"
        
        return explanation
    
    def generate_counterfactual(self, trade_idx: int, features: pd.DataFrame, 
                               baselines: Dict) -> str:
        """Generate counterfactual explanation."""
        if trade_idx >= len(features):
            return "Trade index out of range."
        
        trade = features.iloc[trade_idx]
        counterfactuals = []
        
        # Position size counterfactual
        if 'position_size_ratio' in trade and 'position_size_ratio' in baselines:
            baseline = baselines['position_size_ratio']
            current_size = trade['position_size_ratio']
            baseline_size = baseline['mean']
            
            if abs(current_size - baseline_size) > baseline['std']:
                counterfactuals.append(
                    f"If position size were within normal range ({baseline_size:.4f} instead of {current_size:.4f}), "
                    f"this trade would align with baseline behavior."
                )
        
        # Timing counterfactual
        if 'holding_duration' in trade and 'holding_duration' in baselines:
            baseline = baselines['holding_duration']
            current_duration = trade['holding_duration']
            if not pd.isna(current_duration):
                baseline_duration = baseline['median']
                if abs(current_duration - baseline_duration) > baseline['std']:
                    counterfactuals.append(
                        f"If holding duration were typical ({baseline_duration:.1f} days instead of {current_duration:.1f} days), "
                        f"this trade would match baseline patterns."
                    )
        
        if counterfactuals:
            explanation = "Counterfactual Analysis:\n"
            explanation += "\n".join(f"  • {cf}" for cf in counterfactuals)
        else:
            explanation = "This trade aligns with baseline behavior patterns."
        explanation += "\n\nNote: Full counterfactual analysis is available via CounterfactualEngine."
        return explanation
    
    def _generate_probabilistic_section(self, features: pd.DataFrame, baselines: Dict) -> List[str]:
        """Compute bootstrap 95% credible intervals for key metrics; return list of report lines."""
        lines = []
        n_resamples = 1000
        rng = np.random.default_rng(42)
        n = len(features)
        if n < 10:
            return ["Insufficient data for probabilistic intervals (need at least 10 trades)."]
        for col, label in [
            ('holding_duration_days', 'Holding duration (days)'),
            ('position_size_dollar_value', 'Position size (dollar value)'),
            ('trades_per_day', 'Trades per day'),
        ]:
            if col not in features.columns:
                continue
            series = features[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(series) < 5:
                continue
            means = []
            for _ in range(n_resamples):
                idx = rng.integers(0, len(series), size=len(series))
                means.append(series.iloc[idx].mean())
            lo = np.percentile(means, 2.5)
            hi = np.percentile(means, 97.5)
            lines.append(f"  {label}: 95%% CI [{lo:.2f}, {hi:.2f}] (point estimate: {series.mean():.2f})")
        if not lines:
            return ["No key metrics available for bootstrap intervals."]
        return lines
    
    def generate_report(self, features: pd.DataFrame, baselines: Dict, 
                       pattern_results: Dict) -> str:
        """Generate comprehensive behavioral report."""
        report = []
        report.append("=" * 80)
        report.append("BEHAVIORAL TRADING ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 80)
        report.append(f"Total trades analyzed: {len(features)}")
        report.append(f"Date range: {features['date'].min()} to {features['date'].max()}")
        if 'realized_pnl' in features.columns:
            report.append(f"Total P&L: ${features['realized_pnl'].sum():.2f}")
            report.append(f"Average P&L per trade: ${features['realized_pnl'].mean():.2f}")
            report.append(f"Win rate: {(features['realized_pnl'] > 0).mean():.1%}")
        report.append("")
        
        # Baseline summary
        report.append("BEHAVIORAL BASELINES")
        report.append("-" * 80)
        for feature, stats in baselines.items():
            if isinstance(stats, dict) and 'mean' in stats:
                report.append(f"{feature}:")
                report.append(f"  Mean: {stats['mean']:.4f}")
                report.append(f"  Median: {stats['median']:.4f}")
                report.append(f"  Std: {stats['std']:.4f}")
        report.append("")
        
        # Cluster analysis
        if 'clusters' in pattern_results and 'analysis' in pattern_results['clusters']:
            report.append("BEHAVIORAL CLUSTERS")
            report.append("-" * 80)
            for cluster_id, info in pattern_results['clusters']['analysis'].items():
                report.append(self.explain_cluster(int(cluster_id.split('_')[1]), 
                                                  pattern_results['clusters']['analysis']))
                report.append("")
        
        # Change points
        if 'change_points' in pattern_results and pattern_results['change_points']['indices']:
            report.append("BEHAVIORAL CHANGE POINTS")
            report.append("-" * 80)
            for cp_idx in pattern_results['change_points']['indices']:
                report.append(self.explain_change_point(cp_idx, 
                                                       pattern_results['change_points']['segments'],
                                                       features))
                report.append("")
        
        # Anomalies
        if 'anomalies' in pattern_results and pattern_results['anomalies']['indices']:
            report.append("ANOMALOUS TRADES")
            report.append("-" * 80)
            report.append(f"Found {len(pattern_results['anomalies']['indices'])} anomalous trades")
            report.append("These trades deviate significantly from baseline behavior patterns.")
            report.append("")
        
        # Behavioral Biases (NEW)
        report.append("BEHAVIORAL BIAS MAPPING")
        report.append("-" * 80)
        bias_mappings = self.map_behavioral_biases(features, pattern_results, baselines)
        if bias_mappings['total_biases_detected'] > 0:
            for bias_info in bias_mappings['biases']:
                report.append(f"\n{bias_info['bias']}:")
                report.append(f"  Pattern: {bias_info['pattern']}")
                report.append(f"  Probability: {bias_info['probability']*100:.0f}% ({bias_info['strength']} signal)")
                report.append(f"  Explanation: {bias_info['explanation']}")
                report.append("")
        else:
            report.append("No significant behavioral biases detected in the analyzed patterns.")
            report.append("")
        report.append(f"Note: {bias_mappings['note']}")
        report.append("")
        
        # Probabilistic insights (bootstrap credible intervals)
        report.append("PROBABILISTIC INSIGHTS")
        report.append("-" * 80)
        for line in self._generate_probabilistic_section(features, baselines):
            report.append(line)
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def generate_xai_summary(self, features: pd.DataFrame, baselines: Dict, 
                            pattern_results: Dict, stability_results: Optional[Dict] = None) -> str:
        """
        Generate an explainable AI summary report using rule-based NLG.
        
        This uses:
        - Rule-based Natural Language Generation (NLG)
        - Feature contribution ranking to explain trade differences
        - Quantifiable statistics
        - Simple English explanations
        
        Args:
            features: DataFrame with trade features
            baselines: Baseline statistics dictionary
            pattern_results: Pattern discovery results
            
        Returns:
            Natural language summary paragraph (~100 words)
        """
        summary_parts = []
        
        # 1. Overall Performance Summary
        total_trades = len(features)
        if 'realized_pnl' in features.columns:
            total_pnl = features['realized_pnl'].sum()
            avg_pnl = features['realized_pnl'].mean()
            win_rate = (features['realized_pnl'] > 0).mean() * 100
            winning_trades = (features['realized_pnl'] > 0).sum()
            losing_trades = (features['realized_pnl'] < 0).sum()
            
            # Performance assessment
            if total_pnl < 0:
                perf_desc = f"lost ${abs(total_pnl):,.0f} overall"
            else:
                perf_desc = f"gained ${total_pnl:,.0f} overall"
            
            summary_parts.append(
                f"Analyzed {total_trades} trades: {perf_desc} with {win_rate:.0f}% win rate "
                f"({winning_trades} wins, {losing_trades} losses), averaging ${avg_pnl:.0f} per trade."
            )
        
        # 2. Behavioral Pattern Summary (from clusters)
        if 'clusters' in pattern_results and 'labels' in pattern_results['clusters']:
            clusters = pattern_results['clusters']['labels']
            unique_clusters = np.unique(clusters)
            
            # Find best and worst performing clusters
            cluster_performance = {}
            for cluster_id in unique_clusters:
                cluster_mask = clusters == cluster_id
                if 'realized_pnl' in features.columns:
                    cluster_pnl = features.loc[cluster_mask, 'realized_pnl'].mean()
                    cluster_size = cluster_mask.sum()
                    cluster_performance[cluster_id] = {
                        'pnl': cluster_pnl,
                        'size': cluster_size,
                        'win_rate': (features.loc[cluster_mask, 'realized_pnl'] > 0).mean() * 100
                    }
            
            if cluster_performance:
                best_cluster = max(cluster_performance.items(), key=lambda x: x[1]['pnl'])
                worst_cluster = min(cluster_performance.items(), key=lambda x: x[1]['pnl'])
                
                best_id, best_stats = best_cluster
                worst_id, worst_stats = worst_cluster
                
                summary_parts.append(
                    f"Identified {len(unique_clusters)} behavioral patterns: Cluster {best_id} performed best "
                    f"(${best_stats['pnl']:.0f} avg, {best_stats['win_rate']:.0f}% win rate, {best_stats['size']} trades) "
                    f"while Cluster {worst_id} underperformed (${worst_stats['pnl']:.0f} avg, {worst_stats['win_rate']:.0f}% win rate)."
                )
        
        # 3. Key Behavioral Deviations (Feature Contribution Ranking)
        key_features = [
            'position_size_normalized_by_volatility',
            'holding_duration_days',
            'trades_per_day',
            'entry_price_distance_from_ema20',
            'trades_after_loss'
        ]
        
        deviations = []
        for feature in key_features:
            if feature in features.columns and feature in baselines:
                baseline = baselines[feature]
                if 'mean' in baseline and 'std' in baseline:
                    # Calculate average absolute deviation
                    avg_deviation = abs(features[feature] - baseline['mean']).mean()
                    relative_dev = avg_deviation / (baseline['std'] + 1e-6)
                    
                    if relative_dev > 1.0:  # Significant deviation
                        feature_name = feature.replace('_', ' ').title()
                        deviations.append((feature_name, relative_dev, feature))
        
        # Sort by deviation magnitude
        deviations.sort(key=lambda x: x[1], reverse=True)
        
        if deviations:
            top_deviation = deviations[0]
            feature_col = top_deviation[2]
            bl = baselines[feature_col]
            rel_devs = (abs(features[feature_col] - bl['mean']) / (bl['std'] + 1e-6)).replace([np.inf, -np.inf], np.nan).dropna()
            ci_text = ""
            if len(rel_devs) >= 10:
                rng = np.random.default_rng(42)
                arr = rel_devs.values
                n = len(arr)
                boot_means = [arr[rng.integers(0, n, size=n)].mean() for _ in range(1000)]
                lo, hi = np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)
                ci_text = f" with 95%% confidence interval [{lo:.2f}, {hi:.2f}]"
            summary_parts.append(
                f"Primary behavioral deviation: {top_deviation[0]} was {top_deviation[1]:.1f}x "
                f"more variable than normal{ci_text}, indicating inconsistent trading discipline."
            )
        
        # 4. Market Signal Following
        if 'rsi_14' in features.columns:
            overbought = (features['rsi_14'] > 70).sum()
            oversold = (features['rsi_14'] < 30).sum()
            
            if 'side' in features.columns:
                overbought_sells = ((features['rsi_14'] > 70) & (features['side'] == 'sell')).sum()
                oversold_buys = ((features['rsi_14'] < 30) & (features['side'] == 'buy')).sum()
                
                rsi_follow_rate = ((overbought_sells + oversold_buys) / (overbought + oversold) * 100) if (overbought + oversold) > 0 else 0
                
                summary_parts.append(
                    f"Signal adherence: Followed {rsi_follow_rate:.0f}% of RSI signals "
                    f"({overbought_sells + oversold_buys}/{overbought + oversold} signals acted upon)."
                )
        
        # 5. Behavioral Stability Score (NEW)
        if stability_results and stability_results.get('stability_score') is not None:
            stability_score = stability_results['stability_score']
            if stability_score >= 70:
                stability_desc = "highly consistent"
            elif stability_score >= 50:
                stability_desc = "moderately consistent"
            elif stability_score >= 30:
                stability_desc = "somewhat variable"
            else:
                stability_desc = "highly variable"
            
            summary_parts.append(
                f"Behavioral stability score: {stability_score:.0f}/100 ({stability_desc}), "
                f"indicating {stability_desc.replace('highly ', '').replace('moderately ', '').replace('somewhat ', '')} "
                f"trading patterns over time."
            )
        
        # 6. Change Points and Anomalies
        if 'change_points' in pattern_results and pattern_results['change_points']['indices']:
            num_change_points = len(pattern_results['change_points']['indices'])
            summary_parts.append(
                f"Detected {num_change_points} behavioral shift{'s' if num_change_points > 1 else ''}, "
                f"suggesting trading strategy evolved or emotional responses changed."
            )
        
        if 'anomalies' in pattern_results and pattern_results['anomalies']['indices']:
            num_anomalies = len(pattern_results['anomalies']['indices'])
            summary_parts.append(
                f"{num_anomalies} anomalous trade{'s' if num_anomalies > 1 else ''} deviated significantly "
                f"from baseline patterns, likely driven by emotional decisions or external factors."
            )
        
        # 6. Risk Management Assessment
        if 'position_size_normalized_by_volatility' in features.columns and 'volatility_rolling_std' in features.columns:
            high_vol_trades = (features['volatility_rolling_std'] > features['volatility_rolling_std'].quantile(0.75)).sum()
            if high_vol_trades > 0:
                avg_size_high_vol = features.loc[features['volatility_rolling_std'] > features['volatility_rolling_std'].quantile(0.75), 
                                                  'position_size_normalized_by_volatility'].mean()
                avg_size_normal = features.loc[features['volatility_rolling_std'] <= features['volatility_rolling_std'].quantile(0.75),
                                               'position_size_normalized_by_volatility'].mean()
                
                if avg_size_high_vol > avg_size_normal * 1.2:
                    summary_parts.append(
                        f"Risk concern: Position sizes were {avg_size_high_vol/avg_size_normal:.1f}x larger "
                        f"during high volatility periods, increasing exposure when markets were most unstable."
                    )
        
        # Combine all parts into a coherent paragraph
        full_summary = " ".join(summary_parts)
        
        # Ensure it's approximately 100 words (target: 90-110 words)
        words = full_summary.split()
        word_count = len(words)
        
        if word_count > 110:
            # Trim to ~100 words, keeping most important parts
            full_summary = " ".join(words[:100]) + "..."
        elif word_count < 90:
            # Add more details to reach ~100 words
            additional_info = []
            
            # Add feature contribution ranking details
            if deviations and len(deviations) > 1:
                second_dev = deviations[1] if len(deviations) > 1 else None
                if second_dev:
                    additional_info.append(
                        f"Secondary deviation: {second_dev[0]} showed {second_dev[1]:.1f}x normal variation."
                    )
            
            # Add holding duration insight if available
            if 'holding_duration_days' in features.columns and 'holding_duration_days' in baselines:
                baseline_duration = baselines['holding_duration_days']['median']
                avg_duration = features['holding_duration_days'].mean()
                if abs(avg_duration - baseline_duration) > baseline_duration * 0.3:
                    additional_info.append(
                        f"Average holding period was {avg_duration:.1f} days vs typical {baseline_duration:.1f} days."
                    )
            
            # Add trade frequency insight
            if 'trades_per_day' in features.columns:
                avg_freq = features['trades_per_day'].mean()
                additional_info.append(f"Trading frequency averaged {avg_freq:.1f} trades per day.")
            
            # Add concluding recommendation
            if 'realized_pnl' in features.columns and features['realized_pnl'].sum() < 0:
                additional_info.append("Recommendation: Improve signal following and maintain consistent risk management.")
            else:
                additional_info.append("Continue monitoring patterns to sustain performance.")
            
            # Add enough to reach ~100 words
            while len(full_summary.split()) < 95 and additional_info:
                full_summary += " " + additional_info.pop(0)
        
        # Final word count check and trim if needed
        final_words = full_summary.split()
        if len(final_words) > 110:
            full_summary = " ".join(final_words[:105])
        
        return full_summary
    
    def map_behavioral_biases(self, features: pd.DataFrame, 
                             pattern_results: Dict,
                             baselines: Optional[Dict] = None) -> Dict:
        """
        Explicit Bias Mapping Layer - maps detected patterns to behavioral biases.
        
        This is a rule-based, probabilistic mapping system that identifies potential
        behavioral biases without making definitive diagnoses. Uses probabilistic language
        and explains patterns in human terms.
        
        Mapped Biases:
        - Revenge Trading: Increased trade frequency after loss
        - Overconfidence: Larger position sizes during high volatility
        - FOMO (Fear of Missing Out): Buying far above EMA repeatedly
        - Loss Aversion: Holding losers longer than winners
        
        Args:
            features: DataFrame with trade features
            pattern_results: Pattern discovery results
            baselines: Optional baseline statistics
            
        Returns:
            Dictionary with bias mappings and probabilities
        """
        bias_mappings = []
        
        # 1. Revenge Trading Detection
        revenge_trading = self._detect_revenge_trading(features, baselines)
        if revenge_trading['detected']:
            bias_mappings.append(revenge_trading)
        
        # 2. Overconfidence Detection
        overconfidence = self._detect_overconfidence(features, baselines)
        if overconfidence['detected']:
            bias_mappings.append(overconfidence)
        
        # 3. FOMO Detection
        fomo = self._detect_fomo(features, baselines)
        if fomo['detected']:
            bias_mappings.append(fomo)
        
        # 4. Loss Aversion Detection
        loss_aversion = self._detect_loss_aversion(features, baselines)
        if loss_aversion['detected']:
            bias_mappings.append(loss_aversion)
        
        return {
            'biases': bias_mappings,
            'total_biases_detected': len(bias_mappings),
            'note': "These are probabilistic mappings based on patterns. They are not diagnoses."
        }
    
    def _detect_revenge_trading(self, features: pd.DataFrame, 
                                baselines: Optional[Dict]) -> Dict:
        """
        Detect revenge trading: Increased trade frequency after loss.
        
        Pattern: Increased trade frequency after a loss (within 48 hours)
        """
        if 'trades_after_loss' not in features.columns or 'is_loss' not in features.columns:
            return {'detected': False, 'bias': 'Revenge Trading'}
        
        # Count trades after losses
        loss_trades = features[features['is_loss'] == 1]
        if len(loss_trades) == 0:
            return {'detected': False, 'bias': 'Revenge Trading'}
        
        # Calculate average trades after loss
        avg_trades_after_loss = features['trades_after_loss'].mean()
        
        # Compare to baseline trade frequency
        baseline_freq = features['trades_per_day'].mean() if 'trades_per_day' in features.columns else 1.0
        
        # Threshold: if trades after loss > 1.5x baseline, likely revenge trading
        threshold = baseline_freq * 1.5
        
        if avg_trades_after_loss > threshold:
            # Calculate probability/strength
            ratio = avg_trades_after_loss / (baseline_freq + 1e-6)
            probability = min(0.95, 0.5 + (ratio - 1.5) * 0.15)  # Scale to 0.5-0.95
            
            return {
                'detected': True,
                'bias': 'Revenge Trading',
                'pattern': f"Increased trade frequency after loss (avg {avg_trades_after_loss:.1f} trades vs baseline {baseline_freq:.1f})",
                'probability': probability,
                'strength': 'strong' if probability > 0.75 else 'moderate' if probability > 0.6 else 'weak',
                'explanation': f"There is a {probability*100:.0f}% probability that you may be engaging in revenge trading. "
                              f"After losses, your trade frequency increases by {ratio:.1f}x, suggesting emotional reactions "
                              f"to losses rather than systematic decision-making."
            }
        
        return {'detected': False, 'bias': 'Revenge Trading'}
    
    def _detect_overconfidence(self, features: pd.DataFrame, 
                              baselines: Optional[Dict]) -> Dict:
        """
        Detect overconfidence: Larger position sizes during high volatility.
        
        Pattern: Position sizes are larger during high volatility periods
        """
        if 'position_size_normalized_by_volatility' not in features.columns:
            return {'detected': False, 'bias': 'Overconfidence'}
        
        if 'volatility_rolling_std' not in features.columns:
            return {'detected': False, 'bias': 'Overconfidence'}
        
        # Split into high and low volatility periods
        vol_median = features['volatility_rolling_std'].median()
        high_vol_mask = features['volatility_rolling_std'] > vol_median
        low_vol_mask = features['volatility_rolling_std'] <= vol_median
        
        if high_vol_mask.sum() == 0 or low_vol_mask.sum() == 0:
            return {'detected': False, 'bias': 'Overconfidence'}
        
        # Compare position sizes
        high_vol_sizes = features.loc[high_vol_mask, 'position_size_normalized_by_volatility']
        low_vol_sizes = features.loc[low_vol_mask, 'position_size_normalized_by_volatility']
        
        avg_high_vol_size = high_vol_sizes.mean()
        avg_low_vol_size = low_vol_sizes.mean()
        
        # Threshold: if high vol size > 1.2x low vol size, likely overconfidence
        if avg_high_vol_size > avg_low_vol_size * 1.2:
            ratio = avg_high_vol_size / (avg_low_vol_size + 1e-6)
            probability = min(0.95, 0.5 + (ratio - 1.2) * 0.2)  # Scale to 0.5-0.95
            
            return {
                'detected': True,
                'bias': 'Overconfidence',
                'pattern': f"Larger position sizes during high volatility ({avg_high_vol_size:.2f} vs {avg_low_vol_size:.2f} in low vol)",
                'probability': probability,
                'strength': 'strong' if probability > 0.75 else 'moderate' if probability > 0.6 else 'weak',
                'explanation': f"There is a {probability*100:.0f}% probability that overconfidence may be influencing your trading. "
                              f"During high volatility periods, your position sizes are {ratio:.1f}x larger, suggesting "
                              f"you may be overestimating your ability to handle market uncertainty."
            }
        
        return {'detected': False, 'bias': 'Overconfidence'}
    
    def _detect_fomo(self, features: pd.DataFrame, 
                    baselines: Optional[Dict]) -> Dict:
        """
        Detect FOMO (Fear of Missing Out): Buying far above EMA repeatedly.
        
        Pattern: Repeatedly buying at prices significantly above EMA (chasing trends)
        """
        if 'entry_price_distance_from_ema20' not in features.columns:
            return {'detected': False, 'bias': 'FOMO'}
        
        if 'side' not in features.columns:
            return {'detected': False, 'bias': 'FOMO'}
        
        # Focus on buy trades
        buy_trades = features[features['side'] == 'buy']
        if len(buy_trades) == 0:
            return {'detected': False, 'bias': 'FOMO'}
        
        # Count trades where entry is significantly above EMA (e.g., > 2%)
        fomo_threshold = 0.02  # 2% above EMA
        fomo_trades = buy_trades[buy_trades['entry_price_distance_from_ema20'] > fomo_threshold]
        
        fomo_ratio = len(fomo_trades) / len(buy_trades)
        
        # Threshold: if > 30% of buys are far above EMA, likely FOMO
        if fomo_ratio > 0.3:
            avg_distance = fomo_trades['entry_price_distance_from_ema20'].mean()
            probability = min(0.95, 0.5 + (fomo_ratio - 0.3) * 1.5)  # Scale to 0.5-0.95
            
            return {
                'detected': True,
                'bias': 'FOMO (Fear of Missing Out)',
                'pattern': f"Buying far above EMA repeatedly ({len(fomo_trades)}/{len(buy_trades)} buys, avg {avg_distance*100:.1f}% above EMA)",
                'probability': probability,
                'strength': 'strong' if probability > 0.75 else 'moderate' if probability > 0.6 else 'weak',
                'explanation': f"There is a {probability*100:.0f}% probability that FOMO may be influencing your trading. "
                              f"{fomo_ratio*100:.0f}% of your buy trades occur when prices are significantly above the EMA, "
                              f"suggesting you may be chasing trends rather than waiting for better entry points."
            }
        
        return {'detected': False, 'bias': 'FOMO'}
    
    def _detect_loss_aversion(self, features: pd.DataFrame, 
                             baselines: Optional[Dict]) -> Dict:
        """
        Detect loss aversion: Holding losers longer than winners.
        
        Pattern: Average holding duration for losing trades > winning trades
        """
        if 'holding_duration_days' not in features.columns:
            return {'detected': False, 'bias': 'Loss Aversion'}
        
        if 'realized_pnl' not in features.columns:
            return {'detected': False, 'bias': 'Loss Aversion'}
        
        # Split into winners and losers
        winners = features[features['realized_pnl'] > 0]
        losers = features[features['realized_pnl'] < 0]
        
        if len(winners) == 0 or len(losers) == 0:
            return {'detected': False, 'bias': 'Loss Aversion'}
        
        # Compare holding durations
        avg_winner_duration = winners['holding_duration_days'].mean()
        avg_loser_duration = losers['holding_duration_days'].mean()
        
        # Threshold: if losers held > 1.3x longer than winners, likely loss aversion
        if avg_loser_duration > avg_winner_duration * 1.3:
            ratio = avg_loser_duration / (avg_winner_duration + 1e-6)
            probability = min(0.95, 0.5 + (ratio - 1.3) * 0.3)  # Scale to 0.5-0.95
            
            return {
                'detected': True,
                'bias': 'Loss Aversion',
                'pattern': f"Holding losers longer than winners (losers: {avg_loser_duration:.1f} days vs winners: {avg_winner_duration:.1f} days)",
                'probability': probability,
                'strength': 'strong' if probability > 0.75 else 'moderate' if probability > 0.6 else 'weak',
                'explanation': f"There is a {probability*100:.0f}% probability that loss aversion may be influencing your trading. "
                              f"You hold losing positions {ratio:.1f}x longer than winning positions, suggesting "
                              f"you may be reluctant to realize losses and hoping positions will recover."
            }
        
        return {'detected': False, 'bias': 'Loss Aversion'}
    
    def analyze_stock_performance(self, features: pd.DataFrame, baselines: Optional[Dict] = None) -> str:
        """
        Analyze best and worst performing stocks with detailed explanations.
        
        For each stock, analyzes:
        - Total P&L and win rate
        - Entry/exit timing relative to volatility
        - RSI conditions at entry/exit
        - EMA positioning
        - Market regime conditions
        - Holding duration patterns
        - Position sizing
        
        Args:
            features: DataFrame with trade features
            baselines: Optional baseline statistics
            
        Returns:
            Detailed text report explaining best and worst stock performance
        """
        if 'symbol' not in features.columns or 'realized_pnl' not in features.columns:
            return "Cannot analyze stock performance: Missing 'symbol' or 'realized_pnl' columns."
        
        report = []
        report.append("=" * 80)
        report.append("STOCK PERFORMANCE ANALYSIS - BEST & WORST PERFORMERS")
        report.append("=" * 80)
        report.append("")
        report.append("This analysis explains why certain stocks performed well or poorly,")
        report.append("examining timing, volatility, market indicators, and trading behavior.")
        report.append("")
        
        # Group trades by symbol and calculate performance metrics
        symbol_performance = {}
        
        for symbol in features['symbol'].dropna().unique():
            symbol_trades = features[features['symbol'] == symbol].copy()
            
            if len(symbol_trades) == 0:
                continue
            
            # Calculate performance metrics
            total_pnl = symbol_trades['realized_pnl'].sum()
            avg_pnl = symbol_trades['realized_pnl'].mean()
            win_rate = (symbol_trades['realized_pnl'] > 0).mean() * 100
            num_trades = len(symbol_trades)
            num_wins = (symbol_trades['realized_pnl'] > 0).sum()
            num_losses = (symbol_trades['realized_pnl'] < 0).sum()
            
            # Separate buy and sell trades
            if 'side' in symbol_trades.columns:
                side_col = symbol_trades['side'].astype(str).str.lower().str.strip()
                buy_trades = symbol_trades[(side_col == 'buy') & (side_col != 'nan')]
                sell_trades = symbol_trades[(side_col == 'sell') & (side_col != 'nan')]
            else:
                buy_trades = pd.DataFrame()
                sell_trades = pd.DataFrame()
            
            # Analyze entry conditions (buy trades)
            entry_analysis = {}
            if len(buy_trades) > 0:
                if 'rsi_14' in buy_trades.columns:
                    rsi_data = buy_trades['rsi_14'].dropna()
                    if len(rsi_data) > 0:
                        entry_analysis['avg_rsi'] = rsi_data.mean()
                        entry_analysis['oversold_entries'] = (rsi_data < 30).sum()
                        entry_analysis['overbought_entries'] = (rsi_data > 70).sum()
                
                if 'volatility_rolling_std' in buy_trades.columns:
                    vol_data = buy_trades['volatility_rolling_std'].dropna()
                    if len(vol_data) > 0:
                        entry_analysis['avg_volatility'] = vol_data.mean()
                        vol_median = vol_data.median()
                        entry_analysis['high_vol_entries'] = (vol_data > vol_median).sum()
                
                if 'entry_price_distance_from_ema20' in buy_trades.columns:
                    ema_data = buy_trades['entry_price_distance_from_ema20'].dropna()
                    if len(ema_data) > 0:
                        entry_analysis['avg_ema20_distance'] = ema_data.mean()
                        entry_analysis['above_ema_entries'] = (ema_data > 0).sum()
                
                if 'trend_regime' in buy_trades.columns:
                    trend_data = buy_trades['trend_regime'].dropna()
                    if len(trend_data) > 0:
                        entry_analysis['trend_regimes'] = trend_data.value_counts().to_dict()
                
                if 'volatility_regime' in buy_trades.columns:
                    vol_regime_data = buy_trades['volatility_regime'].dropna()
                    if len(vol_regime_data) > 0:
                        entry_analysis['vol_regimes'] = vol_regime_data.value_counts().to_dict()
            
            # Analyze exit conditions (sell trades)
            exit_analysis = {}
            if len(sell_trades) > 0:
                if 'rsi_14' in sell_trades.columns:
                    rsi_data = sell_trades['rsi_14'].dropna()
                    if len(rsi_data) > 0:
                        exit_analysis['avg_rsi'] = rsi_data.mean()
                
                if 'volatility_rolling_std' in sell_trades.columns:
                    vol_data = sell_trades['volatility_rolling_std'].dropna()
                    if len(vol_data) > 0:
                        exit_analysis['avg_volatility'] = vol_data.mean()
                
                if 'holding_duration_days' in sell_trades.columns:
                    duration_data = sell_trades['holding_duration_days'].dropna()
                    if len(duration_data) > 0:
                        exit_analysis['avg_holding_duration'] = duration_data.mean()
                        exit_analysis['median_holding_duration'] = duration_data.median()
            
            # Calculate average holding duration
            if 'holding_duration_days' in symbol_trades.columns:
                duration_data = symbol_trades['holding_duration_days'].dropna()
                avg_holding = duration_data.mean() if len(duration_data) > 0 else 0
            else:
                avg_holding = 0
            
            symbol_performance[symbol] = {
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'win_rate': win_rate,
                'num_trades': num_trades,
                'num_wins': num_wins,
                'num_losses': num_losses,
                'avg_holding_duration': avg_holding,
                'entry_analysis': entry_analysis,
                'exit_analysis': exit_analysis,
                'trades': symbol_trades
            }
        
        if len(symbol_performance) == 0:
            return "No stock performance data available for analysis."
        
        # Find best and worst performers
        best_stock = max(symbol_performance.items(), key=lambda x: x[1]['total_pnl'])
        worst_stock = min(symbol_performance.items(), key=lambda x: x[1]['total_pnl'])
        
        best_symbol, best_data = best_stock
        worst_symbol, worst_data = worst_stock
        
        # Analyze best performing stock
        report.append("-" * 80)
        report.append(f"BEST PERFORMING STOCK: {best_symbol}")
        report.append("-" * 80)
        report.append("")
        report.append(f"Total P&L: ${best_data['total_pnl']:,.2f}")
        report.append(f"Average P&L per trade: ${best_data['avg_pnl']:,.2f}")
        report.append(f"Win Rate: {best_data['win_rate']:.1f}% ({best_data['num_wins']} wins, {best_data['num_losses']} losses)")
        report.append(f"Number of trades: {best_data['num_trades']}")
        report.append(f"Average holding duration: {best_data['avg_holding_duration']:.1f} days")
        report.append("")
        report.append("WHY THIS STOCK PERFORMED WELL:")
        report.append("")
        
        # Entry timing analysis
        entry_analysis = best_data['entry_analysis']
        if entry_analysis:
            if 'avg_rsi' in entry_analysis:
                rsi = entry_analysis['avg_rsi']
                if rsi < 35:
                    report.append(f"✓ Excellent Entry Timing: Average RSI at entry was {rsi:.1f} (oversold conditions).")
                    report.append("  This suggests you bought when the stock was undervalued, catching good entry points.")
                elif rsi > 65:
                    report.append(f"⚠ Suboptimal Entry Timing: Average RSI at entry was {rsi:.1f} (overbought conditions).")
                    report.append("  Despite this, the stock still performed well, suggesting strong momentum or good exit timing.")
                else:
                    report.append(f"✓ Good Entry Timing: Average RSI at entry was {rsi:.1f} (neutral conditions).")
                    report.append("  You entered at reasonable price levels, avoiding extreme overvaluation.")
                
                if 'oversold_entries' in entry_analysis and entry_analysis['oversold_entries'] > 0:
                    best_buy_trades = best_data['trades']
                    if 'side' in best_buy_trades.columns:
                        side_col = best_buy_trades['side'].astype(str).str.lower().str.strip()
                        buy_count = len(best_buy_trades[(side_col == 'buy') & (side_col != 'nan')])
                        report.append(f"  - {entry_analysis['oversold_entries']} out of {buy_count} buy trades occurred during oversold conditions (RSI < 30)")
            
            if 'avg_volatility' in entry_analysis:
                if 'high_vol_entries' in entry_analysis:
                    best_buy_trades = best_data['trades']
                    if 'side' in best_buy_trades.columns:
                        side_col = best_buy_trades['side'].astype(str).str.lower().str.strip()
                        buy_count = len(best_buy_trades[(side_col == 'buy') & (side_col != 'nan')])
                        if buy_count > 0:
                            high_vol_pct = (entry_analysis['high_vol_entries'] / buy_count * 100)
                            if high_vol_pct > 60:
                                report.append(f"✓ Volatility Advantage: {high_vol_pct:.0f}% of entries occurred during high volatility periods.")
                                report.append("  High volatility can create better entry opportunities if timed correctly.")
                            elif high_vol_pct < 40:
                                report.append(f"✓ Low Volatility Entries: {high_vol_pct:.0f}% of entries occurred during high volatility periods.")
                                report.append("  You mostly entered during calmer market conditions, reducing entry risk.")
            
            if 'avg_ema20_distance' in entry_analysis:
                ema_dist = entry_analysis['avg_ema20_distance']
                # Handle both ratio (0.02 = 2%) and percentage (2.0 = 2%) formats
                if abs(ema_dist) > 1:
                    # Already in percentage format
                    ema_pct = ema_dist
                else:
                    # Ratio format, convert to percentage
                    ema_pct = ema_dist * 100
                
                if ema_pct < -2:  # More than 2% below EMA
                    report.append(f"✓ Contrarian Entries: Average entry was {abs(ema_pct):.1f}% below EMA(20).")
                    report.append("  You bought when prices were below the moving average, catching value opportunities.")
                elif ema_pct > 2:  # More than 2% above EMA
                    report.append(f"⚠ Momentum Entries: Average entry was {ema_pct:.1f}% above EMA(20).")
                    report.append("  You bought during uptrends, which worked well for this stock.")
                else:
                    report.append(f"✓ Balanced Entries: Average entry was {abs(ema_pct):.1f}% from EMA(20).")
                    report.append("  You entered at reasonable price levels relative to the trend.")
            
            if 'trend_regimes' in entry_analysis:
                regimes = entry_analysis['trend_regimes']
                total_regime_trades = sum(regimes.values())
                if total_regime_trades > 0:
                    if 'uptrend' in regimes:
                        uptrend_pct = (regimes['uptrend'] / total_regime_trades * 100)
                        if uptrend_pct > 50:
                            report.append(f"✓ Trend Following: {uptrend_pct:.0f}% of entries occurred during uptrends.")
                            report.append("  You successfully rode the upward momentum of this stock.")
            
            if 'vol_regimes' in entry_analysis:
                vol_regimes = entry_analysis['vol_regimes']
                total_vol_trades = sum(vol_regimes.values())
                if total_vol_trades > 0:
                    if 'low' in vol_regimes:
                        low_vol_pct = (vol_regimes['low'] / total_vol_trades * 100)
                        if low_vol_pct > 50:
                            report.append(f"✓ Low Volatility Strategy: {low_vol_pct:.0f}% of entries occurred during low volatility periods.")
                            report.append("  You entered when markets were calmer, reducing entry risk.")
        
        # Exit timing analysis
        exit_analysis = best_data['exit_analysis']
        if exit_analysis:
            if 'avg_holding_duration' in exit_analysis:
                duration = exit_analysis['avg_holding_duration']
                if duration < 3:
                    report.append(f"✓ Quick Profits: Average holding period was {duration:.1f} days.")
                    report.append("  You captured profits quickly, avoiding potential reversals.")
                elif duration > 10:
                    report.append(f"✓ Patient Holding: Average holding period was {duration:.1f} days.")
                    report.append("  You held positions long enough to capture significant moves.")
                else:
                    report.append(f"✓ Balanced Holding: Average holding period was {duration:.1f} days.")
                    report.append("  You held positions for an appropriate duration.")
        
        # Position sizing (if available)
        if 'position_size_normalized_by_volatility' in best_data['trades'].columns:
            size_data = best_data['trades']['position_size_normalized_by_volatility'].dropna()
            if len(size_data) > 0:
                avg_size = size_data.mean()
                if avg_size > 0:
                    report.append(f"✓ Position Sizing: Average volatility-adjusted position size was {avg_size:.2f}.")
                    report.append("  Your position sizing was appropriate for the risk level.")
        
        report.append("")
        report.append("-" * 80)
        report.append(f"WORST PERFORMING STOCK: {worst_symbol}")
        report.append("-" * 80)
        report.append("")
        report.append(f"Total P&L: ${worst_data['total_pnl']:,.2f}")
        report.append(f"Average P&L per trade: ${worst_data['avg_pnl']:,.2f}")
        report.append(f"Win Rate: {worst_data['win_rate']:.1f}% ({worst_data['num_wins']} wins, {worst_data['num_losses']} losses)")
        report.append(f"Number of trades: {worst_data['num_trades']}")
        report.append(f"Average holding duration: {worst_data['avg_holding_duration']:.1f} days")
        report.append("")
        report.append("WHY THIS STOCK PERFORMED POORLY:")
        report.append("")
        
        # Entry timing analysis for worst stock
        entry_analysis = worst_data['entry_analysis']
        if entry_analysis:
            if 'avg_rsi' in entry_analysis:
                rsi = entry_analysis['avg_rsi']
                if rsi > 70:
                    report.append(f"✗ Poor Entry Timing: Average RSI at entry was {rsi:.1f} (overbought conditions).")
                    report.append("  You bought when the stock was overvalued, entering at the top of moves.")
                elif rsi < 30:
                    report.append(f"⚠ Oversold Entries: Average RSI at entry was {rsi:.1f} (oversold conditions).")
                    report.append("  Despite buying at oversold levels, the stock continued to decline.")
                    report.append("  This suggests the stock was in a strong downtrend or had fundamental issues.")
                else:
                    report.append(f"⚠ Neutral Entry Timing: Average RSI at entry was {rsi:.1f}.")
                    report.append("  Entry timing was reasonable, but other factors (volatility, trend, exit timing) likely caused losses.")
                
                if 'overbought_entries' in entry_analysis and entry_analysis['overbought_entries'] > 0:
                    worst_buy_trades = worst_data['trades']
                    if 'side' in worst_buy_trades.columns:
                        side_col = worst_buy_trades['side'].astype(str).str.lower().str.strip()
                        buy_count = len(worst_buy_trades[(side_col == 'buy') & (side_col != 'nan')])
                        report.append(f"  - {entry_analysis['overbought_entries']} out of {buy_count} buy trades occurred during overbought conditions (RSI > 70)")
            
            if 'avg_volatility' in entry_analysis:
                if 'high_vol_entries' in entry_analysis:
                    worst_buy_trades = worst_data['trades']
                    if 'side' in worst_buy_trades.columns:
                        side_col = worst_buy_trades['side'].astype(str).str.lower().str.strip()
                        buy_count = len(worst_buy_trades[(side_col == 'buy') & (side_col != 'nan')])
                        if buy_count > 0:
                            high_vol_pct = (entry_analysis['high_vol_entries'] / buy_count * 100)
                            if high_vol_pct > 60:
                                report.append(f"✗ High Volatility Risk: {high_vol_pct:.0f}% of entries occurred during high volatility periods.")
                                report.append("  Entering during high volatility increased risk, and the stock moved against you.")
            
            if 'avg_ema20_distance' in entry_analysis:
                ema_dist = entry_analysis['avg_ema20_distance']
                # Handle both ratio (0.02 = 2%) and percentage (2.0 = 2%) formats
                if abs(ema_dist) > 1:
                    # Already in percentage format
                    ema_pct = ema_dist
                else:
                    # Ratio format, convert to percentage
                    ema_pct = ema_dist * 100
                
                if ema_pct > 5:  # More than 5% above EMA
                    report.append(f"✗ Chasing Trends: Average entry was {ema_pct:.1f}% above EMA(20).")
                    report.append("  You bought too high during uptrends, entering after significant moves had already occurred.")
                elif ema_pct < -5:  # More than 5% below EMA
                    report.append(f"⚠ Value Trap: Average entry was {abs(ema_pct):.1f}% below EMA(20).")
                    report.append("  Despite buying below the moving average, the stock continued to decline.")
                    report.append("  This suggests a strong downtrend or fundamental weakness.")
            
            if 'trend_regimes' in entry_analysis:
                regimes = entry_analysis['trend_regimes']
                total_regime_trades = sum(regimes.values())
                if total_regime_trades > 0:
                    if 'downtrend' in regimes:
                        downtrend_pct = (regimes['downtrend'] / total_regime_trades * 100)
                        if downtrend_pct > 40:
                            report.append(f"✗ Fighting the Trend: {downtrend_pct:.0f}% of entries occurred during downtrends.")
                            report.append("  You bought during declining markets, fighting against the prevailing trend.")
        
        # Exit timing analysis for worst stock
        exit_analysis = worst_data['exit_analysis']
        if exit_analysis:
            if 'avg_holding_duration' in exit_analysis:
                duration = exit_analysis['avg_holding_duration']
                if duration > 15:
                    report.append(f"✗ Holding Losers Too Long: Average holding period was {duration:.1f} days.")
                    report.append("  You held losing positions too long, allowing losses to compound.")
                elif duration < 1:
                    report.append(f"⚠ Premature Exits: Average holding period was {duration:.1f} days.")
                    report.append("  You exited positions too quickly, potentially missing recovery opportunities.")
        
        # Compare with best stock
        report.append("")
        report.append("-" * 80)
        report.append("KEY DIFFERENCES BETWEEN BEST AND WORST PERFORMERS:")
        report.append("-" * 80)
        report.append("")
        
        best_entry = best_data['entry_analysis']
        worst_entry = worst_data['entry_analysis']
        
        if 'avg_rsi' in best_entry and 'avg_rsi' in worst_entry:
            rsi_diff = best_entry['avg_rsi'] - worst_entry['avg_rsi']
            if abs(rsi_diff) > 10:
                report.append(f"• Entry RSI Difference: Best stock entered at RSI {best_entry['avg_rsi']:.1f} vs worst at {worst_entry['avg_rsi']:.1f}.")
                report.append("  Better entry timing (lower RSI) contributed to the best stock's success.")
        
        if best_data['avg_holding_duration'] > 0 and worst_data['avg_holding_duration'] > 0:
            duration_diff = best_data['avg_holding_duration'] - worst_data['avg_holding_duration']
            if abs(duration_diff) > 2:
                report.append(f"• Holding Duration: Best stock held for {best_data['avg_holding_duration']:.1f} days vs worst at {worst_data['avg_holding_duration']:.1f} days.")
                report.append("  Different holding strategies may have impacted results.")
        
        if best_data['win_rate'] > worst_data['win_rate'] + 20:
            report.append(f"• Win Rate: Best stock had {best_data['win_rate']:.1f}% win rate vs worst at {worst_data['win_rate']:.1f}%.")
            report.append("  The best stock had significantly more winning trades.")
        
        report.append("")
        report.append("=" * 80)
        report.append("RECOMMENDATIONS:")
        report.append("=" * 80)
        report.append("")
        report.append("Based on this analysis:")
        report.append("")
        
        # Generate recommendations
        if 'avg_rsi' in worst_entry and worst_entry['avg_rsi'] > 65:
            report.append("1. Improve Entry Timing: Avoid buying when RSI > 65 (overbought).")
            report.append("   Wait for pullbacks or oversold conditions (RSI < 35) for better entries.")
        
        if 'avg_ema20_distance' in worst_entry:
            ema_dist = worst_entry['avg_ema20_distance']
            ema_pct = ema_dist if abs(ema_dist) > 1 else ema_dist * 100
            if ema_pct > 3:
                report.append("2. Avoid Chasing: Don't buy stocks more than 3% above their EMA(20).")
                report.append("   Wait for prices to pull back closer to the moving average.")
        
        if worst_data['win_rate'] < 30:
            report.append("3. Review Trade Selection: Low win rate suggests poor stock selection or timing.")
            report.append("   Consider focusing on stocks with better technical setups and fundamentals.")
        
        if worst_data['avg_holding_duration'] > 10 and worst_data['total_pnl'] < 0:
            report.append("4. Cut Losses Faster: Don't hold losing positions too long.")
            report.append("   Set stop-losses to limit downside and preserve capital.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

