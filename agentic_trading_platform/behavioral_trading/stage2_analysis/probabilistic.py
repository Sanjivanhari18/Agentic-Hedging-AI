"""
Probabilistic Analysis Module for Behavioral Trading.

Provides:
- Bayesian parameter estimation using scipy.stats (lightweight, no PyMC)
- Confidence intervals for all behavioral metrics
- Probability statements: "85% probability your position sizing increases after losses"
- Bootstrap confidence intervals for stability scores
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional, List, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class ProbabilisticAnalyzer:
    """Probabilistic analysis for behavioral trading patterns."""

    def __init__(self, confidence_level: float = 0.95, n_bootstrap: int = 1000):
        """
        Args:
            confidence_level: Confidence level for intervals (default 0.95)
            n_bootstrap: Number of bootstrap samples (default 1000)
        """
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.results: Optional[Dict] = None

    def analyze(self, features: pd.DataFrame, baselines: Dict) -> Dict:
        """Run full probabilistic analysis."""
        results = {}
        results['credible_intervals'] = self.compute_credible_intervals(features)
        results['behavioral_probabilities'] = self.compute_behavioral_probabilities(features)
        results['confidence_bands'] = self.compute_confidence_bands(features)
        results['probability_statements'] = self.generate_probability_statements(
            features, baselines
        )
        self.results = results
        return results

    def bayesian_estimate(
        self, data: np.ndarray, prior: str = 'uninformative'
    ) -> Dict:
        """
        Estimate parameter with credible intervals using conjugate priors.

        Uses Normal-Inverse-Gamma conjugate prior for mean estimation.
        With uninformative prior, approximates the data's posterior.

        Args:
            data: Array of observations
            prior: Prior type ('uninformative' or 'weakly_informative')
        Returns:
            Dict with mean, ci_lower, ci_upper, std
        """
        data = np.asarray(data)
        data = data[np.isfinite(data)]
        if len(data) == 0:
            return {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'std': np.nan}
        n = len(data)
        mean = float(np.mean(data))
        std = float(np.std(data, ddof=1)) if n > 1 else 0.0
        # With uninformative prior on (mu, sigma^2), posterior for mu is
        # t-distributed: t_{n-1}(sample_mean, se) where se = s/sqrt(n)
        se = std / np.sqrt(n) if n > 1 else 0.0
        alpha = 1.0 - self.confidence_level
        if n > 1 and se > 0:
            ci_lower = float(stats.t.ppf(alpha / 2, df=n - 1) * se + mean)
            ci_upper = float(stats.t.ppf(1 - alpha / 2, df=n - 1) * se + mean)
        else:
            ci_lower = ci_upper = mean
        return {
            'mean': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': std,
        }

    def behavioral_probability(
        self,
        features: pd.DataFrame,
        condition_col: str,
        behavior_col: str,
        condition_fn: Optional[Callable] = None,
        behavior_fn: Optional[Callable] = None,
    ) -> Dict:
        """
        Compute P(behavior | condition) using empirical Bayes.
        E.g., P(position_size > baseline | previous_trade_was_loss)
        Uses Beta(successes+1, failures+1) for credible interval on the probability.
        """
        if condition_col not in features.columns or behavior_col not in features.columns:
            logger.warning(
                f"Missing columns {condition_col} or {behavior_col} for behavioral_probability"
            )
            return {
                'probability': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'n_condition': 0,
                'n_behavior_given_condition': 0,
            }
        df = features.dropna(subset=[condition_col, behavior_col])
        if condition_fn is not None:
            cond_mask = condition_fn(df[condition_col])
        else:
            cond_mask = df[condition_col].astype(bool)
        given = df.loc[cond_mask]
        n_condition = len(given)
        if n_condition == 0:
            return {
                'probability': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'n_condition': 0,
                'n_behavior_given_condition': 0,
            }
        if behavior_fn is not None:
            behavior_mask = behavior_fn(given[behavior_col])
        else:
            behavior_mask = given[behavior_col].astype(bool)
        successes = int(behavior_mask.sum())
        failures = n_condition - successes
        # Beta(1, 1) prior (uniform); posterior Beta(successes+1, failures+1)
        alpha_post = successes + 1
        beta_post = failures + 1
        prob = alpha_post / (alpha_post + beta_post)
        alpha_ci = 1.0 - self.confidence_level
        ci_lower = float(stats.beta.ppf(alpha_ci / 2, alpha_post, beta_post))
        ci_upper = float(stats.beta.ppf(1 - alpha_ci / 2, alpha_post, beta_post))
        return {
            'probability': float(prob),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_condition': n_condition,
            'n_behavior_given_condition': successes,
        }

    def confidence_band(
        self, metric_series: pd.Series, method: str = 'bootstrap'
    ) -> Dict:
        """Generate confidence bands for time series metrics using rolling window bootstrap."""
        s = metric_series.dropna()
        if len(s) < 2:
            return {
                'lower': s,
                'upper': s,
                'point_estimate': s,
                'method': method,
            }
        if method != 'bootstrap':
            method = 'bootstrap'
        # Rolling window bootstrap: for each window, bootstrap the values and get CI
        window_size = min(20, max(5, len(s) // 5))
        lower = []
        upper = []
        point_est = []
        for i in range(len(s)):
            start = max(0, i - window_size + 1)
            end = i + 1
            window = s.iloc[start:end].values
            if len(window) < 2:
                lower.append(window[-1] if len(window) else np.nan)
                upper.append(window[-1] if len(window) else np.nan)
                point_est.append(window[-1] if len(window) else np.nan)
                continue
            boot_means = []
            for _ in range(self.n_bootstrap):
                idx = np.random.choice(len(window), size=len(window), replace=True)
                boot_means.append(np.mean(window[idx]))
            boot_means = np.array(boot_means)
            alpha = 1.0 - self.confidence_level
            point_est.append(float(np.mean(window)))
            lower.append(float(np.percentile(boot_means, 100 * alpha / 2)))
            upper.append(float(np.percentile(boot_means, 100 * (1 - alpha / 2))))
        return {
            'lower': pd.Series(lower, index=s.index),
            'upper': pd.Series(upper, index=s.index),
            'point_estimate': pd.Series(point_est, index=s.index),
            'method': method,
        }

    def compute_credible_intervals(self, features: pd.DataFrame) -> Dict:
        """Compute Bayesian credible intervals for key behavioral metrics."""
        key_metrics = [
            'holding_duration_days',
            'position_size_dollar_value',
            'trades_per_day',
            'time_gap_days_since_last_trade',
            'realized_pnl',
        ]
        result = {}
        for col in key_metrics:
            if col not in features.columns:
                continue
            data = features[col].replace([np.inf, -np.inf], np.nan).dropna().values
            if len(data) > 0:
                result[col] = self.bayesian_estimate(data)
            else:
                result[col] = {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'std': np.nan}
        return result

    def compute_behavioral_probabilities(self, features: pd.DataFrame) -> Dict:
        """
        Compute conditional probabilities for key behavioral patterns:
        - P(position_size_increase | previous_loss)
        - P(hold_longer | winning_trade)
        - P(trade_within_48h | loss)
        - P(larger_position | high_volatility)
        """
        df = features.copy()
        results = {}

        # Build previous-trade indicators if not present
        if 'realized_pnl' in df.columns:
            df['is_loss'] = (df['realized_pnl'] < 0).astype(int)
        else:
            df['is_loss'] = 0
        df['previous_was_loss'] = df['is_loss'].shift(1).fillna(0).astype(int)
        df['is_winning'] = (df['realized_pnl'] > 0).astype(int) if 'realized_pnl' in df.columns else 0

        # P(position_size increase | previous loss)
        size_col = 'position_size_dollar_value'
        if size_col in df.columns:
            df['position_size_prev'] = df[size_col].shift(1)
            df['position_size_increase'] = (df[size_col] > df['position_size_prev']).astype(int)
            r = self.behavioral_probability(
                df,
                'previous_was_loss',
                'position_size_increase',
                condition_fn=lambda x: x == 1,
                behavior_fn=lambda x: x == 1,
            )
            results['P_position_size_increase_given_previous_loss'] = r
        else:
            alt = 'position_size_normalized_by_volatility'
            if alt in df.columns:
                df['position_size_prev'] = df[alt].shift(1)
                df['position_size_increase'] = (df[alt] > df['position_size_prev']).astype(int)
                r = self.behavioral_probability(
                    df,
                    'previous_was_loss',
                    'position_size_increase',
                    condition_fn=lambda x: x == 1,
                    behavior_fn=lambda x: x == 1,
                )
                results['P_position_size_increase_given_previous_loss'] = r

        # P(hold longer than median | winning trade)
        hold_col = 'holding_duration_days'
        if hold_col in df.columns:
            median_hold = df[hold_col].median()
            df['hold_longer_than_median'] = (df[hold_col] > median_hold).astype(int)
            r = self.behavioral_probability(
                df,
                'is_winning',
                'hold_longer_than_median',
                condition_fn=lambda x: x == 1,
                behavior_fn=lambda x: x == 1,
            )
            results['P_hold_longer_than_median_given_winning'] = r

        # P(trade within 48h | previous loss)
        gap_col = 'time_gap_hours_since_last_trade'
        if gap_col not in df.columns and 'time_gap_days_since_last_trade' in df.columns:
            df['time_gap_hours_since_last_trade'] = df['time_gap_days_since_last_trade'] * 24
            gap_col = 'time_gap_hours_since_last_trade'
        if gap_col in df.columns:
            df['trade_within_48h'] = (df[gap_col] <= 48).astype(int)
            r = self.behavioral_probability(
                df,
                'previous_was_loss',
                'trade_within_48h',
                condition_fn=lambda x: x == 1,
                behavior_fn=lambda x: x == 1,
            )
            results['P_trade_within_48h_given_previous_loss'] = r

        # P(larger position | high volatility)
        vol_col = 'volatility_regime'
        if vol_col in df.columns:
            df['high_vol'] = (df[vol_col] == 1).astype(int)
        else:
            vol_std = 'volatility_rolling_std'
            if vol_std in df.columns:
                med = df[vol_std].median()
                df['high_vol'] = (df[vol_std] > med).astype(int)
                vol_col = 'high_vol'
            else:
                df['high_vol'] = 0
        if 'position_size_dollar_value' in df.columns:
            med_size = df['position_size_dollar_value'].median()
            df['larger_position'] = (df['position_size_dollar_value'] > med_size).astype(int)
            r = self.behavioral_probability(
                df,
                'high_vol',
                'larger_position',
                condition_fn=lambda x: x == 1,
                behavior_fn=lambda x: x == 1,
            )
            results['P_larger_position_given_high_volatility'] = r

        return results

    def compute_confidence_bands(self, features: pd.DataFrame) -> Dict:
        """Compute rolling confidence bands for behavioral metrics over time."""
        key_metrics = [
            'holding_duration_days',
            'position_size_dollar_value',
            'trades_per_day',
            'time_gap_days_since_last_trade',
            'realized_pnl',
        ]
        result = {}
        for col in key_metrics:
            if col not in features.columns:
                continue
            s = features[col].replace([np.inf, -np.inf], np.nan)
            if s.notna().sum() < 2:
                continue
            result[col] = self.confidence_band(s, method='bootstrap')
        return result

    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_fn: Callable = np.mean,
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for any statistic.
        Returns: (estimate, ci_lower, ci_upper)
        """
        data = np.asarray(data)
        data = data[np.isfinite(data)]
        n = len(data)
        if n == 0:
            return (np.nan, np.nan, np.nan)
        estimate = float(statistic_fn(data))
        if n == 1:
            return (estimate, estimate, estimate)
        boot = np.array(
            [statistic_fn(np.random.choice(data, size=n, replace=True)) for _ in range(self.n_bootstrap)]
        )
        alpha = 1.0 - self.confidence_level
        ci_lower = float(np.percentile(boot, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot, 100 * (1 - alpha / 2)))
        return (estimate, ci_lower, ci_upper)

    def generate_probability_statements(
        self, features: pd.DataFrame, baselines: Dict
    ) -> List[str]:
        """
        Generate natural language probability statements like:
        "There is an 85% probability that your position sizing increases after losses"
        """
        statements = []
        probs = self.compute_behavioral_probabilities(features)
        intervals = self.compute_credible_intervals(features)

        # Round probability to integer for display
        def pct(p: float) -> str:
            if np.isnan(p):
                return "N/A"
            return f"{int(round(100 * p))}"

        # P(position size increase | previous loss)
        r = probs.get('P_position_size_increase_given_previous_loss', {})
        p = r.get('probability', np.nan)
        if not np.isnan(p) and r.get('n_condition', 0) > 0:
            statements.append(
                f"There is a {pct(p)}% probability that your position sizing increases after a loss."
            )

        # P(hold longer | winning)
        r = probs.get('P_hold_longer_than_median_given_winning', {})
        p = r.get('probability', np.nan)
        if not np.isnan(p) and r.get('n_condition', 0) > 0:
            statements.append(
                f"There is a {pct(p)}% probability that you hold winning trades longer than your median holding period."
            )

        # P(trade within 48h | loss)
        r = probs.get('P_trade_within_48h_given_previous_loss', {})
        p = r.get('probability', np.nan)
        if not np.isnan(p) and r.get('n_condition', 0) > 0:
            statements.append(
                f"There is a {pct(p)}% probability that you trade again within 48 hours after a loss."
            )

        # P(larger position | high volatility)
        r = probs.get('P_larger_position_given_high_volatility', {})
        p = r.get('probability', np.nan)
        if not np.isnan(p) and r.get('n_condition', 0) > 0:
            statements.append(
                f"There is a {pct(p)}% probability that you take larger positions when volatility is high."
            )

        # Credible interval statements for key metrics
        for name, col, unit in [
            ('holding duration (days)', 'holding_duration_days', 'days'),
            ('average position size', 'position_size_dollar_value', ''),
            ('trades per day', 'trades_per_day', ''),
        ]:
            ci = intervals.get(col)
            if not ci:
                continue
            mean, low, high = ci.get('mean'), ci.get('ci_lower'), ci.get('ci_upper')
            if np.isnan(mean):
                continue
            u = f" {unit}" if unit else ""
            statements.append(
                f"Your typical {name} is {mean:.2f}{u}, with {int(100*self.confidence_level)}% "
                f"credible interval [{low:.2f}, {high:.2f}]{u}."
            )

        # Realized PnL statement
        ci = intervals.get('realized_pnl')
        if ci and not np.isnan(ci.get('mean')):
            mean, low, high = ci['mean'], ci['ci_lower'], ci['ci_upper']
            statements.append(
                f"Average realized PnL per trade is {mean:.2f}, with {int(100*self.confidence_level)}% "
                f"credible interval [{low:.2f}, {high:.2f}]."
            )

        # Time gap statement
        ci = intervals.get('time_gap_days_since_last_trade')
        if ci and not np.isnan(ci.get('mean')):
            mean, low, high = ci['mean'], ci['ci_lower'], ci['ci_upper']
            statements.append(
                f"There is a {int(100*self.confidence_level)}% probability that your typical gap between "
                f"trades is between {low:.2f} and {high:.2f} days (mean {mean:.2f})."
            )

        return statements
