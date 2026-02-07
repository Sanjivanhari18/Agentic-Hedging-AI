"""
Counterfactual Analysis Engine for Behavioral Trading.

Generates what-if scenarios:
- Holding duration: "If you held X days longer..."
- Position sizing: "If you used ATR-based sizing..."
- Entry timing: "If you entered at EMA touch..."
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
import logging

logger = logging.getLogger(__name__)


class CounterfactualEngine:
    """Generates counterfactual what-if analyses for trading behavior."""

    def __init__(self):
        self.results: Optional[Dict] = None

    def analyze(self, features: pd.DataFrame, market_data_fetcher=None) -> Dict:
        """Run all counterfactual analyses."""
        results: Dict[str, Any] = {}
        results["holding_duration"] = self.simulate_alternative_hold_durations(
            features, market_data_fetcher
        )
        results["position_sizing"] = self.simulate_alternative_sizing(features)
        results["entry_timing"] = self.simulate_alternative_entry(features)
        results["statements"] = self.generate_counterfactual_statements(
            features, results
        )
        self.results = results
        return results

    def simulate_alternative_hold(
        self, trade: pd.Series, extra_days: int, market_data: pd.DataFrame
    ) -> Dict:
        """Simulate P&L if a specific trade was held longer/shorter."""
        entry_price = trade.get("entry_price") or trade.get("price")
        if pd.isna(entry_price) or entry_price <= 0:
            return {"pnl_change": 0.0, "hypothetical_exit_price": np.nan, "valid": False}

        quantity = trade.get("quantity", 0)
        if quantity <= 0:
            return {"pnl_change": 0.0, "hypothetical_exit_price": np.nan, "valid": False}

        actual_exit_price = trade.get("price")
        actual_pnl = (actual_exit_price - entry_price) * quantity

        date_col = "date" if "date" in trade.index else "Date"
        exit_date = trade.get(date_col)
        if pd.isna(exit_date):
            exit_date = trade.get("Date")

        hyp_exit_price = np.nan
        if market_data is not None and len(market_data) > 0 and exit_date is not None:
            exit_dt = pd.to_datetime(exit_date)
            target_dt = exit_dt + pd.Timedelta(days=extra_days)
            market_data = market_data.copy()
            md_date = "Date" if "Date" in market_data.columns else "date"
            market_data[md_date] = pd.to_datetime(market_data[md_date])
            future = market_data[market_data[md_date] >= target_dt]
            if len(future) > 0:
                future = future.sort_values(md_date)
                hyp_exit_price = future["Close"].iloc[0]

        if np.isnan(hyp_exit_price):
            vol = trade.get("volatility_rolling_std") or 0.02
            if vol <= 0:
                vol = 0.02
            drift = 0.0
            random_effect = np.sqrt(extra_days) * vol
            price_return = drift + random_effect
            hyp_exit_price = actual_exit_price * (1 + price_return)

        hyp_pnl = (hyp_exit_price - entry_price) * quantity
        pnl_change = hyp_pnl - actual_pnl

        return {
            "pnl_change": float(pnl_change),
            "hypothetical_exit_price": float(hyp_exit_price),
            "actual_pnl": float(actual_pnl),
            "hypothetical_pnl": float(hyp_pnl),
            "valid": True,
        }

    def simulate_alternative_hold_durations(
        self,
        features: pd.DataFrame,
        market_data_fetcher=None,
    ) -> Dict:
        """
        For sell trades, estimate what P&L would have been if held:
        - 1 day shorter, 1 day longer
        - 3 days shorter, 3 days longer
        - 5 days shorter, 5 days longer
        Uses the Close price from market data if available, otherwise estimates from volatility.
        """
        sell_mask = features["side"].str.lower() == "sell"
        sells = features[sell_mask].copy()
        if len(sells) == 0:
            return {
                "by_duration": {},
                "avg_pnl_change": {},
                "total_actual_pnl": 0.0,
                "n_trades": 0,
            }

        date_col = "date" if "date" in sells.columns else "Date"
        symbol_col = "symbol" if "symbol" in sells.columns else None

        durations = [-5, -3, -1, 1, 3, 5]
        by_duration: Dict[int, List[float]] = {d: [] for d in durations}
        total_actual_pnl = 0.0

        for idx, trade in sells.iterrows():
            entry_price = trade.get("entry_price")
            if pd.isna(entry_price):
                entry_price = trade.get("price")
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            quantity = trade.get("quantity", 0)
            if quantity <= 0:
                continue

            actual_exit = trade.get("price")
            actual_pnl = (actual_exit - entry_price) * quantity
            total_actual_pnl += actual_pnl

            vol = trade.get("volatility_rolling_std")
            if vol is None or pd.isna(vol) or vol <= 0:
                vol = 0.02

            market_data = None
            if market_data_fetcher is not None and hasattr(
                market_data_fetcher, "fetch_ohlcv"
            ):
                try:
                    sym = trade.get(symbol_col) if symbol_col else None
                    if sym and pd.notna(trade.get(date_col)):
                        exit_dt = pd.to_datetime(trade[date_col])
                        start = (exit_dt - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
                        end = (exit_dt + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
                        market_data = market_data_fetcher.fetch_ohlcv(sym, start, end)
                        if "Close" not in market_data.columns and len(market_data.columns) > 0:
                            market_data = None
                except Exception as e:
                    logger.debug("Market data fetch for hold simulation failed: %s", e)

            for extra_days in durations:
                if market_data is not None and len(market_data) > 0:
                    exit_dt = pd.to_datetime(trade[date_col])
                    target_dt = exit_dt + pd.Timedelta(days=extra_days)
                    md = market_data.copy()
                    md_date = "Date" if "Date" in md.columns else md.index.name or "Date"
                    if md_date not in md.columns and hasattr(md.index, "normalize"):
                        md = md.reset_index()
                        md_date = md.columns[0]
                    md[md_date] = pd.to_datetime(md[md_date])
                    if extra_days >= 0:
                        future = md[md[md_date] >= target_dt]
                    else:
                        past = md[md[md_date] <= target_dt]
                        future = past.sort_values(md_date).tail(1) if len(past) > 0 else pd.DataFrame()
                    if len(future) > 0:
                        future = future.sort_values(md_date)
                        hyp_price = future["Close"].iloc[0]
                    else:
                        hyp_price = np.nan
                else:
                    hyp_price = np.nan

                if np.isnan(hyp_price):
                    random_effect = np.sqrt(abs(extra_days)) * vol
                    sign = 1 if extra_days >= 0 else -1
                    price_return = sign * random_effect
                    hyp_price = float(actual_exit) * (1 + price_return)

                hyp_pnl = (hyp_price - entry_price) * quantity
                pnl_change = hyp_pnl - actual_pnl
                by_duration[extra_days].append(pnl_change)

        avg_pnl_change = {}
        for d in durations:
            if by_duration[d]:
                avg_pnl_change[d] = float(np.mean(by_duration[d]))
            else:
                avg_pnl_change[d] = 0.0

        total_actual = float(sells["realized_pnl"].sum()) if "realized_pnl" in sells.columns else total_actual_pnl
        if "realized_pnl" not in sells.columns:
            total_actual = total_actual_pnl

        return {
            "by_duration": {k: [float(x) for x in v] for k, v in by_duration.items()},
            "avg_pnl_change": avg_pnl_change,
            "total_actual_pnl": total_actual,
            "n_trades": len(sells),
        }

    def simulate_alternative_sizing(
        self, features: pd.DataFrame, sizing_strategy: str = "atr_based"
    ) -> Dict:
        """
        Replay trades with different position sizing strategies:
        - 'atr_based': Size = risk_budget / ATR
        - 'equal_weight': Equal dollar amount per trade
        - 'kelly': Kelly criterion based on win rate and payoff ratio

        Compare max drawdown, total P&L, Sharpe-like ratio for each.
        """
        sell_mask = features["side"].str.lower() == "sell"
        sells = features[sell_mask].copy()
        if len(sells) == 0:
            return {
                "atr_based": {"total_pnl": 0, "max_drawdown": 0, "sharpe_like": 0},
                "equal_weight": {"total_pnl": 0, "max_drawdown": 0, "sharpe_like": 0},
                "kelly": {"total_pnl": 0, "max_drawdown": 0, "sharpe_like": 0},
                "actual": {"total_pnl": 0, "max_drawdown": 0, "sharpe_like": 0},
            }

        pnl_col = "realized_pnl" if "realized_pnl" in sells.columns else "pnl"
        if pnl_col not in sells.columns:
            if "entry_price" in sells.columns and "price" in sells.columns and "quantity" in sells.columns:
                sells["realized_pnl"] = (
                    (sells["price"] - sells["entry_price"]) * sells["quantity"]
                )
                pnl_col = "realized_pnl"
            else:
                return {
                    "atr_based": {"total_pnl": 0, "max_drawdown": 0, "sharpe_like": 0},
                    "equal_weight": {"total_pnl": 0, "max_drawdown": 0, "sharpe_like": 0},
                    "kelly": {"total_pnl": 0, "max_drawdown": 0, "sharpe_like": 0},
                    "actual": {"total_pnl": 0, "max_drawdown": 0, "sharpe_like": 0},
                }

        returns = sells[pnl_col].values
        n = len(returns)

        fixed_risk = 1000.0
        atr_col = "atr_14"
        if atr_col not in sells.columns:
            sells[atr_col] = sells.get("volatility_rolling_std", 0.02) * sells["price"].fillna(100)

        atr = sells[atr_col].replace(0, np.nan).fillna(sells["price"].mean() * 0.02).values
        size_atr = fixed_risk / atr
        size_atr = size_atr / size_atr.mean() if size_atr.mean() != 0 else np.ones(n)
        pnl_atr = returns * size_atr
        cum_atr = np.cumsum(pnl_atr)
        dd_atr = np.maximum.accumulate(cum_atr) - cum_atr
        max_dd_atr = float(np.max(dd_atr)) if len(dd_atr) > 0 else 0.0
        total_atr = float(np.sum(pnl_atr))
        std_atr = np.std(pnl_atr)
        sharpe_atr = (np.mean(pnl_atr) / (std_atr + 1e-8)) * np.sqrt(252) if n > 1 else 0.0

        base_value = sells["price"].values * sells["quantity"].values
        base_value = np.where(np.isfinite(base_value), base_value, 0)
        eq_size = np.ones(n)
        if np.sum(base_value) > 0:
            target_per_trade = np.mean(base_value[base_value > 0])
            eq_size = target_per_trade / (base_value + 1e-8)
            eq_size = np.clip(eq_size, 0.1, 10)
        pnl_eq = returns * eq_size
        cum_eq = np.cumsum(pnl_eq)
        dd_eq = np.maximum.accumulate(cum_eq) - cum_eq
        max_dd_eq = float(np.max(dd_eq)) if len(dd_eq) > 0 else 0.0
        total_eq = float(np.sum(pnl_eq))
        std_eq = np.std(pnl_eq)
        sharpe_eq = (np.mean(pnl_eq) / (std_eq + 1e-8)) * np.sqrt(252) if n > 1 else 0.0

        wins = returns > 0
        win_rate = np.mean(wins) if n > 0 else 0.5
        avg_win = np.mean(returns[wins]) if np.any(wins) else 1.0
        avg_loss = np.mean(returns[~wins]) if np.any(~wins) else -1.0
        kelly_f = 0.0
        if avg_win > 1e-8:
            kelly_f = (win_rate * avg_win - (1 - win_rate) * abs(avg_loss)) / avg_win
        kelly_f = np.clip(kelly_f, 0.01, 0.25)
        size_kelly = np.ones(n) * kelly_f * 2
        size_kelly = np.clip(size_kelly, 0.1, 2.0)
        pnl_kelly = returns * size_kelly
        cum_kelly = np.cumsum(pnl_kelly)
        dd_kelly = np.maximum.accumulate(cum_kelly) - cum_kelly
        max_dd_kelly = float(np.max(dd_kelly)) if len(dd_kelly) > 0 else 0.0
        total_kelly = float(np.sum(pnl_kelly))
        std_kelly = np.std(pnl_kelly)
        sharpe_kelly = (np.mean(pnl_kelly) / (std_kelly + 1e-8)) * np.sqrt(252) if n > 1 else 0.0

        cum_actual = np.cumsum(returns)
        dd_actual = np.maximum.accumulate(cum_actual) - cum_actual
        max_dd_actual = float(np.max(dd_actual)) if len(dd_actual) > 0 else 0.0
        total_actual = float(np.sum(returns))
        std_actual = np.std(returns)
        sharpe_actual = (np.mean(returns) / (std_actual + 1e-8)) * np.sqrt(252) if n > 1 else 0.0

        return {
            "atr_based": {
                "total_pnl": total_atr,
                "max_drawdown": max_dd_atr,
                "sharpe_like": sharpe_atr,
            },
            "equal_weight": {
                "total_pnl": total_eq,
                "max_drawdown": max_dd_eq,
                "sharpe_like": sharpe_eq,
            },
            "kelly": {
                "total_pnl": total_kelly,
                "max_drawdown": max_dd_kelly,
                "sharpe_like": sharpe_kelly,
            },
            "actual": {
                "total_pnl": total_actual,
                "max_drawdown": max_dd_actual,
                "sharpe_like": sharpe_actual,
            },
        }

    def simulate_alternative_entry(self, features: pd.DataFrame) -> Dict:
        """
        Simulate alternative entry strategies:
        - EMA touch entries (enter when price touches EMA-20 instead of current entry)
        - RSI-based entries (only enter when RSI < 40)
        Compare win rates and average P&L.
        """
        sell_mask = features["side"].str.lower() == "sell"
        sells = features[sell_mask].copy()
        if len(sells) == 0:
            return {
                "current_win_rate": 0.0,
                "current_avg_pnl": 0.0,
                "ema_touch_win_rate": 0.0,
                "ema_touch_avg_pnl": 0.0,
                "rsi_entry_win_rate": 0.0,
                "rsi_entry_avg_pnl": 0.0,
                "n_trades": 0,
            }

        if "realized_pnl" not in sells.columns and "entry_price" in sells.columns:
            sells["realized_pnl"] = (
                (sells["price"] - sells["entry_price"]) * sells["quantity"]
            )
        pnl = sells["realized_pnl"].values if "realized_pnl" in sells.columns else np.zeros(len(sells))
        wins = pnl > 0
        current_win_rate = float(np.mean(wins)) if len(wins) > 0 else 0.0
        current_avg_pnl = float(np.mean(pnl)) if len(pnl) > 0 else 0.0

        ema_touch_win_rate = current_win_rate
        ema_touch_avg_pnl = current_avg_pnl
        if "entry_price" in sells.columns and "ema_20" in sells.columns:
            entry_dist_ema = (sells["entry_price"] - sells["ema_20"]) / (sells["ema_20"] + 1e-8)
            near_ema = entry_dist_ema.abs() < 0.02
            if np.any(near_ema):
                sub = pnl[near_ema.values]
                ema_touch_win_rate = float(np.mean(sub > 0))
                ema_touch_avg_pnl = float(np.mean(sub))
            else:
                above_ema = sells["entry_price"] > sells["ema_20"]
                if np.any(above_ema) and np.any(~above_ema):
                    wr_above = float(np.mean(pnl[above_ema.values] > 0))
                    wr_below = float(np.mean(pnl[(~above_ema).values] > 0))
                    if wr_below >= wr_above:
                        ema_touch_win_rate = wr_below
                        ema_touch_avg_pnl = float(np.mean(pnl[(~above_ema).values]))
                    else:
                        ema_touch_win_rate = wr_above
                        ema_touch_avg_pnl = float(np.mean(pnl[above_ema.values]))
        elif "entry_price_distance_from_ema20" in sells.columns:
            near_ema = sells["entry_price_distance_from_ema20"].abs() < 0.02
            if np.any(near_ema):
                sub = pnl[near_ema.values]
                ema_touch_win_rate = float(np.mean(sub > 0))
                ema_touch_avg_pnl = float(np.mean(sub))

        rsi_entry_win_rate = current_win_rate
        rsi_entry_avg_pnl = current_avg_pnl
        if "rsi_14" in sells.columns:
            rsi_low = sells["rsi_14"] < 40
            rsi_low = rsi_low.fillna(False)
            if np.any(rsi_low):
                sub = pnl[rsi_low.values]
                rsi_entry_win_rate = float(np.mean(sub > 0))
                rsi_entry_avg_pnl = float(np.mean(sub))

        return {
            "current_win_rate": current_win_rate,
            "current_avg_pnl": current_avg_pnl,
            "ema_touch_win_rate": float(ema_touch_win_rate),
            "ema_touch_avg_pnl": float(ema_touch_avg_pnl),
            "rsi_entry_win_rate": float(rsi_entry_win_rate),
            "rsi_entry_avg_pnl": float(rsi_entry_avg_pnl),
            "n_trades": len(sells),
        }

    def generate_counterfactual_statements(
        self, features: pd.DataFrame, analysis_results: Dict
    ) -> List[str]:
        """
        Generate natural language counterfactual insights.
        Returns list of statements like:
        "If you had held positions 2 days shorter on average, your estimated return would be 8% higher"
        """
        statements: List[str] = []
        h = analysis_results.get("holding_duration") or {}
        s = analysis_results.get("position_sizing") or {}
        e = analysis_results.get("entry_timing") or {}

        n_hold = h.get("n_trades", 0)
        avg_changes = h.get("avg_pnl_change") or {}
        total_actual = h.get("total_actual_pnl") or 0.0

        if n_hold > 0 and total_actual != 0 and avg_changes:
            for days, change in [(3, 3), (-3, -3), (1, 1), (5, 5)]:
                if days not in avg_changes:
                    continue
                avg_ch = avg_changes[days]
                pct = (avg_ch / (abs(total_actual) / max(n_hold, 1) + 1e-8)) * 100
                direction = "longer" if days > 0 else "shorter"
                effect = "higher" if (avg_ch > 0 and days > 0) or (avg_ch > 0 and days < 0) else "lower"
                statements.append(
                    f"If you had held positions {abs(days)} day(s) {direction}, "
                    f"estimated P&L change per trade: {avg_ch:+.2f} ({pct:+.1f}% vs actual); "
                    f"overall effect would be {effect}."
                )
            if len(statements) > 2:
                statements = statements[:2]

        actual_sizing = s.get("actual") or {}
        atr_sizing = s.get("atr_based") or {}
        if actual_sizing and atr_sizing:
            dd_actual = actual_sizing.get("max_drawdown") or 0
            dd_atr = atr_sizing.get("max_drawdown") or 0
            if dd_actual != 0 or dd_atr != 0:
                statements.append(
                    f"If you had used ATR-based sizing, max drawdown would be "
                    f"{dd_atr:.1f} vs {dd_actual:.1f} (actual)."
                )
            total_a = actual_sizing.get("total_pnl", 0)
            total_atr = atr_sizing.get("total_pnl", 0)
            if abs(total_a) > 1e-6:
                statements.append(
                    f"ATR-based sizing would have yielded total P&L {total_atr:.2f} vs "
                    f"actual {total_a:.2f}."
                )

        wr_cur = e.get("current_win_rate") or 0
        wr_ema = e.get("ema_touch_win_rate") or 0
        wr_rsi = e.get("rsi_entry_win_rate") or 0
        if wr_ema > 0 or wr_rsi > 0:
            if abs(wr_ema - wr_cur) > 0.02:
                statements.append(
                    f"If you had entered at EMA touch instead of current entries, "
                    f"win rate would be {wr_ema*100:.0f}% vs {wr_cur*100:.0f}% (current)."
                )
            if abs(wr_rsi - wr_cur) > 0.02 and wr_rsi > 0:
                statements.append(
                    f"If you had entered only when RSI < 40, win rate would be "
                    f"{wr_rsi*100:.0f}% vs {wr_cur*100:.0f}% (current)."
                )

        if not statements:
            statements.append(
                "Insufficient trade data for counterfactual statements; add more closed trades."
            )

        return statements[:6]
