"""StressTestAgent: Simulates portfolio performance under historical stress scenarios."""

from typing import Dict, List, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from app.agents.base import BaseAgent
from app.models.agent import AgentType, AgentContext
from app.models.risk import StressScenario, StressTestResult


class StressTestAgent(BaseAgent):
    """
    Simulates portfolio performance under historical stress scenarios.
    
    Responsibilities:
    - Define stress scenario date ranges
    - Calculate portfolio performance during stress periods
    - Compute drawdown, volatility, and recovery metrics
    - Return structured stress test results
    
    Note: This agent does NOT predict future performance.
    It only analyzes how the portfolio would have performed during past crises.
    """
    
    # Historical stress scenario definitions
    STRESS_SCENARIOS = {
        StressScenario.FINANCIAL_CRISIS_2008: {
            "start_date": datetime(2008, 9, 15),  # Lehman Brothers bankruptcy
            "end_date": datetime(2009, 3, 9),     # Market bottom
            "recovery_end_date": datetime(2013, 3, 9)  # Approximate recovery
        },
        StressScenario.COVID_19_CRASH: {
            "start_date": datetime(2020, 2, 19),  # Market peak before crash
            "end_date": datetime(2020, 3, 23),     # Market bottom
            "recovery_end_date": datetime(2020, 8, 18)  # Recovery to pre-crash levels
        },
        StressScenario.CUSTOM_VOLATILITY_SHOCK: {
            "start_date": None,  # Will be determined dynamically
            "end_date": None,
            "recovery_end_date": None
        }
    }
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.STRESS_TEST
    
    def execute(self, context: AgentContext) -> Dict[str, Any]:
        """
        Run stress tests for all defined scenarios.
        
        Args:
            context: Must contain price_data and dates from DataFetchAgent
            
        Returns:
            Dictionary with stress test results for each scenario
        """
        # Extract data from context
        if "price_data" not in context.raw_data:
            raise ValueError("Price data not available. DataFetchAgent must run first.")
        
        price_data = context.raw_data["price_data"]
        dates = [datetime.fromisoformat(d) if isinstance(d, str) else d 
                for d in context.raw_data["dates"]]
        weights = context.weights
        
        # Calculate portfolio value time series
        portfolio_values = self._calculate_portfolio_values(price_data, weights, dates)
        
        # Run stress tests for each scenario
        results = {}
        
        for scenario in [StressScenario.FINANCIAL_CRISIS_2008, StressScenario.COVID_19_CRASH]:
            result = self._run_stress_test(
                scenario, portfolio_values, dates, context.analysis_date
            )
            if result:
                results[scenario.value] = result.dict()
        
        # Run custom volatility shock test
        custom_result = self._run_custom_volatility_shock(
            portfolio_values, dates, context.analysis_date
        )
        if custom_result:
            results[StressScenario.CUSTOM_VOLATILITY_SHOCK.value] = custom_result.dict()
        
        return {
            "stress_test_results": results,
            "portfolio_value_timeseries": portfolio_values,
            "scenarios_tested": list(results.keys())
        }
    
    def _calculate_portfolio_values(self, price_data: Dict[str, List[float]], 
                                   weights: Dict[str, float],
                                   dates: List[datetime]) -> List[float]:
        """
        Calculate portfolio value over time.
        
        Args:
            price_data: Price data per ticker
            weights: Portfolio weights
            dates: Date index
            
        Returns:
            List of portfolio values (normalized to start at 100)
        """
        if not price_data:
            raise ValueError("No price data available")
        
        # Initialize portfolio value
        num_periods = len(dates)
        portfolio_values = []
        
        # Calculate initial portfolio value (normalized)
        initial_value = 100.0
        
        # For each time period, calculate portfolio value
        for i in range(num_periods):
            portfolio_value = 0.0
            
            for ticker, ticker_prices in price_data.items():
                if i < len(ticker_prices) and ticker_prices[i] is not None:
                    weight = weights.get(ticker, 0.0)
                    # Normalize by first price to get relative performance
                    if i == 0:
                        portfolio_value += weight * 100.0  # Start at 100
                    else:
                        # Calculate return from first period
                        if len(ticker_prices) > 0 and ticker_prices[0] is not None:
                            return_factor = ticker_prices[i] / ticker_prices[0]
                            portfolio_value += weight * 100.0 * return_factor
            
            portfolio_values.append(portfolio_value)
        
        return portfolio_values
    
    def _run_stress_test(self, scenario: StressScenario, 
                        portfolio_values: List[float],
                        dates: List[datetime],
                        analysis_date: datetime) -> StressTestResult:
        """
        Run stress test for a specific historical scenario.
        
        Args:
            scenario: Stress scenario to test
            portfolio_values: Portfolio value time series
            dates: Date index
            analysis_date: Reference date for analysis
            
        Returns:
            StressTestResult or None if scenario dates not in data range
        """
        scenario_config = self.STRESS_SCENARIOS[scenario]
        start_date = scenario_config["start_date"]
        end_date = scenario_config["end_date"]
        recovery_end_date = scenario_config["recovery_end_date"]
        
        # Find indices for stress period
        start_idx = None
        end_idx = None
        recovery_idx = None
        
        for i, date in enumerate(dates):
            if start_idx is None and date >= start_date:
                start_idx = i
            if end_idx is None and date >= end_date:
                end_idx = i
            if recovery_idx is None and date >= recovery_end_date:
                recovery_idx = i
        
        # Check if we have data for this scenario
        if start_idx is None or end_idx is None:
            return None  # Scenario dates not in available data
        
        # Extract stress period data
        stress_values = portfolio_values[start_idx:end_idx+1]
        stress_dates = dates[start_idx:end_idx+1]
        
        if not stress_values:
            return None
        
        # Calculate metrics
        initial_value = stress_values[0]
        min_value = min(stress_values)
        peak_drawdown = ((initial_value - min_value) / initial_value) * 100.0
        
        # Find index of minimum value
        min_idx = stress_values.index(min_value)
        portfolio_value_at_peak_loss = min_value
        
        # Calculate volatility during stress (annualized)
        if len(stress_values) > 1:
            returns = np.diff(stress_values) / stress_values[:-1]
            volatility = np.std(returns) * np.sqrt(252) * 100.0  # Annualized %
        else:
            volatility = 0.0
        
        # Calculate recovery days
        recovery_days = None
        if recovery_idx is not None and recovery_idx < len(portfolio_values):
            recovery_values = portfolio_values[end_idx:recovery_idx+1]
            for i, value in enumerate(recovery_values):
                if value >= initial_value:
                    recovery_days = i
                    break
        
        return StressTestResult(
            scenario=scenario,
            start_date=stress_dates[0],
            end_date=stress_dates[-1],
            peak_drawdown=peak_drawdown,
            recovery_days=recovery_days,
            volatility_during_stress=volatility,
            portfolio_value_at_peak_loss=portfolio_value_at_peak_loss
        )
    
    def _run_custom_volatility_shock(self, portfolio_values: List[float],
                                    dates: List[datetime],
                                    analysis_date: datetime) -> StressTestResult:
        """
        Run custom volatility shock test (finds worst 30-day period).
        
        Args:
            portfolio_values: Portfolio value time series
            dates: Date index
            analysis_date: Reference date
            
        Returns:
            StressTestResult for worst volatility period
        """
        if len(portfolio_values) < 30:
            return None
        
        # Find worst 30-day rolling drawdown
        window = 30
        max_drawdown = 0.0
        worst_start_idx = 0
        worst_end_idx = window
        
        for i in range(len(portfolio_values) - window):
            window_values = portfolio_values[i:i+window]
            initial = window_values[0]
            minimum = min(window_values)
            drawdown = ((initial - minimum) / initial) * 100.0
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                worst_start_idx = i
                worst_end_idx = i + window
        
        # Calculate volatility for this period
        worst_values = portfolio_values[worst_start_idx:worst_end_idx]
        if len(worst_values) > 1:
            returns = np.diff(worst_values) / worst_values[:-1]
            volatility = np.std(returns) * np.sqrt(252) * 100.0
        else:
            volatility = 0.0
        
        return StressTestResult(
            scenario=StressScenario.CUSTOM_VOLATILITY_SHOCK,
            start_date=dates[worst_start_idx],
            end_date=dates[worst_end_idx-1],
            peak_drawdown=max_drawdown,
            recovery_days=None,  # Not applicable for custom scenario
            volatility_during_stress=volatility,
            portfolio_value_at_peak_loss=min(worst_values)
        )
