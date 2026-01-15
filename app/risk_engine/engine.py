"""Risk Analysis Engine: Computes VaR, CVaR, Sharpe Ratio, and Max Drawdown."""

from typing import Dict, List, Any
import numpy as np
from app.models.risk import RiskMetrics
from app.models.portfolio import Portfolio


class RiskEngine:
    """
    Core risk computation engine.
    
    Responsibilities:
    - Calculate Value at Risk (VaR) at multiple confidence levels
    - Calculate Conditional VaR (CVaR / Expected Shortfall)
    - Compute Sharpe Ratio
    - Calculate Maximum Drawdown
    - Compute portfolio volatility
    
    Note: All calculations are based on historical data.
    This engine does NOT predict future risk.
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize risk engine.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation (default: 0)
        """
        self.risk_free_rate = risk_free_rate
    
    def compute_risk_metrics(self, portfolio: Portfolio, 
                            portfolio_values: List[float] = None) -> RiskMetrics:
        """
        Compute all risk metrics for a portfolio.
        
        Args:
            portfolio: Portfolio object with price data and weights
            portfolio_values: Optional pre-computed portfolio value time series
            
        Returns:
            RiskMetrics object with all computed metrics
        """
        # Calculate portfolio returns if not provided
        if portfolio_values is None:
            portfolio_values = self._calculate_portfolio_values(portfolio)
        
        # Calculate returns from portfolio values
        returns = self._calculate_returns_from_values(portfolio_values)
        
        if len(returns) == 0:
            raise ValueError("Insufficient data to compute risk metrics")
        
        # Compute metrics
        var_95 = self._compute_var(returns, confidence_level=0.95)
        var_99 = self._compute_var(returns, confidence_level=0.99)
        cvar_95 = self._compute_cvar(returns, confidence_level=0.95)
        cvar_99 = self._compute_cvar(returns, confidence_level=0.99)
        sharpe_ratio = self._compute_sharpe_ratio(returns)
        max_drawdown = self._compute_max_drawdown(portfolio_values)
        volatility = self._compute_volatility(returns)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            portfolio_value_timeseries=portfolio_values
        )
    
    def _calculate_portfolio_values(self, portfolio: Portfolio) -> List[float]:
        """
        Calculate portfolio value time series from price data.
        
        Args:
            portfolio: Portfolio object
            
        Returns:
            List of portfolio values (normalized to start at 100)
        """
        price_data = portfolio.price_data
        weights = portfolio.weights
        
        if not price_data:
            return []
        
        # Get number of periods
        num_periods = len(portfolio.dates)
        portfolio_values = []
        
        for i in range(num_periods):
            portfolio_value = 0.0
            
            for ticker in portfolio.tickers:
                if ticker in price_data and i < len(price_data[ticker]):
                    price = price_data[ticker][i]
                    weight = weights.get(ticker, 0.0)
                    
                    if i == 0:
                        # First period: normalize to 100
                        portfolio_value += weight * 100.0
                    else:
                        # Calculate return from first period
                        if len(price_data[ticker]) > 0 and price_data[ticker][0] is not None:
                            if price is not None and price_data[ticker][0] != 0:
                                return_factor = price / price_data[ticker][0]
                                portfolio_value += weight * 100.0 * return_factor
            
            portfolio_values.append(portfolio_value)
        
        return portfolio_values
    
    def _calculate_returns_from_values(self, values: List[float]) -> np.ndarray:
        """
        Calculate returns from portfolio value time series.
        
        Args:
            values: Portfolio values over time
            
        Returns:
            Array of daily returns
        """
        if len(values) < 2:
            return np.array([])
        
        returns = []
        for i in range(1, len(values)):
            if values[i-1] is not None and values[i] is not None and values[i-1] != 0:
                ret = (values[i] - values[i-1]) / values[i-1]
                returns.append(ret)
        
        return np.array(returns)
    
    def _compute_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Compute Value at Risk (VaR) using historical simulation.
        
        VaR is the maximum loss expected at a given confidence level.
        
        Args:
            returns: Portfolio return time series
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            VaR as a positive number (loss amount)
        """
        if len(returns) == 0:
            return 0.0
        
        # VaR is the (1 - confidence_level) percentile of losses
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)
        
        # Return as positive number (loss)
        return abs(var)
    
    def _compute_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Compute Conditional VaR (CVaR) / Expected Shortfall.
        
        CVaR is the expected loss given that the loss exceeds VaR.
        It provides a measure of tail risk.
        
        Args:
            returns: Portfolio return time series
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            CVaR as a positive number (expected loss)
        """
        if len(returns) == 0:
            return 0.0
        
        # Calculate VaR threshold
        var_threshold = self._compute_var(returns, confidence_level)
        
        # Find returns that exceed VaR (losses beyond VaR)
        tail_returns = returns[returns <= -var_threshold]
        
        if len(tail_returns) == 0:
            # If no tail events, use worst returns
            tail_percentile = (1 - confidence_level) * 100
            tail_returns = returns[returns <= np.percentile(returns, tail_percentile)]
        
        if len(tail_returns) == 0:
            return abs(var_threshold)
        
        # CVaR is the mean of tail losses
        cvar = np.mean(tail_returns)
        
        # Return as positive number
        return abs(cvar)
    
    def _compute_sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        Compute annualized Sharpe Ratio.
        
        Sharpe Ratio = (Mean Return - Risk-Free Rate) / Volatility
        
        Args:
            returns: Portfolio return time series
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        if volatility == 0:
            return 0.0
        
        # Annualize (assuming daily returns)
        annualized_return = mean_return * 252
        annualized_volatility = volatility * np.sqrt(252)
        
        sharpe = (annualized_return - self.risk_free_rate) / annualized_volatility
        
        return float(sharpe)
    
    def _compute_max_drawdown(self, values: List[float]) -> float:
        """
        Compute maximum drawdown as percentage.
        
        Drawdown = (Peak - Trough) / Peak
        
        Args:
            values: Portfolio value time series
            
        Returns:
            Maximum drawdown as percentage (0-100)
        """
        if len(values) < 2:
            return 0.0
        
        values_array = np.array(values)
        peak = values_array[0]
        max_drawdown = 0.0
        
        for value in values_array:
            if value > peak:
                peak = value
            drawdown = ((peak - value) / peak) * 100.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return float(max_drawdown)
    
    def _compute_volatility(self, returns: np.ndarray) -> float:
        """
        Compute annualized portfolio volatility.
        
        Args:
            returns: Portfolio return time series
            
        Returns:
            Annualized volatility (standard deviation)
        """
        if len(returns) == 0:
            return 0.0
        
        # Annualize daily volatility
        daily_volatility = np.std(returns)
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return float(annualized_volatility)
