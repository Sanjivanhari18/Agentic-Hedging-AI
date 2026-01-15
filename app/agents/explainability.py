"""ExplainabilityAgent: Computes risk attribution and generates human-readable explanations."""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from app.agents.base import BaseAgent
from app.models.agent import AgentType, AgentContext
from app.models.risk import RiskAttribution


class ExplainabilityAgent(BaseAgent):
    """
    Computes asset-level risk contribution and generates explanations.
    
    Responsibilities:
    - Calculate risk contribution per asset (SHAP-style attribution)
    - Compute marginal contribution to portfolio risk
    - Generate human-readable explanations
    - Identify which assets drive portfolio risk
    
    Note: Uses variance-based risk decomposition (proxy for SHAP).
    In production, you might use actual SHAP values or other attribution methods.
    """
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.EXPLAINABILITY
    
    def execute(self, context: AgentContext) -> Dict[str, Any]:
        """
        Compute risk attribution for each asset.
        
        Args:
            context: Must contain price_data from DataFetchAgent
            
        Returns:
            Dictionary with risk attribution per asset and explanations
        """
        if "price_data" not in context.raw_data:
            raise ValueError("Price data not available. DataFetchAgent must run first.")
        
        price_data = context.raw_data["price_data"]
        weights = context.weights
        
        # Calculate returns for each asset
        returns_dict = self._calculate_returns(price_data)
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(returns_dict, weights)
        
        # Calculate risk metrics
        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        portfolio_variance = portfolio_volatility ** 2
        
        # Compute risk attribution for each asset
        attributions = []
        
        for ticker in context.tickers:
            attribution = self._compute_risk_attribution(
                ticker, returns_dict, weights, portfolio_returns, portfolio_variance
            )
            attributions.append(attribution.dict())
        
        # Normalize risk contributions to sum to 100%
        total_contribution = sum(a["risk_contribution"] for a in attributions)
        if total_contribution > 0:
            for attr in attributions:
                attr["risk_contribution"] = (attr["risk_contribution"] / total_contribution) * 100.0
        
        return {
            "risk_attributions": attributions,
            "portfolio_volatility": float(portfolio_volatility),
            "total_risk_contribution": sum(a["risk_contribution"] for a in attributions)
        }
    
    def _calculate_returns(self, price_data: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Calculate daily returns for each asset.
        
        Args:
            price_data: Price time series per ticker
            
        Returns:
            Daily returns per ticker
        """
        returns_dict = {}
        
        for ticker, prices in price_data.items():
            if len(prices) < 2:
                returns_dict[ticker] = [0.0]
                continue
            
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] is not None and prices[i] is not None and prices[i-1] != 0:
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(ret)
                else:
                    returns.append(0.0)
            
            returns_dict[ticker] = returns
        
        return returns_dict
    
    def _calculate_portfolio_returns(self, returns_dict: Dict[str, List[float]], 
                                     weights: Dict[str, float]) -> np.ndarray:
        """
        Calculate portfolio returns as weighted sum of asset returns.
        
        Args:
            returns_dict: Returns per ticker
            weights: Portfolio weights
            
        Returns:
            Portfolio return time series
        """
        # Align all return series to same length
        min_length = min(len(returns) for returns in returns_dict.values()) if returns_dict else 0
        
        if min_length == 0:
            return np.array([])
        
        portfolio_returns = np.zeros(min_length)
        
        for ticker, returns in returns_dict.items():
            weight = weights.get(ticker, 0.0)
            aligned_returns = returns[:min_length]
            portfolio_returns += weight * np.array(aligned_returns)
        
        return portfolio_returns
    
    def _compute_risk_attribution(self, ticker: str,
                                 returns_dict: Dict[str, List[float]],
                                 weights: Dict[str, float],
                                 portfolio_returns: np.ndarray,
                                 portfolio_variance: float) -> RiskAttribution:
        """
        Compute risk attribution for a single asset using variance decomposition.
        
        Uses the formula:
        Risk Contribution = weight_i * (Covariance(i, portfolio) / Portfolio Variance)
        
        This is a proxy for SHAP-style attribution in the risk context.
        
        Args:
            ticker: Asset ticker
            returns_dict: Returns per ticker
            weights: Portfolio weights
            portfolio_returns: Portfolio return time series
            portfolio_variance: Portfolio variance
            
        Returns:
            RiskAttribution object
        """
        if ticker not in returns_dict or len(returns_dict[ticker]) == 0:
            return RiskAttribution(
                ticker=ticker,
                risk_contribution=0.0,
                marginal_contribution=0.0,
                explanation=f"{ticker} has no return data available."
            )
        
        asset_returns = np.array(returns_dict[ticker])
        weight = weights.get(ticker, 0.0)
        
        # Align lengths
        min_length = min(len(asset_returns), len(portfolio_returns))
        asset_returns = asset_returns[:min_length]
        portfolio_returns_aligned = portfolio_returns[:min_length]
        
        if min_length < 2 or portfolio_variance == 0:
            return RiskAttribution(
                ticker=ticker,
                risk_contribution=0.0,
                marginal_contribution=0.0,
                explanation=f"{ticker} has insufficient data for risk attribution."
            )
        
        # Calculate covariance between asset and portfolio
        covariance = np.cov(asset_returns, portfolio_returns_aligned)[0, 1]
        
        # Risk contribution (component contribution to portfolio variance)
        risk_contribution_pct = (weight * covariance / portfolio_variance) * 100.0 if portfolio_variance > 0 else 0.0
        
        # Marginal contribution (sensitivity of portfolio risk to this asset)
        # Approximated as the derivative of portfolio variance w.r.t. weight
        marginal_contribution = 2 * weight * covariance if portfolio_variance > 0 else 0.0
        
        # Generate explanation
        explanation = self._generate_explanation(
            ticker, weight, risk_contribution_pct, covariance, portfolio_variance
        )
        
        return RiskAttribution(
            ticker=ticker,
            risk_contribution=abs(risk_contribution_pct),  # Use absolute value for contribution
            marginal_contribution=marginal_contribution,
            explanation=explanation
        )
    
    def _generate_explanation(self, ticker: str, weight: float, 
                             risk_contribution: float, covariance: float,
                             portfolio_variance: float) -> str:
        """
        Generate human-readable explanation of asset's risk role.
        
        Args:
            ticker: Asset ticker
            weight: Portfolio weight
            risk_contribution: Risk contribution percentage
            covariance: Covariance with portfolio
            portfolio_variance: Portfolio variance
            
        Returns:
            Human-readable explanation string
        """
        # Determine risk characteristics
        is_high_contributor = risk_contribution > 20.0
        is_diversifying = covariance < 0
        is_high_weight = weight > 0.3
        
        explanation_parts = []
        
        # Weight analysis
        if is_high_weight:
            explanation_parts.append(
                f"{ticker} has a high portfolio weight ({weight*100:.1f}%)"
            )
        else:
            explanation_parts.append(
                f"{ticker} has a portfolio weight of {weight*100:.1f}%"
            )
        
        # Risk contribution
        if is_high_contributor:
            explanation_parts.append(
                f"and contributes {risk_contribution:.1f}% to portfolio risk"
            )
        else:
            explanation_parts.append(
                f"contributing {risk_contribution:.1f}% to portfolio risk"
            )
        
        # Diversification effect
        if is_diversifying:
            explanation_parts.append(
                "with a diversifying effect (negative correlation with portfolio)"
            )
        elif covariance > 0:
            explanation_parts.append(
                "with positive correlation to the portfolio"
            )
        
        # Combine
        explanation = ", ".join(explanation_parts) + "."
        
        return explanation
