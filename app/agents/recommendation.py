"""RecommendationAgent: Identifies structural risk and diversification issues."""

from typing import Dict, List, Any, Optional
import numpy as np
from app.agents.base import BaseAgent
from app.models.agent import AgentType, AgentContext
from app.models.risk import StructuralRiskInsight


class RecommendationAgent(BaseAgent):
    """
    Identifies concentration risk and diversification issues.
    
    Responsibilities:
    - Calculate concentration metrics (Herfindahl index, top N weights)
    - Compute correlation risk (average pairwise correlation)
    - Identify structural risk issues
    - Provide insights WITHOUT trading recommendations
    
    Note: This agent provides structural analysis only.
    It does NOT suggest buy/sell actions or predict future performance.
    """
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.RECOMMENDATION
    
    def execute(self, context: AgentContext) -> Dict[str, Any]:
        """
        Analyze structural risk characteristics of the portfolio.
        
        Args:
            context: Must contain price_data from DataFetchAgent
            
        Returns:
            Dictionary with structural risk insights
        """
        if "price_data" not in context.raw_data:
            raise ValueError("Price data not available. DataFetchAgent must run first.")
        
        weights = context.weights
        price_data = context.raw_data["price_data"]
        
        # Calculate concentration metrics
        concentration_risk = self._calculate_concentration_risk(weights)
        
        # Calculate correlation risk
        correlation_risk = self._calculate_correlation_risk(price_data)
        
        # Generate insights
        insights = self._generate_insights(concentration_risk, correlation_risk, weights)
        
        # Build structural risk insight
        structural_insight = StructuralRiskInsight(
            concentration_risk=concentration_risk,
            sector_diversification=None,  # TODO: Add sector mapping if available
            correlation_risk=correlation_risk,
            insights=insights
        )
        
        return {
            "structural_risk": structural_insight.dict(),
            "analysis_summary": {
                "total_holdings": len(weights),
                "concentration_level": self._classify_concentration(concentration_risk),
                "diversification_level": self._classify_diversification(correlation_risk)
            }
        }
    
    def _calculate_concentration_risk(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate concentration risk metrics.
        
        Metrics:
        - top_3_weight: Sum of top 3 holdings
        - top_5_weight: Sum of top 5 holdings
        - herfindahl_index: Sum of squared weights (0-1, higher = more concentrated)
        - max_weight: Largest single holding weight
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary of concentration metrics
        """
        if not weights:
            return {
                "top_3_weight": 0.0,
                "top_5_weight": 0.0,
                "herfindahl_index": 0.0,
                "max_weight": 0.0
            }
        
        sorted_weights = sorted(weights.values(), reverse=True)
        
        top_3_weight = sum(sorted_weights[:3])
        top_5_weight = sum(sorted_weights[:5])
        max_weight = sorted_weights[0] if sorted_weights else 0.0
        
        # Herfindahl-Hirschman Index (HHI)
        herfindahl_index = sum(w ** 2 for w in weights.values())
        
        return {
            "top_3_weight": top_3_weight,
            "top_5_weight": top_5_weight,
            "herfindahl_index": herfindahl_index,
            "max_weight": max_weight
        }
    
    def _calculate_correlation_risk(self, price_data: Dict[str, List[float]]) -> float:
        """
        Calculate average pairwise correlation between assets.
        
        Higher correlation = less diversification benefit.
        
        Args:
            price_data: Price time series per ticker
            
        Returns:
            Average pairwise correlation (0-1, where 1 = perfect correlation)
        """
        if len(price_data) < 2:
            return 0.0
        
        # Calculate returns for each asset
        returns_dict = {}
        for ticker, prices in price_data.items():
            if len(prices) < 2:
                continue
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] is not None and prices[i] is not None and prices[i-1] != 0:
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(ret)
            if returns:
                returns_dict[ticker] = returns
        
        if len(returns_dict) < 2:
            return 0.0
        
        # Align all return series to same length
        min_length = min(len(returns) for returns in returns_dict.values())
        if min_length < 2:
            return 0.0
        
        # Calculate pairwise correlations
        tickers = list(returns_dict.keys())
        correlations = []
        
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                ticker_i = tickers[i]
                ticker_j = tickers[j]
                
                returns_i = np.array(returns_dict[ticker_i][:min_length])
                returns_j = np.array(returns_dict[ticker_j][:min_length])
                
                # Calculate correlation
                if len(returns_i) > 1 and np.std(returns_i) > 0 and np.std(returns_j) > 0:
                    corr = np.corrcoef(returns_i, returns_j)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        # Return average correlation
        if correlations:
            return float(np.mean(correlations))
        else:
            return 0.0
    
    def _generate_insights(self, concentration_risk: Dict[str, float],
                          correlation_risk: float,
                          weights: Dict[str, float]) -> List[str]:
        """
        Generate human-readable structural risk insights.
        
        Args:
            concentration_risk: Concentration metrics
            correlation_risk: Average pairwise correlation
            weights: Portfolio weights
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Concentration insights
        top_3_weight = concentration_risk["top_3_weight"]
        herfindahl = concentration_risk["herfindahl_index"]
        max_weight = concentration_risk["max_weight"]
        
        if top_3_weight > 0.7:
            insights.append(
                f"Portfolio is highly concentrated in top 3 holdings ({top_3_weight*100:.1f}%)"
            )
        elif top_3_weight > 0.5:
            insights.append(
                f"Moderate concentration in top 3 holdings ({top_3_weight*100:.1f}%)"
            )
        
        if max_weight > 0.4:
            insights.append(
                f"Single largest holding represents {max_weight*100:.1f}% of portfolio (high single-stock risk)"
            )
        
        if herfindahl > 0.25:
            insights.append(
                f"Herfindahl index of {herfindahl:.2f} indicates high concentration"
            )
        elif herfindahl < 0.1 and len(weights) > 5:
            insights.append(
                f"Herfindahl index of {herfindahl:.2f} indicates good diversification"
            )
        
        # Correlation insights
        if correlation_risk > 0.7:
            insights.append(
                f"High average correlation ({correlation_risk:.2f}) suggests limited diversification benefit"
            )
        elif correlation_risk > 0.5:
            insights.append(
                f"Moderate correlation ({correlation_risk:.2f}) provides some diversification"
            )
        elif correlation_risk < 0.3:
            insights.append(
                f"Low correlation ({correlation_risk:.2f}) indicates good diversification potential"
            )
        
        # Combined insights
        if top_3_weight > 0.7 and correlation_risk > 0.6:
            insights.append(
                "Portfolio faces both concentration risk and high correlation, "
                "amplifying downside risk potential"
            )
        
        if not insights:
            insights.append("Portfolio shows balanced structural characteristics")
        
        return insights
    
    def _classify_concentration(self, concentration_risk: Dict[str, float]) -> str:
        """Classify concentration level."""
        herfindahl = concentration_risk["herfindahl_index"]
        if herfindahl > 0.25:
            return "high"
        elif herfindahl > 0.15:
            return "moderate"
        else:
            return "low"
    
    def _classify_diversification(self, correlation_risk: float) -> str:
        """Classify diversification level."""
        if correlation_risk > 0.7:
            return "low"
        elif correlation_risk > 0.4:
            return "moderate"
        else:
            return "high"
