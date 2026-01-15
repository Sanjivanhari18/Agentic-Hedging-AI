"""Risk metrics and analysis result models."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class StressScenario(str, Enum):
    """Historical stress test scenarios."""
    FINANCIAL_CRISIS_2008 = "financial_crisis_2008"
    COVID_19_CRASH = "covid_19_crash"
    CUSTOM_VOLATILITY_SHOCK = "custom_volatility_shock"


class RiskMetrics(BaseModel):
    """Core risk metrics for portfolio."""
    
    var_95: float = Field(
        ...,
        description="Value at Risk at 95% confidence level",
        ge=0.0
    )
    var_99: float = Field(
        ...,
        description="Value at Risk at 99% confidence level",
        ge=0.0
    )
    cvar_95: float = Field(
        ...,
        description="Conditional VaR (Expected Shortfall) at 95% confidence",
        ge=0.0
    )
    cvar_99: float = Field(
        ...,
        description="Conditional VaR (Expected Shortfall) at 99% confidence",
        ge=0.0
    )
    sharpe_ratio: float = Field(
        ...,
        description="Sharpe ratio (annualized, assuming risk-free rate = 0)"
    )
    max_drawdown: float = Field(
        ...,
        description="Maximum drawdown as percentage",
        ge=0.0,
        le=100.0
    )
    volatility: float = Field(
        ...,
        description="Annualized portfolio volatility",
        ge=0.0
    )
    portfolio_value_timeseries: List[float] = Field(
        ...,
        description="Historical portfolio value over time"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "var_95": 0.05,
                "var_99": 0.08,
                "cvar_95": 0.07,
                "cvar_99": 0.12,
                "sharpe_ratio": 1.2,
                "max_drawdown": 15.5,
                "volatility": 0.18,
                "portfolio_value_timeseries": [100.0, 102.0, 98.0, 105.0]
            }
        }


class StressTestResult(BaseModel):
    """Results from stress testing under historical scenarios."""
    
    scenario: StressScenario
    start_date: datetime
    end_date: datetime
    peak_drawdown: float = Field(
        ...,
        description="Maximum drawdown during stress period (%)",
        ge=0.0,
        le=100.0
    )
    recovery_days: Optional[int] = Field(
        None,
        description="Days to recover to pre-stress value (None if not recovered)",
        ge=0
    )
    volatility_during_stress: float = Field(
        ...,
        description="Volatility during stress period (annualized)",
        ge=0.0
    )
    portfolio_value_at_peak_loss: float = Field(
        ...,
        description="Portfolio value at maximum drawdown point"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "scenario": "financial_crisis_2008",
                "start_date": "2008-09-15T00:00:00Z",
                "end_date": "2009-03-09T00:00:00Z",
                "peak_drawdown": 45.2,
                "recovery_days": 1200,
                "volatility_during_stress": 0.35,
                "portfolio_value_at_peak_loss": 54.8
            }
        }


class RiskAttribution(BaseModel):
    """Risk contribution attribution per asset."""
    
    ticker: str
    risk_contribution: float = Field(
        ...,
        description="Percentage contribution to portfolio risk",
        ge=0.0,
        le=100.0
    )
    marginal_contribution: float = Field(
        ...,
        description="Marginal contribution to portfolio risk"
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of this asset's risk role"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "ticker": "AAPL",
                "risk_contribution": 35.5,
                "marginal_contribution": 0.12,
                "explanation": "AAPL contributes 35.5% to portfolio risk due to its high weight (40%) and correlation with other tech holdings."
            }
        }


class StructuralRiskInsight(BaseModel):
    """Structural risk insights without trading recommendations."""
    
    concentration_risk: Dict[str, float] = Field(
        ...,
        description="Concentration metrics (e.g., top_3_weight, herfindahl_index)"
    )
    sector_diversification: Optional[Dict[str, float]] = Field(
        None,
        description="Sector concentration if sector data available"
    )
    correlation_risk: float = Field(
        ...,
        description="Average pairwise correlation (higher = less diversification benefit)",
        ge=-1.0,
        le=1.0
    )
    insights: List[str] = Field(
        ...,
        description="Human-readable structural risk insights"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "concentration_risk": {
                    "top_3_weight": 0.85,
                    "herfindahl_index": 0.32
                },
                "correlation_risk": 0.72,
                "insights": [
                    "Portfolio is highly concentrated in top 3 holdings (85%)",
                    "High correlation (0.72) suggests limited diversification benefit"
                ]
            }
        }
