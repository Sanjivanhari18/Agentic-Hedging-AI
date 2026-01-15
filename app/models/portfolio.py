"""Portfolio data models."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime


class PortfolioInput(BaseModel):
    """Input model for portfolio analysis request."""
    
    tickers: List[str] = Field(
        ..., 
        description="List of stock ticker symbols",
        min_items=1,
        example=["AAPL", "GOOGL", "MSFT"]
    )
    weights: List[float] = Field(
        ..., 
        description="Portfolio weights (must sum to 1.0)",
        min_items=1
    )
    analysis_date: Optional[datetime] = Field(
        None,
        description="Reference date for analysis (defaults to current date)"
    )
    
    @field_validator('weights')
    @classmethod
    def weights_must_sum_to_one(cls, v):
        """Validate that weights sum to approximately 1.0."""
        total = sum(v)
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        if len(v) != len(set(v)) and len(v) > 1:
            # Allow duplicate weights but warn if suspicious
            pass
        return v
    
    @field_validator('tickers')
    @classmethod
    def tickers_must_be_unique(cls, v):
        """Ensure tickers are unique."""
        if len(v) != len(set(v)):
            raise ValueError("Tickers must be unique")
        return [t.upper().strip() for t in v]
    
    @model_validator(mode='after')
    def validate_tickers_weights_length(self):
        """Ensure tickers and weights have the same length."""
        if len(self.tickers) != len(self.weights):
            raise ValueError(
                f"Tickers ({len(self.tickers)}) and weights ({len(self.weights)}) must have the same length"
            )
        return self
    
    class Config:
        schema_extra = {
            "example": {
                "tickers": ["AAPL", "GOOGL", "MSFT"],
                "weights": [0.4, 0.3, 0.3],
                "analysis_date": "2024-01-15T00:00:00Z"
            }
        }


class Portfolio(BaseModel):
    """Internal portfolio representation with normalized data."""
    
    tickers: List[str]
    weights: Dict[str, float] = Field(
        ...,
        description="Mapping of ticker to weight"
    )
    analysis_date: datetime
    price_data: Dict[str, List[float]] = Field(
        ...,
        description="Historical price data per ticker (normalized time series)"
    )
    dates: List[datetime] = Field(
        ...,
        description="Aligned date index for all time series"
    )
    
    @model_validator(mode='after')
    def validate_weights(self):
        """Ensure weights match tickers."""
        ticker_set = set(self.tickers)
        weight_set = set(self.weights.keys())
        if ticker_set != weight_set:
            missing = ticker_set - weight_set
            extra = weight_set - ticker_set
            raise ValueError(
                f"Weights don't match tickers. Missing: {missing}, Extra: {extra}"
            )
        return self
    
    class Config:
        schema_extra = {
            "example": {
                "tickers": ["AAPL", "GOOGL"],
                "weights": {"AAPL": 0.6, "GOOGL": 0.4},
                "analysis_date": "2024-01-15T00:00:00Z",
                "price_data": {
                    "AAPL": [150.0, 151.0, 152.0],
                    "GOOGL": [2500.0, 2510.0, 2520.0]
                },
                "dates": ["2024-01-01", "2024-01-02", "2024-01-03"]
            }
        }
