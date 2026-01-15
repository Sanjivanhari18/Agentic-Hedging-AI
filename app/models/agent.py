"""Agent interface and output models."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class AgentType(str, Enum):
    """Types of agents in the system."""
    DATA_FETCH = "data_fetch"
    STRESS_TEST = "stress_test"
    EXPLAINABILITY = "explainability"
    RECOMMENDATION = "recommendation"


class AgentStatus(str, Enum):
    """Agent execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentContext(BaseModel):
    """Shared context passed between agents."""
    
    portfolio_id: str = Field(
        ...,
        description="Unique identifier for this analysis run"
    )
    tickers: list[str]
    weights: Dict[str, float]
    analysis_date: datetime
    raw_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Raw data cache for agent outputs"
    )
    
    class Config:
        arbitrary_types_allowed = True


class AgentOutput(BaseModel):
    """Standard output format for all agents."""
    
    agent_type: AgentType
    status: AgentStatus
    execution_time_seconds: float = Field(
        ...,
        description="Time taken to execute",
        ge=0.0
    )
    data: Dict[str, Any] = Field(
        ...,
        description="Agent-specific output data"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if status is FAILED"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about execution"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "agent_type": "data_fetch",
                "status": "completed",
                "execution_time_seconds": 2.5,
                "data": {
                    "price_data": {"AAPL": [150.0, 151.0]},
                    "dates": ["2024-01-01", "2024-01-02"]
                },
                "metadata": {"data_points": 252, "missing_data_handled": True}
            }
        }
