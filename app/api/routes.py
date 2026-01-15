"""API routes for portfolio risk analysis."""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
from app.models.portfolio import PortfolioInput
from app.coordinator.coordinator import AICoordinator
from app.risk_engine.engine import RiskEngine

router = APIRouter()

# Initialize coordinator and risk engine (in production, use dependency injection)
coordinator = AICoordinator()
risk_engine = RiskEngine()

# In-memory storage for reports (TODO: Replace with database)
reports_store: Dict[str, Dict[str, Any]] = {}


@router.post("/portfolio/analyze", status_code=status.HTTP_201_CREATED)
async def analyze_portfolio(portfolio_input: PortfolioInput) -> Dict[str, Any]:
    """
    Analyze portfolio risk using agent-based architecture.
    
    This endpoint:
    1. Orchestrates all agents (DataFetch, StressTest, Explainability, Recommendation)
    2. Computes risk metrics (VaR, CVaR, Sharpe, Max Drawdown)
    3. Returns comprehensive risk analysis report
    
    Args:
        portfolio_input: Portfolio specification with tickers and weights
        
    Returns:
        Complete risk analysis report with:
        - portfolio_id: Unique identifier for this analysis
        - risk_metrics: Core risk metrics
        - stress_test_results: Historical stress scenario results
        - risk_attributions: Per-asset risk contributions
        - structural_insights: Concentration and diversification analysis
        - execution_summary: Agent execution metadata
    """
    try:
        # Run coordinator to execute all agents
        coordinator_result = coordinator.analyze_portfolio(portfolio_input)
        
        if "error" in coordinator_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=coordinator_result["error"]
            )
        
        portfolio_id = coordinator_result["portfolio_id"]
        
        # Build portfolio object for risk engine
        from app.models.portfolio import Portfolio
        portfolio_dict = coordinator_result["portfolio"]
        portfolio = Portfolio(**portfolio_dict)
        
        # Extract portfolio values from stress test agent output
        stress_output = coordinator_result["agent_outputs"].get("stress_test", {})
        portfolio_values = stress_output.get("data", {}).get("portfolio_value_timeseries")
        
        # Compute risk metrics
        risk_metrics = risk_engine.compute_risk_metrics(
            portfolio, 
            portfolio_values=portfolio_values
        )
        
        # Build comprehensive report
        report = {
            "portfolio_id": portfolio_id,
            "portfolio": {
                "tickers": portfolio.tickers,
                "weights": portfolio.weights,
                "analysis_date": portfolio.analysis_date.isoformat()
            },
            "risk_metrics": risk_metrics.dict(),
            "stress_test_results": coordinator_result["agent_outputs"].get(
                "stress_test", {}
            ).get("data", {}).get("stress_test_results", {}),
            "risk_attributions": coordinator_result["agent_outputs"].get(
                "explainability", {}
            ).get("data", {}).get("risk_attributions", []),
            "structural_insights": coordinator_result["agent_outputs"].get(
                "recommendation", {}
            ).get("data", {}).get("structural_risk", {}),
            "execution_summary": coordinator_result["execution_summary"]
        }
        
        # Store report (TODO: Replace with database)
        reports_store[portfolio_id] = report
        
        return report
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/portfolio/report/{portfolio_id}")
async def get_report(portfolio_id: str) -> Dict[str, Any]:
    """
    Retrieve a previously generated risk analysis report.
    
    Args:
        portfolio_id: Unique identifier from analyze endpoint
        
    Returns:
        Stored risk analysis report
        
    Raises:
        404: If report not found
    """
    if portfolio_id not in reports_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report with ID {portfolio_id} not found"
        )
    
    return reports_store[portfolio_id]
