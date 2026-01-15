"""AI Coordinator: Orchestrates agent execution and aggregates outputs."""

from typing import Dict, Any, List
import uuid
from datetime import datetime
from app.models.agent import AgentContext, AgentOutput, AgentType
from app.models.portfolio import Portfolio, PortfolioInput
from app.agents.data_fetch import DataFetchAgent
from app.agents.stress_test import StressTestAgent
from app.agents.explainability import ExplainabilityAgent
from app.agents.recommendation import RecommendationAgent


class AICoordinator:
    """
    Central orchestrator for agent-based portfolio risk analysis.
    
    Responsibilities:
    - Manage agent execution order
    - Pass shared context between agents
    - Aggregate agent outputs
    - Prepare unified input for risk engine
    - Handle errors and partial failures
    
    Design Pattern: Coordinator pattern with agent pipeline
    """
    
    def __init__(self):
        """Initialize coordinator with all agents."""
        self.data_fetch_agent = DataFetchAgent()
        self.stress_test_agent = StressTestAgent()
        self.explainability_agent = ExplainabilityAgent()
        self.recommendation_agent = RecommendationAgent()
        
        # Agent execution order (must be respected)
        self.agent_order = [
            AgentType.DATA_FETCH,
            AgentType.STRESS_TEST,
            AgentType.EXPLAINABILITY,
            AgentType.RECOMMENDATION
        ]
    
    def analyze_portfolio(self, portfolio_input: PortfolioInput) -> Dict[str, Any]:
        """
        Main entry point for portfolio analysis.
        
        Orchestrates all agents and returns aggregated results.
        
        Args:
            portfolio_input: Portfolio specification (tickers, weights, date)
            
        Returns:
            Dictionary containing:
            - portfolio_id: Unique analysis ID
            - agent_outputs: Results from each agent
            - portfolio: Normalized portfolio object
            - execution_summary: Metadata about execution
        """
        # Generate unique analysis ID
        portfolio_id = str(uuid.uuid4())
        
        # Set analysis date
        analysis_date = portfolio_input.analysis_date or datetime.now()
        
        # Create initial context
        context = AgentContext(
            portfolio_id=portfolio_id,
            tickers=portfolio_input.tickers,
            weights={ticker: weight for ticker, weight in 
                    zip(portfolio_input.tickers, portfolio_input.weights)},
            analysis_date=analysis_date,
            raw_data={}
        )
        
        # Execute agents in order
        agent_outputs = {}
        execution_summary = {
            "portfolio_id": portfolio_id,
            "analysis_date": analysis_date.isoformat(),
            "agents_executed": [],
            "agents_failed": [],
            "total_execution_time": 0.0
        }
        
        import time
        total_start_time = time.time()
        
        # 1. Data Fetch Agent (must run first)
        data_output = self.data_fetch_agent.run(context)
        agent_outputs[AgentType.DATA_FETCH.value] = data_output.dict()
        execution_summary["agents_executed"].append(AgentType.DATA_FETCH.value)
        
        if data_output.status.value == "failed":
            execution_summary["agents_failed"].append(AgentType.DATA_FETCH.value)
            # Cannot proceed without data
            return {
                "portfolio_id": portfolio_id,
                "agent_outputs": agent_outputs,
                "execution_summary": execution_summary,
                "error": "Data fetch failed. Cannot proceed with analysis."
            }
        
        # Update context with data fetch results
        context.raw_data.update(data_output.data)
        
        # Build normalized Portfolio object
        try:
            portfolio = self._build_portfolio(portfolio_input, data_output.data)
        except Exception as e:
            return {
                "portfolio_id": portfolio_id,
                "agent_outputs": agent_outputs,
                "execution_summary": execution_summary,
                "error": f"Failed to build portfolio object: {str(e)}"
            }
        
        # 2. Stress Test Agent
        stress_output = self.stress_test_agent.run(context)
        agent_outputs[AgentType.STRESS_TEST.value] = stress_output.dict()
        execution_summary["agents_executed"].append(AgentType.STRESS_TEST.value)
        if stress_output.status.value == "failed":
            execution_summary["agents_failed"].append(AgentType.STRESS_TEST.value)
        else:
            context.raw_data.update(stress_output.data)
        
        # 3. Explainability Agent
        explain_output = self.explainability_agent.run(context)
        agent_outputs[AgentType.EXPLAINABILITY.value] = explain_output.dict()
        execution_summary["agents_executed"].append(AgentType.EXPLAINABILITY.value)
        if explain_output.status.value == "failed":
            execution_summary["agents_failed"].append(AgentType.EXPLAINABILITY.value)
        else:
            context.raw_data.update(explain_output.data)
        
        # 4. Recommendation Agent
        recommendation_output = self.recommendation_agent.run(context)
        agent_outputs[AgentType.RECOMMENDATION.value] = recommendation_output.dict()
        execution_summary["agents_executed"].append(AgentType.RECOMMENDATION.value)
        if recommendation_output.status.value == "failed":
            execution_summary["agents_failed"].append(AgentType.RECOMMENDATION.value)
        else:
            context.raw_data.update(recommendation_output.data)
        
        total_execution_time = time.time() - total_start_time
        execution_summary["total_execution_time"] = total_execution_time
        
        return {
            "portfolio_id": portfolio_id,
            "agent_outputs": agent_outputs,
            "portfolio": portfolio.dict(),
            "execution_summary": execution_summary
        }
    
    def _build_portfolio(self, portfolio_input: PortfolioInput, 
                        data_output: Dict[str, Any]) -> Portfolio:
        """
        Build normalized Portfolio object from input and data.
        
        Args:
            portfolio_input: Original input
            data_output: Output from DataFetchAgent
            
        Returns:
            Portfolio object
        """
        price_data = data_output.get("price_data", {})
        dates_str = data_output.get("dates", [])
        
        # Convert date strings back to datetime
        dates = []
        for d in dates_str:
            if isinstance(d, str):
                try:
                    dates.append(datetime.fromisoformat(d.replace('Z', '+00:00')))
                except:
                    dates.append(datetime.fromisoformat(d))
            else:
                dates.append(d)
        
        # Build weights dict
        weights = {
            ticker: weight for ticker, weight in 
            zip(portfolio_input.tickers, portfolio_input.weights)
        }
        
        return Portfolio(
            tickers=portfolio_input.tickers,
            weights=weights,
            analysis_date=portfolio_input.analysis_date or datetime.now(),
            price_data=price_data,
            dates=dates
        )
