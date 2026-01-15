"""Base agent interface and abstract implementation."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import time
from app.models.agent import AgentType, AgentStatus, AgentContext, AgentOutput


class BaseAgent(ABC):
    """
    Base class for all agents in the system.
    
    Each agent must implement the execute method and define its agent_type.
    Agents follow a single-responsibility principle and communicate through
    the shared AgentContext.
    """
    
    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        pass
    
    @abstractmethod
    def execute(self, context: AgentContext) -> Dict[str, Any]:
        """
        Execute the agent's core logic.
        
        Args:
            context: Shared context containing portfolio info and previous agent outputs
            
        Returns:
            Dictionary containing agent-specific output data
            
        Raises:
            Exception: If execution fails (will be caught and wrapped in AgentOutput)
        """
        pass
    
    def run(self, context: AgentContext) -> AgentOutput:
        """
        Wrapper method that executes the agent and returns standardized output.
        
        This method handles timing, error catching, and output formatting.
        Individual agents should focus on implementing execute().
        
        Args:
            context: Shared context for agent execution
            
        Returns:
            AgentOutput with status, execution time, and data/error
        """
        start_time = time.time()
        status = AgentStatus.RUNNING
        data = {}
        error = None
        metadata = {}
        
        try:
            data = self.execute(context)
            status = AgentStatus.COMPLETED
            metadata = self._get_metadata(context, data)
        except Exception as e:
            status = AgentStatus.FAILED
            error = str(e)
            # Log error details in metadata for debugging
            metadata = {
                "error_type": type(e).__name__,
                "error_details": str(e)
            }
        
        execution_time = time.time() - start_time
        
        return AgentOutput(
            agent_type=self.agent_type,
            status=status,
            execution_time_seconds=execution_time,
            data=data,
            error=error,
            metadata=metadata
        )
    
    def _get_metadata(self, context: AgentContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from execution.
        
        Override in subclasses to add agent-specific metadata.
        
        Args:
            context: Execution context
            data: Output data from execute()
            
        Returns:
            Metadata dictionary
        """
        return {
            "agent_type": self.agent_type.value,
            "tickers_processed": len(context.tickers)
        }
