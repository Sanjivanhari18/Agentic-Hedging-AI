"""
Chat API endpoints - connected to the AgentOrchestrator.

POST /chat - accept user message, route to appropriate agent, return response.
GET  /chat/history - return conversation history.
DELETE /chat/history - clear conversation history.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# In-memory store for the orchestrator and chat history (single session)
_orchestrator = None
_analysis_results: Optional[Dict] = None


class ChatRequest(BaseModel):
    """Incoming chat message."""
    message: str


class ChatResponse(BaseModel):
    """Chat response from the agent system."""
    response: str
    agent_used: str = "general"
    status: str = "ok"


class ChatHistoryResponse(BaseModel):
    """Full conversation history."""
    history: List[Dict[str, str]]
    count: int


def _get_orchestrator():
    """Get or create the AgentOrchestrator singleton."""
    global _orchestrator
    if _orchestrator is not None:
        return _orchestrator
    try:
        from agents.orchestrator import AgentOrchestrator
        api_key = os.getenv("GROQ_API_KEY", "")
        _orchestrator = AgentOrchestrator(
            analysis_results=_analysis_results,
            groq_api_key=api_key,
        )
        return _orchestrator
    except Exception as e:
        logger.warning("Could not initialize AgentOrchestrator: %s", e)
        return None


def update_chat_analysis_results(results: Dict) -> None:
    """Update the orchestrator with new analysis results (called after analysis)."""
    global _analysis_results
    _analysis_results = results
    if _orchestrator is not None:
        try:
            _orchestrator.update_analysis_results(results)
        except Exception as e:
            logger.warning("Failed to update orchestrator results: %s", e)


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat message through the AI agent system.

    Routes the query to the appropriate agent:
    - Behavioral Insights Agent: questions about trading patterns, clusters, PnL
    - Market Analysis Agent: market data, prices, technical indicators
    - Stock Recommender Agent: stock picks and suggestions
    """
    if not request.message or not request.message.strip():
        return ChatResponse(
            response="Please enter a question about your trading behavior, market data, or stock recommendations.",
            agent_used="none",
            status="ok",
        )

    orchestrator = _get_orchestrator()
    if orchestrator is None:
        return ChatResponse(
            response=(
                "The AI agent system is not available. Ensure langchain-groq and chromadb "
                "are installed, and GROQ_API_KEY is set in your environment."
            ),
            agent_used="none",
            status="error",
        )

    try:
        # Classify query to determine which agent will handle it
        agent_used = orchestrator._classify_query(request.message)
        response_text = orchestrator.process_query(request.message)
        return ChatResponse(
            response=response_text,
            agent_used=agent_used,
            status="ok",
        )
    except Exception as e:
        logger.exception("Chat processing failed: %s", e)
        return ChatResponse(
            response=f"Sorry, I encountered an error: {e}",
            agent_used="error",
            status="error",
        )


@router.get("/history", response_model=ChatHistoryResponse)
def get_chat_history() -> ChatHistoryResponse:
    """Return the full conversation history."""
    orchestrator = _get_orchestrator()
    if orchestrator is not None:
        history = orchestrator.chat_history
    else:
        history = []
    return ChatHistoryResponse(history=history, count=len(history))


@router.delete("/history")
def clear_chat_history() -> dict:
    """Clear the conversation history."""
    orchestrator = _get_orchestrator()
    if orchestrator is not None:
        orchestrator.chat_history.clear()
    return {"status": "ok", "message": "Chat history cleared."}
