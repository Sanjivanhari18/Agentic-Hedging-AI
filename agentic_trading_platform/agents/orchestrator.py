"""Main router agent that orchestrates sub-agents."""

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, SystemMessage
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    ChatGroq = None
    HumanMessage = None
    SystemMessage = None

from .behavioral_agent import BehavioralInsightsAgent
from .market_agent import MarketAnalysisAgent
from .recommendation_agent import StockRecommenderAgent
from .tools.analysis_tools import set_analysis_results


# Query classification: behavioral, market, recommendation, general
BEHAVIORAL_KEYWORDS = [
    "my trades", "trading history", "pattern", "patterns", "cluster", "clusters",
    "behavioral", "behaviour", "stability", "anomal", "outlier", "win rate",
    "pnl", "profit", "loss", "holding duration", "position size", "baseline",
    "counterfactual", "what-if", "probability", "credible",
]
MARKET_KEYWORDS = [
    "market", "price", "quote", "rsi", "macd", "ema", "technical", "indicator",
    "s&p", "vix", "sector", "overview", "chart", "stock price", "current price",
]
RECOMMENDATION_KEYWORDS = [
    "recommend", "suggest", "stock to buy", "stocks to buy", "pick", "choose",
    "sector", "risk", "budget", "horizon", "investment", "portfolio idea",
]


def _keyword_classify(query: str) -> str:
    """Classify query using keyword matching."""
    q = query.lower().strip()
    if not q:
        return "general"
    b = sum(1 for k in BEHAVIORAL_KEYWORDS if k in q)
    m = sum(1 for k in MARKET_KEYWORDS if k in q)
    r = sum(1 for k in RECOMMENDATION_KEYWORDS if k in q)
    if b > m and b >= r:
        return "behavioral"
    if r > m and r >= b:
        return "recommendation"
    if m >= b and m >= r:
        return "market"
    return "general"


class AgentOrchestrator:
    """Routes user queries to appropriate specialized agents."""

    def __init__(self, analysis_results: Optional[Dict] = None, groq_api_key: Optional[str] = None):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.analysis_results = analysis_results or {}
        self.behavioral_agent: Optional[BehavioralInsightsAgent] = None
        self.market_agent: Optional[MarketAnalysisAgent] = None
        self.recommend_agent: Optional[StockRecommenderAgent] = None
        self.chat_history: List[Dict[str, str]] = []
        self._llm = None
        self._initialize_agents()

    def _initialize_agents(self) -> None:
        """Initialize sub-agents with shared API key."""
        try:
            self.behavioral_agent = BehavioralInsightsAgent(
                analysis_results=self.analysis_results,
                api_key=self.api_key,
            )
            self.market_agent = MarketAnalysisAgent(api_key=self.api_key)
            self.recommend_agent = StockRecommenderAgent(
                analysis_results=self.analysis_results,
                api_key=self.api_key,
            )
            if self.api_key and HAS_LANGCHAIN:
                try:
                    self._llm = ChatGroq(
                        model="llama-3.1-70b-versatile",
                        api_key=self.api_key,
                        temperature=0.1,
                    )
                except Exception as e:
                    logger.warning("Orchestrator LLM not available: %s", e)
        except Exception as e:
            logger.exception("Failed to initialize agents: %s", e)
            if self.behavioral_agent is None:
                self.behavioral_agent = BehavioralInsightsAgent(
                    analysis_results=self.analysis_results,
                    api_key=self.api_key,
                )
            if self.market_agent is None:
                self.market_agent = MarketAnalysisAgent(api_key=self.api_key)
            if self.recommend_agent is None:
                self.recommend_agent = StockRecommenderAgent(
                    analysis_results=self.analysis_results,
                    api_key=self.api_key,
                )

    def _classify_query(self, query: str) -> str:
        """Classify query into: behavioral, market, recommendation, general."""
        label = _keyword_classify(query)
        if label != "general" or not self._llm:
            return label
        try:
            sys_msg = (
                "Classify the user message into exactly one of: behavioral, market, recommendation, general. "
                "behavioral = about their own trading history, patterns, clusters, stability, PnL. "
                "market = current market data, prices, RSI, MACD, sectors, indices. "
                "recommendation = asking for stock picks, suggestions, what to buy. "
                "Reply with only one word: behavioral, market, recommendation, or general."
            )
            messages = [
                SystemMessage(content=sys_msg),
                HumanMessage(content=query[:500]),
            ]
            resp = self._llm.invoke(messages)
            if hasattr(resp, "content") and resp.content:
                raw = resp.content.strip().lower()
                for choice in ("behavioral", "market", "recommendation", "general"):
                    if choice in raw:
                        return choice
        except Exception as e:
            logger.warning("LLM classification failed: %s", e)
        return label

    def process_query(self, query: str) -> str:
        """Route to appropriate agent and return response."""
        if not query or not query.strip():
            return "Please enter a question about your trading behavior, market data, or stock recommendations."

        set_analysis_results(self.analysis_results)
        self.chat_history.append({"role": "user", "content": query})

        label = self._classify_query(query)
        response = ""

        try:
            if label == "behavioral" and self.behavioral_agent:
                response = self.behavioral_agent.answer(query)
            elif label == "market" and self.market_agent:
                response = self.market_agent.answer(query)
            elif label == "recommendation" and self.recommend_agent:
                response = self.recommend_agent.answer(query)
            else:
                response = self._general_response(query, label)
        except Exception as e:
            logger.exception("Agent processing failed: %s", e)
            response = (
                "Sorry, I encountered an error processing your request. "
                "You can try rephrasing or ask about behavioral insights, market data, or stock recommendations."
            )

        if not response:
            response = self._general_response(query, label)
        self.chat_history.append({"role": "assistant", "content": response})
        return response

    def _general_response(self, query: str, label: str) -> str:
        """Fallback when no specialized agent handles the query."""
        if self._llm:
            try:
                sys_msg = (
                    "You are a helpful assistant for an Agentic Trading Platform. "
                    "Answer briefly. If the question is about trading analysis, market data, or "
                    "recommendations, suggest they try asking more specifically (e.g. 'What are my trading patterns?', "
                    "'What is AAPL price?', 'Suggest low-risk stocks')."
                )
                messages = [
                    SystemMessage(content=sys_msg),
                    HumanMessage(content=query[:800]),
                ]
                resp = self._llm.invoke(messages)
                if hasattr(resp, "content") and resp.content:
                    return resp.content.strip()
            except Exception as e:
                logger.warning("General LLM fallback failed: %s", e)
        return (
            "I can help with: (1) Your trading behavior and patterns — e.g. 'What are my clusters?' "
            "(2) Market data — e.g. 'What is the price of AAPL?' or 'Market overview' "
            "(3) Stock recommendations — e.g. 'Suggest low-risk technology stocks'. "
            "Ask a specific question in one of these areas."
        )

    def update_analysis_results(self, results: Dict) -> None:
        """Update all agents with new analysis results."""
        self.analysis_results = results or {}
        set_analysis_results(self.analysis_results)
        if self.behavioral_agent:
            self.behavioral_agent.update_results(self.analysis_results)
        if self.recommend_agent:
            self.recommend_agent.analysis_results = self.analysis_results
