"""
Recommendation API endpoints - connected to ML models.

POST /recommend - accept constraints, run stock screening + momentum scoring + risk classification.
POST /recommend/predict - predict trend for a specific symbol using LSTM.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recommendations"])


class RecommendRequest(BaseModel):
    """Constraints for generating recommendations."""
    sector: Optional[str] = Field(None, description="Sector filter (e.g. technology, healthcare, finance, energy)")
    risk_tolerance: Optional[str] = Field(None, description="low / medium / high")
    horizon: Optional[str] = Field(None, description="short (<1mo) / medium (1-6mo) / long (>6mo)")
    budget: Optional[float] = Field(None, ge=0, description="Budget or allocation amount")
    market: Optional[str] = Field("us", description="us or india")
    top_n: int = Field(10, ge=1, le=30, description="Number of recommendations to return")


class PredictRequest(BaseModel):
    """Request for LSTM trend prediction."""
    symbol: str = Field(..., description="Stock ticker symbol (e.g. AAPL, RELIANCE.NS)")


class RecommendResponse(BaseModel):
    """Stock recommendations response."""
    success: bool
    message: str
    constraints: Dict[str, Any]
    recommendations: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class PredictResponse(BaseModel):
    """LSTM trend prediction response."""
    symbol: str
    trend: Optional[str] = None
    prob_up: Optional[float] = None
    prob_down: Optional[float] = None
    prob_sideways: Optional[float] = None
    error: Optional[str] = None


@router.post("", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    """
    Generate stock recommendations using ML models.

    Pipeline:
    1. StockScreener selects universe by market (US/India) and sector
    2. RiskClassifier filters by risk tolerance (volatility, beta, drawdown)
    3. MomentumScorer ranks by technical momentum (RSI, MACD, EMA, ROC)
    4. Returns top N stocks sorted by composite score
    """
    constraints = request.model_dump(exclude_none=True)

    try:
        from models.ml.stock_screener import StockScreener
        from models.ml.momentum_scorer import MomentumScorer
        from models.ml.risk_classifier import RiskClassifier

        scorer = MomentumScorer()
        classifier = RiskClassifier()
        screener = StockScreener(momentum_scorer=scorer, risk_classifier=classifier)

        screener_constraints = {
            "sector": request.sector,
            "risk_tolerance": request.risk_tolerance,
            "horizon": request.horizon or "medium",
            "market": request.market or "us",
            "budget": request.budget,
        }

        results_df = screener.screen(screener_constraints)

        if results_df is not None and len(results_df) > 0:
            # Limit to top_n
            results_df = results_df.head(request.top_n)
            recommendations = results_df.to_dict(orient="records")
            return RecommendResponse(
                success=True,
                message=f"Found {len(recommendations)} recommendations matching your criteria.",
                constraints=constraints,
                recommendations=recommendations,
            )
        else:
            return RecommendResponse(
                success=True,
                message="No stocks matched your criteria. Try relaxing constraints.",
                constraints=constraints,
                recommendations=[],
            )
    except ImportError as e:
        logger.warning("ML model import failed: %s", e)
        return RecommendResponse(
            success=False,
            message="Recommendation engine not available.",
            constraints=constraints,
            error=f"Missing dependency: {e}",
        )
    except Exception as e:
        logger.exception("Recommendation failed: %s", e)
        return RecommendResponse(
            success=False,
            message="Recommendation engine error.",
            constraints=constraints,
            error=str(e),
        )


@router.post("/predict", response_model=PredictResponse)
def predict_trend(request: PredictRequest) -> PredictResponse:
    """
    Predict 5-day trend direction for a stock using LSTM model.

    Returns probabilities for Up, Down, and Sideways trends.
    Note: The model trains on-the-fly if no pre-trained weights are available.
    """
    try:
        from models.dl.lstm_predictor import LSTMTrendPredictor

        predictor = LSTMTrendPredictor()

        # Prepare data and do quick training if not pre-trained
        X, y = predictor.prepare_data(request.symbol, period="2y")
        if X.size == 0:
            return PredictResponse(
                symbol=request.symbol,
                error="Insufficient data to make a prediction.",
            )

        # Quick train on available data (in production, use pre-trained weights)
        if not predictor.is_trained:
            predictor.train(X, y, epochs=20, validation_split=0.2)

        result = predictor.predict_symbol(request.symbol)

        if result.get("error"):
            return PredictResponse(
                symbol=request.symbol,
                error=result["error"],
            )

        return PredictResponse(
            symbol=request.symbol,
            trend=result.get("trend"),
            prob_up=result.get("prob_up"),
            prob_down=result.get("prob_down"),
            prob_sideways=result.get("prob_sideways"),
        )
    except ImportError as e:
        return PredictResponse(
            symbol=request.symbol,
            error=f"TensorFlow not installed: {e}",
        )
    except Exception as e:
        logger.exception("Prediction failed for %s: %s", request.symbol, e)
        return PredictResponse(
            symbol=request.symbol,
            error=str(e),
        )
