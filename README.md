# Portfolio Risk Intelligence Platform

A production-ready backend system for explainable, non-predictive portfolio risk analysis using an agent-based architecture.

## 🎯 Project Overview

This system analyzes the **structural and downside risk** of stock portfolios using multiple specialized AI agents. It provides:

- **Risk Metrics**: VaR, CVaR, Sharpe Ratio, Max Drawdown
- **Stress Testing**: Historical scenario analysis (2008 Crisis, COVID-19)
- **Risk Attribution**: Per-asset risk contribution with explanations
- **Structural Insights**: Concentration and diversification analysis

**Important**: This system does **NOT** predict prices or provide trading recommendations. It only analyzes historical risk characteristics.

## 🏗️ Architecture

### Agent-Oriented Modular Design

```
┌─────────────────────────────────────────┐
│         FastAPI Application            │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         AI Coordinator                  │
│    (Orchestrates Agent Execution)      │
└─────┬──────┬──────┬──────┬──────────────┘
      │      │      │      │
   ┌──▼──┐ ┌─▼──┐ ┌─▼──┐ ┌─▼──┐
   │Data │ │Str │ │Exp │ │Rec │
   │Fetch│ │ess │ │lain│ │omm │
   │Agent│ │Test│ │abil│ │enda│
   │     │ │    │ │ity │ │tion│
   └──┬──┘ └─┬──┘ └─┬──┘ └─┬──┘
      │      │      │      │
      └──────┴──────┴──────┘
             │
      ┌──────▼──────┐
      │ Risk Engine │
      │  (VaR, CVaR│
      │   Sharpe)  │
      └────────────┘
```

### Layer Separation

1. **Data Layer**: `app/models/` - Pydantic models for data validation
2. **Agent Layer**: `app/agents/` - Specialized agents with single responsibilities
3. **Risk Computation Layer**: `app/risk_engine/` - Core risk metrics calculation
4. **API/Delivery Layer**: `app/api/` - FastAPI routes and endpoints

## 📁 Project Structure

```
Agentic_Hedging_app/
├── app/
│   ├── __init__.py
│   ├── models/              # Data models (Pydantic)
│   │   ├── __init__.py
│   │   ├── portfolio.py     # Portfolio input/output models
│   │   ├── risk.py          # Risk metrics models
│   │   └── agent.py         # Agent interface models
│   ├── agents/              # Agent implementations
│   │   ├── __init__.py
│   │   ├── base.py          # Base agent interface
│   │   ├── data_fetch.py    # DataFetchAgent
│   │   ├── stress_test.py   # StressTestAgent
│   │   ├── explainability.py # ExplainabilityAgent
│   │   └── recommendation.py # RecommendationAgent
│   ├── coordinator/         # AI Coordinator
│   │   ├── __init__.py
│   │   └── coordinator.py   # Orchestration logic
│   ├── risk_engine/         # Risk computation
│   │   ├── __init__.py
│   │   └── engine.py        # Risk metrics calculation
│   └── api/                 # FastAPI application
│       ├── __init__.py
│       ├── main.py          # FastAPI app setup
│       └── routes.py        # API endpoints
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the API

```bash
# Start FastAPI server
uvicorn app.api.main:app --reload --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### 1. Analyze Portfolio

**POST** `/api/v1/portfolio/analyze`

Analyze portfolio risk using all agents.

**Request Body:**
```json
{
  "tickers": ["AAPL", "GOOGL", "MSFT"],
  "weights": [0.4, 0.3, 0.3],
  "analysis_date": "2024-01-15T00:00:00Z"
}
```

**Response:**
```json
{
  "portfolio_id": "uuid-here",
  "portfolio": {
    "tickers": ["AAPL", "GOOGL", "MSFT"],
    "weights": {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3},
    "analysis_date": "2024-01-15T00:00:00Z"
  },
  "risk_metrics": {
    "var_95": 0.05,
    "var_99": 0.08,
    "cvar_95": 0.07,
    "cvar_99": 0.12,
    "sharpe_ratio": 1.2,
    "max_drawdown": 15.5,
    "volatility": 0.18,
    "portfolio_value_timeseries": [100.0, 102.0, ...]
  },
  "stress_test_results": {
    "financial_crisis_2008": {
      "peak_drawdown": 45.2,
      "recovery_days": 1200,
      "volatility_during_stress": 0.35
    },
    "covid_19_crash": {
      "peak_drawdown": 32.1,
      "recovery_days": 150,
      "volatility_during_stress": 0.28
    }
  },
  "risk_attributions": [
    {
      "ticker": "AAPL",
      "risk_contribution": 35.5,
      "marginal_contribution": 0.12,
      "explanation": "AAPL contributes 35.5% to portfolio risk..."
    }
  ],
  "structural_insights": {
    "concentration_risk": {
      "top_3_weight": 0.85,
      "herfindahl_index": 0.32
    },
    "correlation_risk": 0.72,
    "insights": [
      "Portfolio is highly concentrated in top 3 holdings (85%)"
    ]
  },
  "execution_summary": {
    "agents_executed": ["data_fetch", "stress_test", "explainability", "recommendation"],
    "total_execution_time": 3.5
  }
}
```

### 2. Get Report

**GET** `/api/v1/portfolio/report/{portfolio_id}`

Retrieve a previously generated analysis report.

## 🤖 Agents

### 1. DataFetchAgent
- **Responsibility**: Fetch and normalize historical OHLC data
- **Output**: Aligned price time series for all tickers
- **Data Source**: yfinance (placeholder - replace with production API)

### 2. StressTestAgent
- **Responsibility**: Simulate portfolio under historical stress scenarios
- **Scenarios**: 2008 Financial Crisis, COVID-19 crash, custom volatility shock
- **Output**: Drawdown, volatility, and recovery metrics per scenario

### 3. ExplainabilityAgent
- **Responsibility**: Compute risk attribution per asset
- **Method**: Variance-based risk decomposition (proxy for SHAP)
- **Output**: Risk contribution percentages and human-readable explanations

### 4. RecommendationAgent
- **Responsibility**: Identify structural risk issues
- **Metrics**: Concentration (Herfindahl index), correlation risk
- **Output**: Structural insights (NO trading recommendations)

## 🧮 Risk Metrics

The Risk Engine computes:

- **VaR (Value at Risk)**: Maximum expected loss at 95% and 99% confidence
- **CVaR (Conditional VaR)**: Expected shortfall beyond VaR threshold
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Max Drawdown**: Largest peak-to-trough decline
- **Volatility**: Annualized portfolio volatility

## 🔧 Design Decisions

### Why Agent-Based Architecture?

1. **Single Responsibility**: Each agent has one clear purpose
2. **Modularity**: Easy to add/remove/modify agents
3. **Testability**: Agents can be tested independently
4. **Scalability**: Agents can be distributed across services
5. **Explainability**: Clear separation makes it easier to explain results

### Why No Price Prediction?

This system focuses on **risk analysis**, not prediction:
- Risk metrics are based on historical data
- Stress tests use past scenarios
- No ML models for price forecasting
- Explains **what happened** and **structural characteristics**, not **what will happen**

### Data Source Strategy

Currently uses `yfinance` as a placeholder. In production:
- Replace with Alpha Vantage, IEX Cloud, or Bloomberg API
- Implement data caching layer
- Add data quality validation
- Support multiple data sources

## 📝 TODO / Production Considerations

- [ ] Replace yfinance with production data API
- [ ] Add database for report storage (PostgreSQL/MongoDB)
- [ ] Implement authentication/authorization
- [ ] Add rate limiting
- [ ] Implement proper logging (structured logging)
- [ ] Add unit tests and integration tests
- [ ] Add data validation and error handling improvements
- [ ] Implement caching layer for price data
- [ ] Add sector mapping for diversification analysis
- [ ] Support for additional asset classes (bonds, commodities)
- [ ] Add monitoring and observability (Prometheus, Grafana)

## 🧪 Testing

```bash
# Run tests (when implemented)
pytest

# Type checking
mypy app/
```

## 📄 License

This is a project scaffold for educational/development purposes.

## 👥 Contributing

This is a production-ready scaffold. Extend as needed for your use case.

---

**Built with**: Python, FastAPI, Pydantic, NumPy, Pandas

**Architecture Pattern**: Agent-Oriented, Coordinator Pattern

**Design Philosophy**: Explainability > Prediction, Modularity > Monolith
