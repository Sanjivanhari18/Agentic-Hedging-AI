# Portfolio Risk Intelligence Platform

A production-ready backend system for explainable, non-predictive portfolio risk analysis using an agent-based architecture.

## 🎯 Project Overview

This system analyzes the **structural and downside risk** of stock portfolios using multiple specialized AI agents. It provides:

- **Risk Metrics**: VaR, CVaR, Sharpe Ratio, Max Drawdown
- **Stress Testing**: Historical scenario analysis (2008 Crisis, COVID-19)
- **Risk Attribution**: Per-asset risk contribution with explanations
- **Structural Insights**: Concentration and diversification analysis

**Important**: This system does **NOT** predict prices or provide trading recommendations. It only analyzes historical risk characteristics.


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
