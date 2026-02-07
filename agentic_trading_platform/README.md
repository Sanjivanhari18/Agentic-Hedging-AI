# Agentic Trading Intelligence Platform

An enhanced behavioral trading analysis system with AI-powered insights, probabilistic analysis, counterfactual reasoning, and stock recommendations.

## Architecture

```
agentic_trading_platform/
│
├── app/                              # Web Application
│   ├── main.py                       # FastAPI backend
│   ├── streamlit_app.py              # Streamlit frontend
│   ├── config.py                     # Configuration
│   └── routes/
│       ├── analysis.py               # Analysis API endpoints
│       ├── chat.py                   # Chat API endpoints
│       └── recommend.py              # Recommendation endpoints
│
├── behavioral_trading/               # Core Analysis Engine (Enhanced)
│   ├── main.py                       # BehavioralAnalyzer orchestrator
│   ├── stage1_data/                  # Data Ingestion (CSV/PDF)
│   │   ├── csv_loader.py
│   │   ├── pdf_loader.py
│   │   ├── validator.py
│   │   └── cleaner.py
│   ├── stage2_analysis/              # Enhanced Analysis
│   │   ├── feature_engineering.py    # Behavioral features
│   │   ├── baseline.py              # Statistical baselines
│   │   ├── pattern_discovery.py     # GMM + K-Means + HMM + HDBSCAN
│   │   ├── stability_analyzer.py    # Behavioral consistency
│   │   ├── probabilistic.py         # Bayesian estimation & CI
│   │   └── counterfactual.py        # What-if analysis
│   ├── stage3_viz/                   # Visualization (19 chart types)
│   │   ├── visualizer.py
│   │   └── explainer.py
│   └── utils/
│       └── market_data.py            # yfinance + indicators
│
├── agents/                           # AI Agent System
│   ├── orchestrator.py               # Query router
│   ├── behavioral_agent.py           # Trading pattern insights (RAG)
│   ├── market_agent.py               # Technical analysis
│   ├── recommendation_agent.py       # Stock recommendations
│   └── tools/
│       ├── market_tools.py           # yfinance wrappers
│       └── analysis_tools.py         # Analysis result lookups
│
├── models/                           # ML/DL Models
│   ├── ml/
│   │   ├── momentum_scorer.py        # Momentum ranking
│   │   ├── risk_classifier.py        # Risk classification
│   │   └── stock_screener.py         # Constraint-based screening
│   └── dl/
│       ├── lstm_predictor.py         # LSTM trend prediction
│       └── weights/                  # Saved model weights
│
├── data/
│   ├── vector_store/                 # ChromaDB for RAG
│   └── cache/                        # Market data cache
│
├── requirements.txt
├── .env.example
└── README.md
```

## Features

### Core Analysis (Enhanced)
- **Pattern Discovery**: GMM, K-Means, HMM regime detection, HDBSCAN density clustering
- **Probabilistic Analysis**: Bayesian credible intervals, conditional probabilities, bootstrap CIs
- **Counterfactual Engine**: "What if you held longer?", "What if you used ATR sizing?"
- **19 Visualization Types**: Including probability distributions, confidence bands, Sankey diagrams
- **Explainable AI**: Probability-enriched explanations with credible intervals

### Web Application
- **Streamlit Frontend**: Upload PDF/CSV tradebooks, interactive dashboard, chat interface
- **FastAPI Backend**: RESTful API for analysis, chat, and recommendations

### AI Chatbot
- **LLM-Powered**: Groq API (free tier) with Llama 3 70B
- **RAG**: ChromaDB vector store with analysis context
- **Multi-Agent**: Behavioral insights, market analysis, stock recommendations

### Stock Recommendations
- **ML Models**: Momentum scorer, risk classifier, sector ranker
- **DL Model**: LSTM trend predictor (5-day direction)
- **Constraint-Based**: Filter by sector, risk, horizon, budget, market (US/India)

## Quick Start

### 1. Install Dependencies

```bash
cd agentic_trading_platform
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
cp .env.example .env
# Edit .env and add your Groq API key (free at https://console.groq.com/)
```

### 3. Run the Web App

**Streamlit (recommended):**
```bash
cd agentic_trading_platform
streamlit run app/streamlit_app.py
```

**FastAPI backend:**
```bash
cd agentic_trading_platform
uvicorn app.main:app --reload
```

### 4. Use Programmatically

```python
from behavioral_trading import BehavioralAnalyzer

# Initialize
analyzer = BehavioralAnalyzer(n_clusters=3)

# Load tradebook
analyzer.load_tradebook("path/to/tradebook.pdf")

# Enrich with market data
analyzer.enrich_with_market_data()

# Run full analysis (includes probabilistic + counterfactual)
results = analyzer.analyze()

# Generate visualizations
analyzer.visualize(output_dir="output/")

# Generate report
report = analyzer.generate_report(output_file="output/report.txt")

# Access results
print(results['probabilistic']['probability_statements'])
print(results['counterfactual']['statements'])
```

### 5. Chat with AI Agent

```python
from agents import AgentOrchestrator

# Initialize with analysis results
orchestrator = AgentOrchestrator(analysis_results=results)

# Ask questions
response = orchestrator.process_query("What are my main trading patterns?")
print(response)

response = orchestrator.process_query("Recommend tech stocks for medium risk")
print(response)
```

## Technology Stack

| Category | Libraries |
|----------|-----------|
| **Data** | pandas, numpy, scipy |
| **ML** | scikit-learn, xgboost, hmmlearn, hdbscan |
| **DL** | TensorFlow/Keras (LSTM) |
| **Market Data** | yfinance |
| **Visualization** | plotly, matplotlib, seaborn |
| **Web** | Streamlit, FastAPI, uvicorn |
| **LLM** | LangChain, langchain-groq (Llama 3 70B) |
| **RAG** | ChromaDB, sentence-transformers |
| **PDF** | pdfplumber |
| **Stats** | ruptures, statsmodels |

## Free Tools

All tools are free/open-source:
- **Groq API**: Free tier (30 req/min, very generous)
- **ChromaDB**: Open source, runs locally
- **All Python libraries**: Open source
- **No paid subscriptions required**

## Key Improvements Over Base System

| Aspect | Before | After |
|--------|--------|-------|
| **Pattern Detection** | GMM, K-Means | + HMM, HDBSCAN |
| **Statistics** | Point estimates | Probability distributions + CI |
| **Explanations** | Rule-based only | + Counterfactuals + Probabilities |
| **Interface** | Static HTML files | Interactive web app |
| **Interaction** | None | AI chatbot with RAG |
| **Recommendations** | None | ML/DL-powered |
| **Markets** | Single | US + India |

## License

MIT
