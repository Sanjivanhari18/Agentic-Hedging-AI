"""
Streamlit frontend for the Agentic Trading Platform.

Pages: Upload & Analyze, Dashboard, Chat, Recommendations.
Uses st.session_state for analysis results and a modern UI with custom CSS.
"""

import html
import logging
import sys
from pathlib import Path

import streamlit as st

# Ensure platform root is on path for app and behavioral_trading imports
_APP_DIR = Path(__file__).resolve().parent
_PLATFORM_ROOT = _APP_DIR.parent
if str(_PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLATFORM_ROOT))

from app.config import (
    ALLOWED_EXTENSIONS,
    DATA_DIR,
    DEFAULT_BASELINE_WINDOW,
    DEFAULT_N_CLUSTERS,
    OUTPUT_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="Agentic Trading Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI
st.markdown(
    """
    <style>
    /* Main container */
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
    /* Sidebar */
    [data-testid="stSidebar"] { background: rgba(22, 33, 62, 0.95); }
    [data-testid="stSidebar"] .stMarkdown { color: #e0e0e0; }
    /* Headers */
    h1, h2, h3 { color: #00d4aa !important; font-weight: 600; }
    /* Cards and blocks */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    /* Metric and info boxes */
    [data-testid="stMetricValue"] { color: #00d4aa; }
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d4aa 0%, #00a884 100%);
        color: #0f0f1a;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.25rem;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #00a884 0%, #008066 100%);
        color: #0f0f1a;
        box-shadow: 0 4px 12px rgba(0, 212, 170, 0.3);
    }
    /* Expander */
    .streamlit-expanderHeader { background: rgba(0, 212, 170, 0.08); border-radius: 6px; }
    /* Report text area */
    .report-box {
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(0, 212, 170, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Consolas', monospace;
        font-size: 0.9rem;
        color: #e0e0e0;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state defaults
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None
if "report_text" not in st.session_state:
    st.session_state["report_text"] = None
if "xai_text" not in st.session_state:
    st.session_state["xai_text"] = None
if "plot_paths" not in st.session_state:
    st.session_state["plot_paths"] = None
if "summary" not in st.session_state:
    st.session_state["summary"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def upload_and_analyze_page():
    """Upload PDF/CSV and run analysis with progress."""
    st.title("📤 Upload & Analyze")
    st.markdown("Upload a tradebook (CSV or PDF) to run behavioral analysis.")

    allowed_suffixes = [e.lstrip(".") for e in ALLOWED_EXTENSIONS]
    uploaded = st.file_uploader(
        "Choose a file",
        type=allowed_suffixes,
        help="Supported: CSV, PDF",
    )

    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.number_input(
            "Number of clusters",
            min_value=2,
            max_value=10,
            value=DEFAULT_N_CLUSTERS,
            help="Behavioral pattern clusters",
        )
    with col2:
        baseline_window = st.number_input(
            "Baseline window (days)",
            min_value=5,
            max_value=90,
            value=DEFAULT_BASELINE_WINDOW,
        )

    if uploaded is not None:
        if st.button("Run analysis", type="primary"):
            cache_dir = DATA_DIR / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            file_path = cache_dir / uploaded.name
            file_path.write_bytes(uploaded.getvalue())
            file_type = Path(uploaded.name).suffix.lower().lstrip(".")

            progress = st.progress(0, text="Loading tradebook...")
            try:
                progress.progress(10, text="Loading & validating...")
                from behavioral_trading import BehavioralAnalyzer

                analyzer = BehavioralAnalyzer(
                    n_clusters=n_clusters,
                    baseline_window=baseline_window,
                )
                analyzer.load_tradebook(str(file_path), file_type=file_type)
                progress.progress(30, text="Enriching with market data...")
                analyzer.enrich_with_market_data()
                progress.progress(50, text="Running behavioral analysis...")
                results = analyzer.analyze()
                progress.progress(70, text="Generating visualizations...")
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                plot_paths = analyzer.visualize(
                    results=results, output_dir=str(OUTPUT_DIR)
                )
                progress.progress(85, text="Writing report...")
                report_path = str(OUTPUT_DIR / "behavioral_report.txt")
                report_text = analyzer.generate_report(
                    results=results, output_file=report_path
                )
                xai_path = OUTPUT_DIR / "xai_explanation.txt"
                xai_text = (
                    xai_path.read_text(encoding="utf-8") if xai_path.exists() else ""
                )
                summary = analyzer.get_summary()
                progress.progress(100, text="Done.")

                st.session_state["analysis_results"] = results
                st.session_state["report_text"] = report_text
                st.session_state["xai_text"] = xai_text
                st.session_state["plot_paths"] = dict(plot_paths)
                st.session_state["summary"] = summary
                st.success("Analysis complete. Open **Dashboard** to view results.")
            except Exception as e:
                logger.exception("Analysis failed: %s", e)
                st.error(f"Analysis failed: {e}")
            finally:
                progress.empty()


def dashboard_page():
    """Display plotly figures (HTML), XAI report, and full behavioral report."""
    st.title("📊 Dashboard")
    if st.session_state.get("plot_paths") is None and st.session_state.get(
        "report_text"
    ) is None:
        st.info(
            "No analysis results yet. Go to **Upload & Analyze**, upload a file, and run analysis."
        )
        return

    # Summary metrics
    summary = st.session_state.get("summary")
    if summary:
        st.subheader("Summary")
        cols = st.columns(4)
        if "total_trades" in summary:
            cols[0].metric("Total trades", summary["total_trades"])
        if summary.get("pnl"):
            cols[1].metric("Total P&L", f"${summary['pnl']['total']:.2f}")
            cols[2].metric("Win rate", f"{100 * summary['pnl']['win_rate']:.1f}%")
            cols[3].metric("Avg P&L/Trade", f"${summary['pnl']['average']:.2f}")
        if summary.get("date_range"):
            st.caption(
                f"Date range: {summary['date_range'].get('start', 'N/A')} to "
                f"{summary['date_range'].get('end', 'N/A')}"
            )
        st.divider()

    # Tabs for organized content
    tab_xai, tab_charts, tab_report = st.tabs(
        ["XAI Report", "Visualizations", "Full Behavioral Report"]
    )

    with tab_xai:
        xai_text = st.session_state.get("xai_text") or ""
        if xai_text:
            st.markdown(
                f'<div class="report-box">{html.escape(xai_text)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No XAI explanation available.")

    with tab_charts:
        plot_paths = st.session_state.get("plot_paths") or {}
        if not plot_paths:
            st.caption("No plots generated.")
        else:
            st.markdown(f"**{len(plot_paths)} visualizations generated**")
            for name, path in plot_paths.items():
                p = Path(path)
                if p.exists():
                    try:
                        html_content = p.read_text(encoding="utf-8")
                        with st.expander(f"📈 {name.replace('_', ' ').title()}", expanded=False):
                            st.components.v1.html(html_content, height=450, scrolling=True)
                    except Exception as e:
                        st.warning(f"Could not load {name}: {e}")

    with tab_report:
        report_text = st.session_state.get("report_text") or ""
        if report_text:
            st.markdown(
                f'<div class="report-box">{html.escape(report_text)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No behavioral report available.")


def _get_orchestrator():
    """Get or create the agent orchestrator, connected to analysis results."""
    if "orchestrator" not in st.session_state:
        try:
            from agents import AgentOrchestrator
            st.session_state["orchestrator"] = AgentOrchestrator(
                analysis_results=st.session_state.get("analysis_results")
            )
        except Exception:
            st.session_state["orchestrator"] = None
    # Update with latest analysis results
    orch = st.session_state.get("orchestrator")
    if orch is not None and st.session_state.get("analysis_results"):
        try:
            orch.update_analysis_results(st.session_state["analysis_results"])
        except Exception:
            pass
    return orch


def chat_page():
    """Chat UI connected to the AI agent orchestrator."""
    st.title("💬 Chat")
    st.markdown("Ask questions about your trading analysis, market conditions, or get stock recommendations.")

    for entry in st.session_state["chat_history"]:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    prompt = st.chat_input("Ask a question...")
    if prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            orchestrator = _get_orchestrator()
            if orchestrator is not None:
                try:
                    msg = orchestrator.process_query(prompt)
                except Exception as e:
                    msg = f"Sorry, I encountered an error: {e}"
            else:
                msg = (
                    "The AI agent system requires additional dependencies "
                    "(langchain-groq, chromadb). Install them and set GROQ_API_KEY "
                    "in your .env file for full chat capabilities."
                )
            st.markdown(msg)
        st.session_state["chat_history"].append({"role": "assistant", "content": msg})


def recommendations_page():
    """Form for constraints and stock screening results, plus LSTM trend prediction."""
    st.title("🎯 Stock Recommendations")

    tab_screen, tab_predict = st.tabs(["ML Stock Screener", "LSTM Trend Predictor"])

    with tab_screen:
        st.markdown("Set your constraints to get personalized stock recommendations powered by ML.")

        with st.form("recommend_form"):
            col1, col2 = st.columns(2)
            with col1:
                sector = st.selectbox(
                    "Sector",
                    ["any", "technology", "healthcare", "finance", "energy", "consumer", "telecom"],
                )
                risk = st.selectbox(
                    "Risk tolerance",
                    ["low", "medium", "high"],
                )
            with col2:
                horizon = st.selectbox(
                    "Investment horizon",
                    ["short", "medium", "long"],
                    help="Short (<1mo), Medium (1-6mo), Long (>6mo)",
                )
                market = st.selectbox("Market", ["us", "india"])
            budget = st.number_input(
                "Budget ($, optional)",
                min_value=0.0,
                value=0.0,
                step=1000.0,
            )
            submitted = st.form_submit_button("Get Recommendations", type="primary")

        if submitted:
            with st.spinner("Screening stocks with ML models..."):
                try:
                    from models.ml.stock_screener import StockScreener
                    from models.ml.momentum_scorer import MomentumScorer
                    from models.ml.risk_classifier import RiskClassifier

                    scorer = MomentumScorer()
                    classifier = RiskClassifier()
                    screener = StockScreener(momentum_scorer=scorer, risk_classifier=classifier)

                    constraints = {
                        "sector": sector if sector != "any" else None,
                        "risk_tolerance": risk,
                        "horizon": horizon,
                        "market": market,
                        "budget": budget if budget > 0 else None,
                    }
                    results_df = screener.screen(constraints)

                    if results_df is not None and len(results_df) > 0:
                        st.success(f"Found {len(results_df)} recommendations")
                        st.dataframe(results_df, use_container_width=True)
                    else:
                        st.warning("No stocks matched your criteria. Try relaxing constraints.")
                except Exception as e:
                    logger.exception("Recommendation failed: %s", e)
                    st.error(f"Recommendation engine error: {e}")

    with tab_predict:
        st.markdown(
            "Predict 5-day trend direction for a stock using LSTM deep learning model. "
            "The model uses OHLCV, RSI, MACD, EMA, ATR, and volatility features."
        )
        symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            help="Enter a ticker symbol (e.g. AAPL, MSFT, RELIANCE.NS)",
        )
        if st.button("Predict Trend", type="primary"):
            with st.spinner(f"Training LSTM and predicting trend for {symbol}..."):
                try:
                    from models.dl.lstm_predictor import LSTMTrendPredictor

                    predictor = LSTMTrendPredictor()
                    X, y = predictor.prepare_data(symbol, period="2y")
                    if X.size == 0:
                        st.warning(f"Insufficient data for {symbol}.")
                    else:
                        predictor.train(X, y, epochs=20, validation_split=0.2)
                        result = predictor.predict_symbol(symbol)

                        if result.get("error"):
                            st.error(result["error"])
                        else:
                            trend = result.get("trend", "unknown").upper()
                            st.subheader(f"Predicted 5-day trend: **{trend}**")
                            cols = st.columns(3)
                            cols[0].metric("P(Up)", f"{(result.get('prob_up') or 0)*100:.1f}%")
                            cols[1].metric("P(Sideways)", f"{(result.get('prob_sideways') or 0)*100:.1f}%")
                            cols[2].metric("P(Down)", f"{(result.get('prob_down') or 0)*100:.1f}%")
                            st.caption(
                                "Disclaimer: This is a model prediction based on historical patterns. "
                                "It is NOT financial advice. Always do your own research."
                            )
                except ImportError:
                    st.error("TensorFlow is required for LSTM predictions. Install with: pip install tensorflow")
                except Exception as e:
                    logger.exception("LSTM prediction failed: %s", e)
                    st.error(f"Prediction error: {e}")


def main():
    """Sidebar navigation and page routing."""
    st.sidebar.title("📈 Agentic Trading Intelligence")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["Upload & Analyze", "Dashboard", "Chat", "Recommendations"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Behavioral analysis & agentic insights")

    if page == "Upload & Analyze":
        upload_and_analyze_page()
    elif page == "Dashboard":
        dashboard_page()
    elif page == "Chat":
        chat_page()
    else:
        recommendations_page()


if __name__ == "__main__":
    main()
