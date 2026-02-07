"""Behavioral Insights Agent - answers questions about user's trading patterns."""

import json
import logging
import uuid
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.prompts import ChatPromptTemplate
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    ChatGroq = None
    HumanMessage = None
    SystemMessage = None
    ChatPromptTemplate = None

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    chromadb = None
    Settings = None

try:
    from chromadb.utils import embedding_functions
    _default_ef = embedding_functions.DefaultEmbeddingFunction()
    HAS_CHROMADB_EF = True
except Exception:
    _default_ef = None
    HAS_CHROMADB_EF = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_SPLITTER = True
except ImportError:
    HAS_SPLITTER = False
    RecursiveCharacterTextSplitter = None


def _simple_split(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Simple chunking when RecursiveCharacterTextSplitter is not available."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_break = max(chunk.rfind("\n"), chunk.rfind(". "), chunk.rfind("; "))
            if last_break > chunk_size // 2:
                end = start + last_break + 1
                chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


class BehavioralInsightsAgent:
    """Answers questions about user's trading behavioral patterns using RAG."""

    COLLECTION_NAME = "behavioral_insights"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    TOP_K_RETRIEVAL = 5

    def __init__(self, analysis_results: Optional[Dict] = None, api_key: str = ""):
        self.api_key = api_key or ""
        self.analysis_results = analysis_results or {}
        self.llm = None
        self.vectorstore = None
        self._collection = None
        self._client = None
        self._setup()

    def _setup(self) -> None:
        """Initialize LLM and vectorstore."""
        if HAS_LANGCHAIN and self.api_key:
            try:
                self.llm = ChatGroq(
                    model="llama-3.1-70b-versatile",
                    api_key=self.api_key,
                    temperature=0.2,
                )
            except Exception as e:
                logger.warning("Could not initialize Groq LLM: %s", e)
                self.llm = None

        if HAS_CHROMADB:
            try:
                self._client = chromadb.Client(Settings(anonymized_telemetry=False))
                kwargs = {
                    "name": self.COLLECTION_NAME,
                    "metadata": {"description": "Behavioral analysis context"},
                }
                if HAS_CHROMADB_EF and _default_ef is not None:
                    kwargs["embedding_function"] = _default_ef
                self._collection = self._client.get_or_create_collection(**kwargs)
                self.vectorstore = True
            except Exception as e:
                logger.warning("Could not initialize ChromaDB: %s", e)
                self._client = None
                self._collection = None
                self.vectorstore = None

        if self.analysis_results:
            self._index_analysis()

    def _build_context(self, analysis_results: Dict) -> str:
        """Build text context from analysis results for RAG."""
        parts = []

        # Features summary
        features = analysis_results.get("features")
        if features is not None:
            try:
                import pandas as pd
                if isinstance(features, pd.DataFrame):
                    n = len(features)
                    parts.append(f"Total trades in analysis: {n}.")
                    if "date" in features.columns:
                        parts.append(
                            f"Date range: {features['date'].min()} to {features['date'].max()}."
                        )
                    if "realized_pnl" in features.columns:
                        pnl = features["realized_pnl"]
                        parts.append(
                            f"Realized PnL: total={pnl.sum():.2f}, mean={pnl.mean():.2f}, "
                            f"win rate={(pnl > 0).mean()*100:.1f}%."
                        )
                    num_cols = [c for c in features.columns if features[c].dtype in ["float64", "int64"]]
                    if num_cols:
                        summary = features[num_cols].describe().to_string()
                        parts.append(f"Feature summary:\n{summary}")
            except Exception as e:
                logger.debug("Feature summary build failed: %s", e)

        # Baselines
        baselines = analysis_results.get("baselines") or {}
        if baselines:
            parts.append("Behavioral baselines (typical values):")
            for feat, stats in list(baselines.items())[:20]:
                if isinstance(stats, dict):
                    mean_v = stats.get("mean")
                    median_v = stats.get("median")
                    if mean_v is not None or median_v is not None:
                        parts.append(f"  {feat}: mean={mean_v}, median={median_v}")

        # Cluster descriptions
        patterns = analysis_results.get("patterns") or {}
        clusters = patterns.get("clusters") or {}
        analysis = clusters.get("analysis") or {}
        if analysis:
            parts.append("Behavioral clusters (pattern descriptions):")
            for cid, desc in analysis.items():
                if isinstance(desc, dict):
                    parts.append(f"  Cluster {cid}: {json.dumps(desc)[:500]}")
                else:
                    parts.append(f"  Cluster {cid}: {desc}")
        labels = clusters.get("labels")
        if labels is not None:
            try:
                import pandas as pd
                if hasattr(labels, "value_counts"):
                    parts.append("Cluster distribution: " + str(labels.value_counts().to_dict()))
            except Exception:
                pass

        # Change points
        cp = patterns.get("change_points") or {}
        indices = cp.get("indices") or []
        if indices:
            parts.append(f"Behavioral change points detected at indices: {indices[:20]}.")

        # Stability
        stability = analysis_results.get("stability") or {}
        if stability.get("stability_score") is not None:
            parts.append(
                f"Behavioral stability score: {stability['stability_score']:.2f}. "
                f"Interpretation: {stability.get('interpretation', 'N/A')}. "
                f"Note: {stability.get('note', '')}"
            )

        # Probabilistic
        prob = analysis_results.get("probabilistic") or {}
        if prob.get("probability_statements"):
            parts.append("Probability statements:")
            for st in prob["probability_statements"][:15]:
                parts.append(f"  - {st}")
        if prob.get("credible_intervals"):
            parts.append("Credible intervals: " + json.dumps(prob["credible_intervals"], default=str)[:800])

        # Counterfactual
        cf = analysis_results.get("counterfactual") or {}
        if cf.get("statements"):
            parts.append("Counterfactual insights (what-if):")
            for st in cf["statements"][:10]:
                parts.append(f"  - {st}")

        return "\n\n".join(parts) if parts else "No behavioral analysis context available."

    def _index_context(self, context_text: str) -> None:
        """Index context into ChromaDB for retrieval."""
        if not context_text or not self._collection:
            return
        try:
            if HAS_SPLITTER and RecursiveCharacterTextSplitter:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.CHUNK_SIZE,
                    chunk_overlap=self.CHUNK_OVERLAP,
                )
                chunks = splitter.split_text(context_text)
            else:
                chunks = _simple_split(context_text, self.CHUNK_SIZE, self.CHUNK_OVERLAP)
            if not chunks:
                return
            ids = [str(uuid.uuid4()) for _ in chunks]
            self._collection.add(
                ids=ids,
                documents=chunks,
                metadatas=[{"chunk": i} for i in range(len(chunks))],
            )
            logger.info("Indexed %d chunks into behavioral context.", len(chunks))
        except Exception as e:
            logger.warning("ChromaDB index failed: %s", e)

    def _index_analysis(self) -> None:
        """Build context from current analysis and index it."""
        try:
            if self._collection:
                existing = self._collection.get()
                if existing and existing.get("ids"):
                    self._collection.delete(ids=existing["ids"])
        except Exception:
            pass
        context = self._build_context(self.analysis_results)
        self._index_context(context)

    def _retrieve(self, question: str, top_k: int = TOP_K_RETRIEVAL) -> List[str]:
        """Retrieve relevant chunks for the question."""
        if not self._collection or not question.strip():
            return []
        try:
            res = self._collection.query(
                query_texts=[question.strip()],
                n_results=min(top_k, 10),
            )
            docs = res.get("documents", [[]])
            return list(docs[0]) if docs else []
        except Exception as e:
            logger.warning("ChromaDB retrieve failed: %s", e)
            return []

    def _rule_based_answer(self, question: str) -> Optional[str]:
        """Keyword-based fallback when LLM is unavailable."""
        q = question.lower()
        context = self._build_context(self.analysis_results)
        if "no behavioral analysis" in context.lower():
            return "No behavioral analysis results are loaded. Please run an analysis first (upload a tradebook and analyze)."
        if any(k in q for k in ["pattern", "cluster", "group"]):
            patterns = self.analysis_results.get("patterns") or {}
            clusters = patterns.get("clusters") or {}
            analysis = clusters.get("analysis") or {}
            if analysis:
                lines = ["Behavioral clusters identified:", ""]
                for cid, desc in analysis.items():
                    lines.append(f"Cluster {cid}: {desc}")
                return "\n".join(lines)
        if any(k in q for k in ["stability", "consistent", "consistency"]):
            s = self.analysis_results.get("stability") or {}
            if s.get("stability_score") is not None:
                return f"Behavioral stability score: {s['stability_score']:.2f}. {s.get('interpretation', '')}"
        if any(k in q for k in ["anomal", "outlier", "unusual"]):
            patterns = self.analysis_results.get("patterns") or {}
            anom = patterns.get("anomalies") or {}
            indices = anom.get("indices") or anom.get("anomaly_indices") or []
            return f"Anomalous trades detected at {len(indices)} indices. Details in analysis anomalies."
        if any(k in q for k in ["pnl", "profit", "win rate", "summary"]):
            from .tools.analysis_tools import get_trade_summary
            summary = get_trade_summary()
            if summary.get("error"):
                return summary["error"]
            lines = [f"Total trades: {summary.get('total_trades', 'N/A')}"]
            if summary.get("date_range"):
                lines.append(f"Date range: {summary['date_range'].get('start')} to {summary['date_range'].get('end')}")
            if summary.get("pnl"):
                p = summary["pnl"]
                lines.append(f"PnL: total={p.get('total')}, avg={p.get('average')}, win rate={p.get('win_rate')}")
            return "\n".join(lines)
        return None

    def answer(self, question: str) -> str:
        """Retrieve relevant context and generate answer."""
        if not question or not question.strip():
            return "Please ask a question about your trading behavior or analysis."

        from .tools.analysis_tools import set_analysis_results
        set_analysis_results(self.analysis_results)

        # Try RAG + LLM
        if self.llm and self._collection:
            chunks = self._retrieve(question)
            context_block = "\n\n".join(chunks) if chunks else self._build_context(self.analysis_results)
            sys_msg = (
                "You are a behavioral trading analyst. Answer the user's question "
                "using ONLY the provided context from their trading analysis. "
                "Be concise and factual. If the context does not contain the answer, say so."
            )
            try:
                messages = [
                    SystemMessage(content=sys_msg),
                    HumanMessage(content=f"Context:\n{context_block}\n\nQuestion: {question}"),
                ]
                resp = self.llm.invoke(messages)
                if hasattr(resp, "content") and resp.content:
                    return resp.content.strip()
            except Exception as e:
                logger.warning("Behavioral LLM invoke failed: %s", e)

        # Fallback: rule-based
        fallback = self._rule_based_answer(question)
        if fallback:
            return fallback
        return (
            "I can answer questions about your trading patterns, clusters, stability, "
            "anomalies, and PnL summary when analysis results are available. "
            "No analysis is loaded or the question could not be answered from it."
        )

    def update_results(self, analysis_results: Dict) -> None:
        """Update with new analysis results and re-index."""
        self.analysis_results = analysis_results or {}
        if self._collection and self.analysis_results:
            self._index_analysis()
