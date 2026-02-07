"""
Enhanced BehavioralAnalyzer - Main application entry point.

Builds on the existing 3-stage pipeline and adds:
- ProbabilisticAnalyzer (probability statements, credible intervals)
- CounterfactualEngine (what-if analysis)
"""

import pandas as pd
import logging
from typing import Optional, Dict
import os

from .stage1_data import (
    CSVTradebookLoader,
    PDFTradebookLoader,
    TradebookValidator,
    TradebookCleaner
)
from .utils.market_data import MarketDataFetcher
from .stage2_analysis import (
    BehavioralFeatureEngineer,
    BaselineConstructor,
    PatternDiscoverer,
    BehavioralStabilityAnalyzer
)
from .stage2_analysis.probabilistic import ProbabilisticAnalyzer
from .stage2_analysis.counterfactual import CounterfactualEngine
from .stage3_viz import (
    BehavioralVisualizer,
    BehavioralExplainer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BehavioralAnalyzer:
    """Main class for enhanced behavioral trading analysis."""
    
    def __init__(self, n_clusters: int = 3, baseline_window: int = 30):
        """
        Initialize the behavioral analyzer.
        
        Args:
            n_clusters: Number of behavioral clusters to discover
            baseline_window: Rolling window size for baseline calculation
        """
        self.n_clusters = n_clusters
        self.baseline_window = baseline_window
        
        # Stage 1: Data Ingestion
        self.csv_loader = CSVTradebookLoader()
        self.pdf_loader = PDFTradebookLoader()
        self.validator = TradebookValidator()
        self.cleaner = TradebookCleaner()
        self.market_fetcher = MarketDataFetcher()
        
        # Stage 2: Analysis
        self.feature_engineer = BehavioralFeatureEngineer()
        self.baseline_constructor = BaselineConstructor(window_size=baseline_window)
        self.pattern_discoverer = PatternDiscoverer(n_clusters=n_clusters)
        self.stability_analyzer = BehavioralStabilityAnalyzer(window_size=baseline_window)
        
        # NEW: Probabilistic & Counterfactual
        self.probabilistic_analyzer = ProbabilisticAnalyzer()
        self.counterfactual_engine = CounterfactualEngine()
        
        # Stage 3: Visualization & Explanation
        self.visualizer = BehavioralVisualizer()
        self.explainer = BehavioralExplainer()
        
        # State
        self.trades: Optional[pd.DataFrame] = None
        self.enriched_trades: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.baselines: Optional[Dict] = None
        self.pattern_results: Optional[Dict] = None
        self.stability_results: Optional[Dict] = None
        self.probabilistic_results: Optional[Dict] = None
        self.counterfactual_results: Optional[Dict] = None
    
    def load_tradebook(self, filepath: str, file_type: Optional[str] = None) -> pd.DataFrame:
        """
        Load tradebook from CSV or PDF file.
        
        Args:
            filepath: Path to tradebook file
            file_type: 'csv' or 'pdf'. If None, inferred from extension.
        
        Returns:
            Loaded trades DataFrame
        """
        if file_type is None:
            file_type = os.path.splitext(filepath)[1].lower().lstrip('.')
        
        if file_type == 'csv':
            logger.info(f"Loading CSV tradebook from {filepath}")
            self.trades = self.csv_loader.load(filepath)
        elif file_type == 'pdf':
            logger.info(f"Loading PDF tradebook from {filepath}")
            self.trades = self.pdf_loader.load(filepath)
        else:
            raise ValueError(f"Unsupported file type: {file_type}. Use 'csv' or 'pdf'.")
        
        # Validate
        validation = self.validator.validate(self.trades)
        if not self.validator.is_valid():
            logger.error(f"Validation errors: {validation['errors']}")
            raise ValueError(f"Tradebook validation failed: {validation['errors']}")
        
        if validation['warnings']:
            logger.warning(f"Validation warnings: {validation['warnings']}")
        
        # Clean
        logger.info("Cleaning tradebook data...")
        self.trades = self.cleaner.clean(self.trades)
        
        logger.info(f"Successfully loaded {len(self.trades)} trades")
        return self.trades
    
    def enrich_with_market_data(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Enrich trades with market context data.
        
        Args:
            symbol: Stock symbol (optional - if None, uses symbol column from trades)
        
        Returns:
            Enriched trades DataFrame
        """
        if self.trades is None:
            raise ValueError("No trades loaded. Call load_tradebook() first.")
        
        if symbol is None:
            logger.info("Enriching trades with market data for all symbols in tradebook...")
        else:
            logger.info(f"Enriching trades with market data for {symbol}...")
        
        self.enriched_trades = self.market_fetcher.enrich_trades_with_market_data(
            self.trades, symbol=symbol
        )
        
        logger.info("Market data enrichment complete")
        return self.enriched_trades
    
    def analyze(self) -> Dict:
        """
        Run full behavioral analysis pipeline.
        
        Returns:
            Dictionary containing all analysis results including
            probabilistic and counterfactual analysis
        """
        if self.enriched_trades is None:
            raise ValueError("No enriched trades available. Call enrich_with_market_data() first.")
        
        logger.info("Starting behavioral analysis...")
        
        # Stage 2.1: Feature Engineering
        logger.info("Engineering behavioral features...")
        self.features = self.feature_engineer.engineer_features(self.enriched_trades)
        
        # Stage 2.2: Baseline Construction
        logger.info("Constructing behavioral baselines...")
        self.baselines = self.baseline_constructor.construct_baselines(self.features)
        self.features = self.baseline_constructor.calculate_deviations(self.features)
        
        # Stage 2.3: Pattern Discovery (enhanced with HMM, HDBSCAN)
        logger.info("Discovering behavioral patterns...")
        self.pattern_results = self.pattern_discoverer.discover_patterns(self.features)
        
        # Add cluster labels to features
        if 'clusters' in self.pattern_results:
            self.features['behavioral_cluster'] = self.pattern_results['clusters']['labels']
        
        # Stage 2.4: Behavioral Stability Analysis
        logger.info("Calculating behavioral stability score...")
        self.stability_results = self.stability_analyzer.calculate_stability_score(self.features)
        
        # Stage 2.5: Probabilistic Analysis (NEW)
        logger.info("Running probabilistic analysis...")
        try:
            self.probabilistic_results = self.probabilistic_analyzer.analyze(
                self.features, self.baselines
            )
        except Exception as e:
            logger.warning(f"Probabilistic analysis failed: {e}")
            self.probabilistic_results = {}
        
        # Stage 2.6: Counterfactual Analysis (NEW)
        logger.info("Running counterfactual analysis...")
        try:
            self.counterfactual_results = self.counterfactual_engine.analyze(
                self.features, market_data_fetcher=self.market_fetcher
            )
        except Exception as e:
            logger.warning(f"Counterfactual analysis failed: {e}")
            self.counterfactual_results = {}
        
        logger.info("Behavioral analysis complete")
        
        return {
            'features': self.features,
            'baselines': self.baselines,
            'patterns': self.pattern_results,
            'stability': self.stability_results,
            'probabilistic': self.probabilistic_results,
            'counterfactual': self.counterfactual_results
        }
    
    def visualize(self, results: Optional[Dict] = None, output_dir: str = "output/") -> Dict[str, str]:
        """
        Generate all visualizations.
        
        Args:
            results: Analysis results (if None, uses internal results)
            output_dir: Directory to save visualizations
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        if results is None:
            if self.features is None or self.pattern_results is None:
                raise ValueError("No analysis results available. Call analyze() first.")
            results = {
                'features': self.features,
                'patterns': self.pattern_results,
                'stability': self.stability_results
            }
        
        features = results['features']
        patterns = results['patterns']
        
        logger.info("Generating visualizations...")
        
        # Regime timeline
        if 'clusters' in patterns and 'labels' in patterns['clusters']:
            clusters = patterns['clusters']['labels']
            change_points = patterns.get('change_points', {}).get('indices', [])
            try:
                self.visualizer.create_regime_timeline(features, clusters, change_points)
            except Exception as e:
                logger.warning(f"Could not create regime timeline: {e}")
        
        # Deviation plots
        try:
            self.visualizer.create_deviation_plots(features)
        except Exception as e:
            logger.warning(f"Could not create deviation plots: {e}")
        
        # Post-event charts
        try:
            self.visualizer.create_post_event_charts(features)
        except Exception as e:
            logger.warning(f"Could not create post-event charts: {e}")
        
        # Performance matrix
        if 'clusters' in patterns and 'labels' in patterns['clusters']:
            clusters = patterns['clusters']['labels']
            try:
                self.visualizer.create_performance_matrix(features, clusters, self.n_clusters)
            except Exception as e:
                logger.warning(f"Could not create performance matrix: {e}")
        
        # Cluster timeline visualization
        if 'clusters' in patterns and 'labels' in patterns['clusters']:
            clusters = patterns['clusters']['labels']
            try:
                self.visualizer.create_cluster_timeline(features, clusters)
            except Exception as e:
                logger.warning(f"Could not create cluster timeline: {e}")
        
        # Cluster scatter plot visualization
        if 'clusters' in patterns and 'labels' in patterns['clusters']:
            clusters = patterns['clusters']['labels']
            cluster_centers = patterns['clusters'].get('cluster_centers')
            cluster_analysis = patterns['clusters'].get('analysis', {})
            if cluster_centers is not None:
                try:
                    self.visualizer.create_cluster_scatter_plot(
                        features, clusters, cluster_centers, cluster_analysis
                    )
                except Exception as e:
                    logger.warning(f"Could not create cluster scatter plot: {e}")
        
        # Trade Journey Timeline
        try:
            self.visualizer.create_trade_journey_timeline(features)
        except Exception as e:
            logger.warning(f"Could not create trade journey timeline: {e}")
        
        # Signal Following Scorecard
        try:
            self.visualizer.create_signal_following_scorecard(features)
        except Exception as e:
            logger.warning(f"Could not create signal scorecard: {e}")
        
        # Behavioral Stability Scorecard
        if 'stability' in results and results['stability']:
            try:
                self.visualizer.create_stability_scorecard(results['stability'])
            except Exception as e:
                logger.warning(f"Could not create stability scorecard: {e}")
        
        # Volatility Timeline with Trade Markers
        try:
            self.visualizer.create_volatility_trade_timeline(features, self.market_fetcher)
        except Exception as e:
            logger.warning(f"Could not create volatility timeline: {e}")
        
        # S&P 500 Volatility Timeline
        try:
            self.visualizer.create_sp500_volatility_timeline(features, self.market_fetcher)
        except Exception as e:
            logger.warning(f"Could not create S&P 500 volatility timeline: {e}")
        
        # MACD Stock Charts
        try:
            macd_charts = self.visualizer.create_macd_stock_charts(features, self.market_fetcher)
            if macd_charts:
                self.visualizer.figures['macd_stock_charts'] = macd_charts
        except Exception as e:
            logger.warning(f"Could not create MACD stock charts: {e}")
        
        os.makedirs(output_dir, exist_ok=True)
        self.visualizer.save_all_figures(output_dir)
        
        # Create unified dashboard
        try:
            dashboard_path = self.visualizer.create_unified_dashboard(
                output_dir=output_dir,
                xai_file="xai_explanation.txt",
                report_file="behavioral_report.txt"
            )
            logger.info(f"Unified dashboard created: {dashboard_path}")
        except Exception as e:
            logger.warning(f"Could not create unified dashboard: {e}")
        
        logger.info(f"Visualizations saved to {output_dir}")
        
        return {name: os.path.join(output_dir, f"{name}.html") 
                for name in self.visualizer.figures.keys()}
    
    def generate_report(self, results: Optional[Dict] = None, 
                       output_file: str = "behavioral_report.txt") -> str:
        """
        Generate comprehensive behavioral report with probability
        statements and counterfactual insights.
        
        Args:
            results: Analysis results (if None, uses internal results)
            output_file: Path to save report
        
        Returns:
            Report text
        """
        if results is None:
            if self.features is None or self.baselines is None or self.pattern_results is None:
                raise ValueError("No analysis results available. Call analyze() first.")
            results = {
                'features': self.features,
                'baselines': self.baselines,
                'patterns': self.pattern_results,
                'stability': self.stability_results,
                'probabilistic': self.probabilistic_results,
                'counterfactual': self.counterfactual_results
            }
        
        logger.info("Generating behavioral report...")
        
        report = self.explainer.generate_report(
            results['features'],
            results['baselines'],
            results['patterns']
        )
        
        # Add stability section
        if results.get('stability') and results['stability'].get('stability_score') is not None:
            stability_section = "\n\n" + "=" * 80 + "\n"
            stability_section += "BEHAVIORAL STABILITY / CONSISTENCY SCORE\n"
            stability_section += "=" * 80 + "\n\n"
            stability_section += results['stability']['interpretation'] + "\n\n"
            stability_section += results['stability']['note'] + "\n"
            report += stability_section
        
        # NEW: Add probability statements section
        prob_results = results.get('probabilistic', {})
        if prob_results and prob_results.get('probability_statements'):
            prob_section = "\n\n" + "=" * 80 + "\n"
            prob_section += "PROBABILISTIC ANALYSIS\n"
            prob_section += "=" * 80 + "\n\n"
            for statement in prob_results['probability_statements']:
                prob_section += f"  - {statement}\n"
            
            # Add credible intervals summary
            if prob_results.get('credible_intervals'):
                prob_section += "\nCredible Intervals (95%):\n"
                for metric, ci in prob_results['credible_intervals'].items():
                    if isinstance(ci, dict) and 'mean' in ci:
                        prob_section += (
                            f"  {metric}: {ci['mean']:.4f} "
                            f"(95% CI: [{ci.get('ci_lower', 'N/A'):.4f}, "
                            f"{ci.get('ci_upper', 'N/A'):.4f}])\n"
                        )
            report += prob_section
        
        # NEW: Add counterfactual insights section
        cf_results = results.get('counterfactual', {})
        if cf_results and cf_results.get('statements'):
            cf_section = "\n\n" + "=" * 80 + "\n"
            cf_section += "COUNTERFACTUAL ANALYSIS (WHAT-IF SCENARIOS)\n"
            cf_section += "=" * 80 + "\n\n"
            for statement in cf_results['statements']:
                cf_section += f"  - {statement}\n"
            report += cf_section
        
        # Generate XAI summary
        xai_summary = self.explainer.generate_xai_summary(
            results['features'],
            results['baselines'],
            results['patterns'],
            results.get('stability')
        )
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report)
        
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "output"
        xai_output_file = os.path.join(output_dir, "xai_explanation.txt")
        with open(xai_output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXPLAINABLE AI (XAI) SUMMARY - BEHAVIORAL TRADING ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write("This summary uses rule-based Natural Language Generation (NLG) and feature\n")
            f.write("contribution ranking to explain your trading behavior in simple, quantifiable terms.\n\n")
            f.write("-" * 80 + "\n")
            f.write("KEY FINDINGS:\n")
            f.write("-" * 80 + "\n\n")
            f.write(xai_summary)
            f.write("\n\n")
            f.write("=" * 80 + "\n")
            f.write("Generated using:\n")
            f.write("- Rule-based NLG for natural language explanations\n")
            f.write("- Feature contribution ranking to identify key behavioral drivers\n")
            f.write("- Statistical analysis of trading patterns and deviations\n")
            f.write("- Bayesian probability estimation for behavioral insights\n")
            f.write("- Counterfactual simulation for what-if analysis\n")
            f.write("=" * 80 + "\n")
        
        # Stock performance report
        try:
            stock_performance_report = self.explainer.analyze_stock_performance(
                results['features'],
                results['baselines']
            )
            stock_perf_file = os.path.join(output_dir, "stock_performance_analysis.txt")
            with open(stock_perf_file, 'w', encoding='utf-8') as f:
                f.write(stock_performance_report)
            logger.info(f"Stock performance analysis saved to {stock_perf_file}")
        except Exception as e:
            logger.warning(f"Could not generate stock performance analysis: {e}")
        
        logger.info(f"Report saved to {output_file}")
        logger.info(f"XAI explanation saved to {xai_output_file}")
        return report
    
    def explain_trade(self, trade_idx: int) -> str:
        """
        Generate explanation for a specific trade.
        
        Args:
            trade_idx: Index of trade to explain
        
        Returns:
            Natural language explanation
        """
        if self.features is None:
            raise ValueError("No features available. Call analyze() first.")
        
        clusters = None
        if self.pattern_results and 'clusters' in self.pattern_results:
            clusters = self.pattern_results['clusters']['labels']
        
        return self.explainer.explain_trade(
            trade_idx,
            self.features,
            self.baselines,
            clusters
        )
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if self.features is None:
            return {}
        
        summary = {
            'total_trades': len(self.features),
            'date_range': {
                'start': str(self.features['date'].min()),
                'end': str(self.features['date'].max())
            }
        }
        
        if 'realized_pnl' in self.features.columns:
            summary['pnl'] = {
                'total': float(self.features['realized_pnl'].sum()),
                'average': float(self.features['realized_pnl'].mean()),
                'win_rate': float((self.features['realized_pnl'] > 0).mean())
            }
        
        if 'behavioral_cluster' in self.features.columns:
            summary['clusters'] = {
                int(cluster): int((self.features['behavioral_cluster'] == cluster).sum())
                for cluster in self.features['behavioral_cluster'].unique()
            }
        
        # NEW: Add probabilistic summary
        if self.probabilistic_results and self.probabilistic_results.get('behavioral_probabilities'):
            summary['behavioral_probabilities'] = self.probabilistic_results['behavioral_probabilities']
        
        return summary
