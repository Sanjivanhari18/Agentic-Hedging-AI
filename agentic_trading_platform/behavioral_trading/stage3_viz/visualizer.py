"""Visualization components for behavioral analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, Optional
import logging
import html

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class BehavioralVisualizer:
    """Creates visualizations for behavioral analysis."""
    
    def __init__(self):
        self.figures: Dict[str, go.Figure] = {}
    
    def create_regime_timeline(self, features: pd.DataFrame, clusters: np.ndarray, 
                              change_points: list) -> go.Figure:
        """Create behavioral regime timeline."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Behavioral Regimes Over Time', 'Market Regime Overlay'),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
        fig.add_trace(
            go.Scatter(
                x=features['date'],
                y=clusters,
                mode='markers',
                marker=dict(
                    size=8,
                    color=clusters,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Behavioral Cluster")
                ),
                name='Behavioral Cluster',
                hovertemplate='Date: %{x}<br>Cluster: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        if change_points and len(change_points) > 0:
            for cp_idx in change_points:
                if cp_idx < len(features):
                    try:
                        change_date = features.iloc[cp_idx]['date']
                        fig.add_vline(
                            x=change_date,
                            line_dash="dash",
                            line_color="red",
                            line_width=2,
                            annotation_text=f"Change Point {cp_idx}",
                            annotation_position="top",
                            row=1, col=1
                        )
                        fig.add_vline(
                            x=change_date,
                            line_dash="dash",
                            line_color="red",
                            line_width=2,
                            row=2, col=1
                        )
                    except (KeyError, IndexError) as e:
                        logger.warning(f"Could not add change point at index {cp_idx}: {e}")
                        continue
        
        if 'volatility_rolling_std' in features.columns:
            fig.add_trace(
                go.Scatter(
                    x=features['date'],
                    y=features['volatility_rolling_std'],
                    mode='lines',
                    name='Volatility',
                    line=dict(color='blue', width=2),
                    hovertemplate='Date: %{x}<br>Volatility: %{y:.4f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Cluster", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", row=2, col=1)
        fig.update_layout(height=600, title_text="Behavioral Regime Timeline")
        
        self.figures['regime_timeline'] = fig
        return fig
    
    def create_deviation_plots(self, features: pd.DataFrame) -> go.Figure:
        """Create baseline vs deviation plots."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Position Size vs Volatility',
                'Trade Frequency vs RSI Regime',
                'Holding Duration Deviation',
                'Overall Deviation Score'
            )
        )
        
        if 'position_size_normalized_by_volatility' in features.columns and 'volatility_rolling_std' in features.columns:
            fig.add_trace(
                go.Scatter(
                    x=features['volatility_rolling_std'],
                    y=features['position_size_normalized_by_volatility'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=features.get('realized_pnl', 0),
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="P&L", x=1.15)
                    ),
                    name='Position Size vs Volatility',
                    hovertemplate='Volatility: %{x:.4f}<br>Position Size Ratio: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        if 'trades_per_day' in features.columns and 'rsi_14' in features.columns:
            fig.add_trace(
                go.Scatter(
                    x=features['rsi_14'],
                    y=features['trades_per_day'],
                    mode='markers',
                    marker=dict(size=6, opacity=0.6),
                    name='Trade Frequency vs RSI',
                    hovertemplate='RSI: %{x:.2f}<br>Trades/Day: %{y:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
        
        if 'holding_duration_vs_volatility' in features.columns:
            fig.add_trace(
                go.Scatter(
                    x=features['date'],
                    y=features['holding_duration_vs_volatility'],
                    mode='lines+markers',
                    name='Holding Duration vs Volatility',
                    line=dict(color='purple', width=2),
                    hovertemplate='Date: %{x}<br>Holding Duration/Volatility: %{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        if 'overall_deviation_score' in features.columns:
            fig.add_trace(
                go.Scatter(
                    x=features['date'],
                    y=features['overall_deviation_score'],
                    mode='lines+markers',
                    name='Overall Deviation',
                    line=dict(color='red', width=2),
                    hovertemplate='Date: %{x}<br>Deviation Score: %{y:.2f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        fig.update_xaxes(title_text="Volatility", row=1, col=1)
        fig.update_yaxes(title_text="Position Size Ratio", row=1, col=1)
        fig.update_xaxes(title_text="RSI (14)", row=1, col=2)
        fig.update_yaxes(title_text="Trades per Day", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Holding Duration/Volatility", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Deviation Score", row=2, col=2)
        
        fig.update_layout(height=800, title_text="Baseline Deviation Analysis")
        
        self.figures['deviation_plots'] = fig
        return fig
    
    def create_post_event_charts(self, features: pd.DataFrame) -> go.Figure:
        """Create post-event behavior charts."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Trades After Loss', 'Risk Escalation After Loss'),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5]
        )
        
        if 'is_loss' in features.columns:
            loss_mask = features['is_loss'] == 1
        elif 'realized_pnl' in features.columns:
            loss_mask = features['realized_pnl'] < 0
        else:
            loss_mask = pd.Series([False] * len(features), index=features.index)
        
        if loss_mask.any() and 'trades_after_loss' in features.columns:
            loss_data = features[loss_mask].copy()
            loss_data = loss_data[loss_data['trades_after_loss'].notna()]
            
            if len(loss_data) > 0:
                fig.add_trace(
                    go.Bar(
                        x=loss_data['date'],
                        y=loss_data['trades_after_loss'],
                        name='Trades in Next 48h',
                        marker_color='orange',
                        hovertemplate='Date: %{x}<br>Trades After: %{y}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        if loss_mask.any() and 'position_size_normalized_by_volatility' in features.columns:
            loss_data = features[loss_mask].copy()
            loss_data = loss_data[loss_data['position_size_normalized_by_volatility'].notna()]
            
            if len(loss_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=loss_data['date'],
                        y=loss_data['position_size_normalized_by_volatility'],
                        mode='markers+lines',
                        name='Position Size After Loss',
                        marker=dict(size=10, color='red', opacity=0.7),
                        line=dict(color='red', width=2),
                        hovertemplate='Date: %{x}<br>Position Size: %{y:.2f}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        fig.update_xaxes(
            title_text="Date",
            tickangle=-45,
            type='date',
            row=1, col=1
        )
        fig.update_xaxes(
            title_text="Date",
            tickangle=-45,
            type='date',
            row=2, col=1
        )
        fig.update_yaxes(title_text="Trades in Next 48h", row=1, col=1)
        fig.update_yaxes(title_text="Position Size (Normalized)", row=2, col=1)
        
        fig.update_layout(
            height=700,
            title_text="Post-Loss Behavior Analysis",
            xaxis=dict(type='date'),
            xaxis2=dict(type='date')
        )
        
        self.figures['post_event'] = fig
        return fig
    
    def create_performance_matrix(self, features: pd.DataFrame, clusters: np.ndarray, 
                                 n_clusters: int = 3) -> go.Figure:
        """Create regime performance matrix."""
        volatility_regimes = ['low', 'medium', 'high']
        trend_regimes = ['uptrend', 'downtrend', 'sideways']
        
        matrix_pnl = []
        matrix_winrate = []
        regime_labels = []
        
        has_vol_regime = 'volatility_regime' in features.columns
        has_trend_regime = 'trend_regime' in features.columns
        has_pnl = 'realized_pnl' in features.columns
        
        if not (has_vol_regime or has_trend_regime) or not has_pnl:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Performance matrix requires volatility_regime/trend_regime and realized_pnl columns",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(title="Regime Performance Matrix - Data Not Available", height=400)
            self.figures['performance_matrix'] = fig
            return fig
        
        if has_vol_regime:
            regimes_to_use = volatility_regimes
            regime_col = 'volatility_regime'
        else:
            regimes_to_use = trend_regimes
            regime_col = 'trend_regime'
        
        for regime in regimes_to_use:
            regime_mask = features[regime_col] == regime
            if not regime_mask.any():
                continue
            
            row_pnl = []
            row_winrate = []
            
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                combined_mask = regime_mask & cluster_mask
                
                if combined_mask.any():
                    avg_pnl = features.loc[combined_mask, 'realized_pnl'].mean()
                    win_rate = (features.loc[combined_mask, 'realized_pnl'] > 0).mean()
                    row_pnl.append(avg_pnl)
                    row_winrate.append(win_rate * 100)  # Convert to percentage
                else:
                    row_pnl.append(np.nan)
                    row_winrate.append(np.nan)
            
            if any(not np.isnan(p) for p in row_pnl):  # Only add if we have data
                matrix_pnl.append(row_pnl)
                matrix_winrate.append(row_winrate)
                regime_labels.append(regime.title())
        
        if matrix_pnl and len(matrix_pnl) > 0:
            cluster_labels = [f'Cluster {i}' for i in range(n_clusters)]
            
            text_matrix = []
            for i, row_pnl in enumerate(matrix_pnl):
                text_row = []
                for j, pnl in enumerate(row_pnl):
                    if not np.isnan(pnl):
                        wr = matrix_winrate[i][j] if i < len(matrix_winrate) and j < len(matrix_winrate[i]) else 0
                        text_row.append(f"P&L: ${pnl:.2f}<br>Win: {wr:.1f}%")
                    else:
                        text_row.append("")
                text_matrix.append(text_row)
            
            fig = go.Figure(data=go.Heatmap(
                z=matrix_pnl,
                x=cluster_labels,
                y=regime_labels,
                colorscale='RdYlGn',
                text=text_matrix,
                texttemplate="%{text}",
                textfont={"size": 11},
                colorbar=dict(title="Avg P&L ($)"),
                hovertemplate='Regime: %{y}<br>Cluster: %{x}<br>Avg P&L: $%{z:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Regime Performance Matrix (Average P&L by Market Regime and Behavioral Cluster)",
                xaxis_title="Behavioral Cluster",
                yaxis_title="Market Regime",
                height=max(400, len(regime_labels) * 80)
            )
            
            self.figures['performance_matrix'] = fig
            return fig
        
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for performance matrix",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(title="Regime Performance Matrix", height=400)
        self.figures['performance_matrix'] = fig
        return fig
    
    def create_cluster_timeline(self, features: pd.DataFrame, clusters: np.ndarray) -> go.Figure:
        """
        Create a line graph showing trades over time with clusters highlighted/circled.
        
        Args:
            features: DataFrame with trade features
            clusters: Cluster labels for each trade
            
        Returns:
            Plotly figure with cluster visualization
        """
        fig = go.Figure()
        
        unique_clusters = np.unique(clusters)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_data = features.loc[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
            
            # Sort by date
            cluster_data = cluster_data.sort_values('date')
            
            # Use price or cumulative P&L for y-axis
            if 'price' in cluster_data.columns:
                y_values = cluster_data['price']
                y_label = 'Trade Price'
            elif 'realized_pnl' in cluster_data.columns:
                y_values = cluster_data['realized_pnl'].cumsum()
                y_label = 'Cumulative P&L'
            else:
                y_values = range(len(cluster_data))
                y_label = 'Trade Index'
            
            # Plot line for this cluster
            fig.add_trace(go.Scatter(
                x=cluster_data['date'],
                y=y_values,
                mode='lines+markers',
                name=f'Cluster {cluster_id}',
                line=dict(
                    color=colors[cluster_id % len(colors)],
                    width=3
                ),
                marker=dict(
                    size=12,
                    color=colors[cluster_id % len(colors)],
                    line=dict(width=2, color='white'),
                    symbol='circle'
                ),
                hovertemplate=f'<b>Cluster {cluster_id}</b><br>' +
                            'Date: %{x}<br>' +
                            f'{y_label}: %{{y:.2f}}<br>' +
                            '<extra></extra>'
            ))
            
            # Add circles/highlights around cluster points
            fig.add_trace(go.Scatter(
                x=cluster_data['date'],
                y=y_values,
                mode='markers',
                name=f'Cluster {cluster_id} Highlight',
                marker=dict(
                    size=20,
                    color=colors[cluster_id % len(colors)],
                    opacity=0.2,
                    line=dict(width=3, color=colors[cluster_id % len(colors)])
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add annotations for cluster labels at first occurrence
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_data = features.loc[cluster_mask]
            
            if len(cluster_data) > 0:
                cluster_data = cluster_data.sort_values('date')
                first_trade = cluster_data.iloc[0]
                
                if 'price' in first_trade:
                    y_val = first_trade['price']
                elif 'realized_pnl' in features.columns:
                    y_val = features.loc[:first_trade.name, 'realized_pnl'].sum() if first_trade.name > 0 else first_trade.get('realized_pnl', 0)
                else:
                    y_val = 0
                
                fig.add_annotation(
                    x=first_trade['date'],
                    y=y_val,
                    text=f'<b>Cluster {cluster_id}</b>',
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=colors[cluster_id % len(colors)],
                    bgcolor='white',
                    bordercolor=colors[cluster_id % len(colors)],
                    borderwidth=2,
                    font=dict(size=12, color=colors[cluster_id % len(colors)])
                )
        
        fig.update_layout(
            title='Behavioral Clusters Over Time',
            xaxis_title='Date',
            yaxis_title=y_label,
            hovermode='closest',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        self.figures['cluster_timeline'] = fig
        return fig
    
    def create_trade_journey_timeline(self, features: pd.DataFrame) -> go.Figure:
        """
        Create a user-friendly trade journey timeline showing each trade's entry to exit journey.
        
        This visualization helps amateur investors easily see:
        - When they entered and exited each trade
        - Profit/loss for each trade
        - Which stocks they traded
        - Visual journey from entry to exit
        """
        fig = go.Figure()
        
        # Sort by date
        df = features.sort_values('date').copy()
        
        # Group trades by symbol for better visualization
        if 'symbol' in df.columns:
            symbols = df['symbol'].unique()
            colors = px.colors.qualitative.Set3[:len(symbols)]
            symbol_colors = dict(zip(symbols, colors))
        else:
            symbol_colors = {}
        
        # Create entry and exit points
        entry_dates = []
        exit_dates = []
        entry_prices = []
        exit_prices = []
        pnl_values = []
        symbols_list = []
        trade_labels = []
        
        # Process buy-sell pairs - match by reconstructing positions
        if 'side' in df.columns and 'symbol' in df.columns:
            buy_trades = df[df['side'] == 'buy'].copy()
            sell_trades = df[df['side'] == 'sell'].copy()
            
            # Match buys with sells by symbol and date order
            for idx, buy in buy_trades.iterrows():
                symbol = buy.get('symbol', 'UNKNOWN')
                buy_date = buy['date']
                buy_price = buy.get('price', 0)
                buy_qty = buy.get('quantity', 0)
                
                # Find corresponding sell (same symbol, after buy date)
                matching_sells = sell_trades[
                    (sell_trades['symbol'] == symbol) & 
                    (sell_trades['date'] > buy_date)
                ].sort_values('date')
                
                if len(matching_sells) > 0:
                    # Take first matching sell
                    sell = matching_sells.iloc[0]
                    sell_date = sell['date']
                    sell_price = sell.get('price', 0)
                    sell_qty = sell.get('quantity', 0)
                    
                    # Calculate P&L
                    if 'realized_pnl' in sell:
                        pnl = sell['realized_pnl']
                    else:
                        # Estimate P&L
                        qty = min(buy_qty, sell_qty)
                        pnl = (sell_price - buy_price) * qty
                    
                    entry_dates.append(buy_date)
                    exit_dates.append(sell_date)
                    entry_prices.append(buy_price)
                    exit_prices.append(sell_price)
                    pnl_values.append(pnl)
                    symbols_list.append(symbol)
                    trade_labels.append(f"{symbol}<br>Entry: ${buy_price:.2f}<br>Exit: ${sell_price:.2f}<br>P&L: ${pnl:.2f}")
        else:
            # Fallback: use all trades as individual points
            for idx, row in df.iterrows():
                entry_dates.append(row['date'])
                exit_dates.append(row['date'])
                entry_prices.append(row.get('price', 0))
                exit_prices.append(row.get('price', 0))
                pnl_values.append(row.get('realized_pnl', 0))
                symbols_list.append(row.get('symbol', 'UNKNOWN'))
        
        # Add entry points (green)
        if entry_dates:
            fig.add_trace(go.Scatter(
                x=entry_dates,
                y=entry_prices,
                mode='markers',
                name='Entry Points',
                marker=dict(
                    size=12,
                    color='green',
                    symbol='triangle-up',
                    line=dict(width=2, color='darkgreen')
                ),
                text=[f"Entry: {s}<br>Price: ${p:.2f}" for s, p in zip(symbols_list, entry_prices)],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Add exit points (color by P&L)
        if exit_dates:
            exit_colors = ['red' if pnl < 0 else 'green' for pnl in pnl_values]
            fig.add_trace(go.Scatter(
                x=exit_dates,
                y=exit_prices,
                mode='markers',
                name='Exit Points',
                marker=dict(
                    size=12,
                    color=exit_colors,
                    symbol='triangle-down',
                    line=dict(width=2, color='darkred')
                ),
                text=[f"Exit: {s}<br>Price: ${p:.2f}<br>P&L: ${pnl:.2f}" 
                      for s, p, pnl in zip(symbols_list, exit_prices, pnl_values)],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Add journey lines (connecting entry to exit)
        for i in range(len(entry_dates)):
            line_color = 'red' if pnl_values[i] < 0 else 'green'
            line_width = 2 if abs(pnl_values[i]) > 100 else 1
            fig.add_trace(go.Scatter(
                x=[entry_dates[i], exit_dates[i]],
                y=[entry_prices[i], exit_prices[i]],
                mode='lines',
                name=f'Trade {i+1}',
                line=dict(color=line_color, width=line_width, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add P&L annotations
        for i in range(len(exit_dates)):
            annotation_text = f"${pnl_values[i]:.2f}"
            fig.add_annotation(
                x=exit_dates[i],
                y=exit_prices[i],
                text=annotation_text,
                showarrow=True,
                arrowhead=2,
                arrowcolor='red' if pnl_values[i] < 0 else 'green',
                bgcolor='white',
                bordercolor='red' if pnl_values[i] < 0 else 'green',
                borderwidth=1,
                font=dict(size=10, color='red' if pnl_values[i] < 0 else 'green')
            )
        
        fig.update_layout(
            title='Trade Journey Timeline - Your Trading Path',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=700,
            hovermode='closest',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        self.figures['trade_journey'] = fig
        return fig
    
    def create_signal_following_scorecard(self, features: pd.DataFrame) -> go.Figure:
        """
        Create a user-friendly scorecard showing how well the investor followed technical signals.
        
        This helps amateur investors understand:
        - Did they follow RSI signals?
        - Did they follow MACD signals?
        - Did they follow EMA trend signals?
        - Visual indicators of missed opportunities
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'RSI Signal Following', 'MACD Signal Following',
                'EMA Trend Following', 'Overall Signal Score',
                'Missed Opportunities', 'Signal Adherence Summary'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "table"}]],
            vertical_spacing=0.25,  # Increased spacing to prevent text overlap
            row_heights=[0.35, 0.35, 0.30],  # Adjusted row heights - more space for row 1
            horizontal_spacing=0.15
        )
        
        # RSI Signal Analysis
        if 'rsi_14' in features.columns:
            # Overbought signals (RSI > 70) - should sell
            overbought_mask = features['rsi_14'] > 70
            overbought_trades = features[overbought_mask]
            
            # Oversold signals (RSI < 30) - should buy
            oversold_mask = features['rsi_14'] < 30
            oversold_trades = features[oversold_mask]
            
            # Count actual trades at these signals
            overbought_sells = len(overbought_trades[overbought_trades.get('side', '') == 'sell']) if 'side' in features.columns else 0
            oversold_buys = len(oversold_trades[oversold_trades.get('side', '') == 'buy']) if 'side' in features.columns else 0
            
            fig.add_trace(go.Bar(
                x=['Overbought Signals<br>(Should Sell)', 'Oversold Signals<br>(Should Buy)'],
                y=[len(overbought_trades), len(oversold_trades)],
                name='Total Signals',
                marker_color='lightblue',
                text=[len(overbought_trades), len(oversold_trades)],
                textposition='outside'
            ), row=1, col=1)
            
            fig.add_trace(go.Bar(
                x=['Overbought Signals<br>(Should Sell)', 'Oversold Signals<br>(Should Buy)'],
                y=[overbought_sells, oversold_buys],
                name='Trades Taken',
                marker_color='green',
                text=[overbought_sells, oversold_buys],
                textposition='outside'
            ), row=1, col=1)
        
        # MACD Signal Analysis
        if 'macd_line' in features.columns and 'macd_signal' in features.columns:
            # MACD bullish: macd_line > macd_signal (should buy)
            # MACD bearish: macd_line < macd_signal (should sell)
            bullish_mask = features['macd_line'] > features['macd_signal']
            bearish_mask = features['macd_line'] < features['macd_signal']
            
            bullish_trades = features[bullish_mask]
            bearish_trades = features[bearish_mask]
            
            bullish_buys = len(bullish_trades[bullish_trades.get('side', '') == 'buy']) if 'side' in features.columns else 0
            bearish_sells = len(bearish_trades[bearish_trades.get('side', '') == 'sell']) if 'side' in features.columns else 0
            
            fig.add_trace(go.Bar(
                x=['Bullish Signals<br>(Should Buy)', 'Bearish Signals<br>(Should Sell)'],
                y=[len(bullish_trades), len(bearish_trades)],
                name='Total Signals',
                marker_color='lightblue',
                showlegend=False,
                text=[len(bullish_trades), len(bearish_trades)],
                textposition='outside'
            ), row=1, col=2)
            
            fig.add_trace(go.Bar(
                x=['Bullish Signals<br>(Should Buy)', 'Bearish Signals<br>(Should Sell)'],
                y=[bullish_buys, bearish_sells],
                name='Trades Taken',
                marker_color='green',
                showlegend=False,
                text=[bullish_buys, bearish_sells],
                textposition='outside'
            ), row=1, col=2)
        
        # EMA Trend Following
        if 'trend_regime' in features.columns and 'side' in features.columns:
            uptrend_trades = features[features['trend_regime'] == 'uptrend']
            downtrend_trades = features[features['trend_regime'] == 'downtrend']
            
            uptrend_buys = len(uptrend_trades[uptrend_trades['side'] == 'buy'])
            downtrend_sells = len(downtrend_trades[downtrend_trades['side'] == 'sell'])
            
            fig.add_trace(go.Bar(
                x=['Uptrend<br>(Should Buy)', 'Downtrend<br>(Should Sell)'],
                y=[len(uptrend_trades), len(downtrend_trades)],
                name='Total Signals',
                marker_color='lightblue',
                showlegend=False,
                text=[len(uptrend_trades), len(downtrend_trades)],
                textposition='outside'
            ), row=2, col=1)
            
            fig.add_trace(go.Bar(
                x=['Uptrend<br>(Should Buy)', 'Downtrend<br>(Should Sell)'],
                y=[uptrend_buys, downtrend_sells],
                name='Trades Taken',
                marker_color='green',
                showlegend=False,
                text=[uptrend_buys, downtrend_sells],
                textposition='outside'
            ), row=2, col=1)
        
        # Overall Signal Score (Gauge)
        total_signals = 0
        signals_followed = 0
        
        if 'rsi_14' in features.columns:
            total_signals += len(overbought_trades) + len(oversold_trades)
            signals_followed += overbought_sells + oversold_buys
        
        if 'macd_line' in features.columns:
            total_signals += len(bullish_trades) + len(bearish_trades)
            signals_followed += bullish_buys + bearish_sells
        
        if 'trend_regime' in features.columns:
            total_signals += len(uptrend_trades) + len(downtrend_trades)
            signals_followed += uptrend_buys + downtrend_sells
        
        signal_score = (signals_followed / total_signals * 100) if total_signals > 0 else 0
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=signal_score,
            domain={'x': [0.1, 0.9], 'y': [0.15, 0.85]},  # Reduced domain to add padding and prevent overlap
            title={'text': "Signal Following<br>Score (%)", 'font': {'size': 14}},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen" if signal_score >= 70 else "orange" if signal_score >= 50 else "red"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ), row=2, col=2)
        
        # Missed Opportunities
        missed_opportunities = []
        if 'rsi_14' in features.columns:
            missed_overbought = len(overbought_trades) - overbought_sells
            missed_oversold = len(oversold_trades) - oversold_buys
            if missed_overbought > 0:
                missed_opportunities.append(['RSI Overbought', missed_overbought, 'Missed Sell Signal'])
            if missed_oversold > 0:
                missed_opportunities.append(['RSI Oversold', missed_oversold, 'Missed Buy Signal'])
        
        if 'macd_line' in features.columns:
            missed_bullish = len(bullish_trades) - bullish_buys
            missed_bearish = len(bearish_trades) - bearish_sells
            if missed_bullish > 0:
                missed_opportunities.append(['MACD Bullish', missed_bullish, 'Missed Buy Signal'])
            if missed_bearish > 0:
                missed_opportunities.append(['MACD Bearish', missed_bearish, 'Missed Sell Signal'])
        
        if missed_opportunities:
            opp_df = pd.DataFrame(missed_opportunities, columns=['Signal Type', 'Count', 'Action'])
            fig.add_trace(go.Bar(
                x=opp_df['Signal Type'],
                y=opp_df['Count'],
                marker_color='orange',
                text=opp_df['Count'],
                textposition='outside',
                name='Missed Opportunities'
            ), row=3, col=1)
        
        # Summary Table
        summary_data = []
        if 'rsi_14' in features.columns:
            rsi_score = ((overbought_sells + oversold_buys) / (len(overbought_trades) + len(oversold_trades)) * 100) if (len(overbought_trades) + len(oversold_trades)) > 0 else 0
            summary_data.append(['RSI Signals', f"{signals_followed}/{total_signals}", f"{rsi_score:.1f}%"])
        
        if 'macd_line' in features.columns:
            macd_score = ((bullish_buys + bearish_sells) / (len(bullish_trades) + len(bearish_trades)) * 100) if (len(bullish_trades) + len(bearish_trades)) > 0 else 0
            summary_data.append(['MACD Signals', f"{bullish_buys + bearish_sells}/{len(bullish_trades) + len(bearish_trades)}", f"{macd_score:.1f}%"])
        
        if 'trend_regime' in features.columns:
            trend_score = ((uptrend_buys + downtrend_sells) / (len(uptrend_trades) + len(downtrend_trades)) * 100) if (len(uptrend_trades) + len(downtrend_trades)) > 0 else 0
            summary_data.append(['EMA Trend', f"{uptrend_buys + downtrend_sells}/{len(uptrend_trades) + len(downtrend_trades)}", f"{trend_score:.1f}%"])
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data, columns=['Signal Type', 'Followed/Total', 'Score'])
            fig.add_trace(go.Table(
                header=dict(values=list(summary_df.columns),
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=[summary_df[col] for col in summary_df.columns],
                          fill_color='lavender',
                          align='left')
            ), row=3, col=2)
        
        fig.update_layout(
            height=1300,  # Increased height to accommodate spacing
            title_text="Signal Following Scorecard - How Well Did You Follow Technical Indicators?",
            showlegend=True,
            margin=dict(t=120, b=80, l=60, r=60)  # Increased bottom margin to prevent overlap
        )
        
        # Add extra bottom margin to MACD chart to prevent overlap with gauge
        fig.update_layout(
            xaxis2=dict(
                automargin=True,
                tickangle=-45
            )
        )
        
        # Add explanation annotation for gauge calculation (positioned below gauge)
        if total_signals > 0:
            fig.add_annotation(
                text=f"<b>Gauge Calculation:</b><br>" +
                     f"Score = (Trades Following Signals / Total Signals) × 100<br>" +
                     f"Your Score: {signals_followed}/{total_signals} = {signal_score:.1f}%<br>" +
                     f"<b>Interpretation:</b><br>" +
                     f"• 70-100%: Excellent signal following<br>" +
                     f"• 50-70%: Good signal following<br>" +
                     f"• 0-50%: Needs improvement",
                xref="paper", yref="paper",
                x=0.75, y=0.10,  # Positioned lower to avoid overlap
                xanchor="left",
                yanchor="top",
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=9, color="black"),
                row=2, col=2
            )
        
        # Update axes with proper spacing
        fig.update_xaxes(title_text="Signal Type", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(
            title_text="Signal Type", 
            row=1, col=2,
            tickangle=-45,  # Angle labels to prevent overlap
            automargin=True  # Automatically adjust margins
        )
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_xaxes(title_text="Trend Type", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Signal Type", row=3, col=1)
        fig.update_yaxes(title_text="Missed Count", row=3, col=1)
        
        self.figures['signal_scorecard'] = fig
        return fig
    
    def create_stability_scorecard(self, stability_results: Dict) -> go.Figure:
        """
        Create a visualization for the Behavioral Stability / Consistency Score.
        
        This score measures how consistent a trader's behavior is over time.
        It does NOT measure skill or profitability - only consistency.
        
        Args:
            stability_results: Dictionary from BehavioralStabilityAnalyzer.calculate_stability_score()
            
        Returns:
            Plotly figure with stability score visualization
        """
        from plotly.subplots import make_subplots
        
        if stability_results.get('stability_score') is None:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient data to calculate behavioral stability score.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(title="Behavioral Stability Score - Data Not Available", height=400)
            self.figures['stability_scorecard'] = fig
            return fig
        
        score = stability_results['stability_score']
        components = stability_results.get('components', {})
        
        # Create subplots: gauge at top, component breakdown below
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Behavioral Stability Score', 'Component Breakdown'),
            vertical_spacing=0.25,
            row_heights=[0.5, 0.5],
            specs=[[{"type": "indicator"}], [{"type": "bar"}]]
        )
        
        # Main gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Behavioral Stability<br>Consistency Index"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen" if score >= 70 else "orange" if score >= 50 else "red"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 50], 'color': "gray"},
                    {'range': [50, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ), row=1, col=1)
        
        # Component breakdown bar chart
        if components:
            comp_names = []
            comp_scores = []
            for comp_name, comp_data in components.items():
                if comp_data.get('stability_score') is not None:
                    comp_names.append(comp_data.get('feature_name', comp_name))
                    comp_scores.append(comp_data['stability_score'])
            
            if comp_names:
                fig.add_trace(go.Bar(
                    x=comp_names,
                    y=comp_scores,
                    marker_color=['green' if s >= 70 else 'orange' if s >= 50 else 'red' for s in comp_scores],
                    text=[f"{s:.1f}" for s in comp_scores],
                    textposition='outside',
                    name='Component Stability'
                ), row=2, col=1)
        
        # Add interpretation text
        interpretation = stability_results.get('interpretation', '')
        note = stability_results.get('note', '')
        
        fig.add_annotation(
            text=f"<b>Interpretation:</b><br>{interpretation.split('Overall Assessment:')[1].split('Component Analysis:')[0] if 'Overall Assessment:' in interpretation else ''}<br><br><b>Note:</b> {note}",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            xanchor="center",
            yanchor="top",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
        
        fig.update_xaxes(title_text="Behavioral Component", row=2, col=1)
        fig.update_yaxes(title_text="Stability Score (0-100)", row=2, col=1, range=[0, 100])
        
        fig.update_layout(
            height=800,
            title_text="Behavioral Stability / Consistency Score<br><sub>This score measures consistency of behavior, not skill or profitability</sub>",
            showlegend=False
        )
        
        self.figures['stability_scorecard'] = fig
        return fig
    
    def create_cluster_scatter_plot(self, features: pd.DataFrame, clusters: np.ndarray, 
                                    cluster_centers: np.ndarray, cluster_analysis: Dict) -> go.Figure:
        """
        Create a 2D scatter plot showing behavioral clusters with circles around each cluster.
        
        Args:
            features: DataFrame with trade features
            clusters: Cluster labels for each trade
            cluster_centers: Cluster center coordinates in original feature space
            cluster_analysis: Dictionary with cluster analysis information
            
        Returns:
            Plotly figure with cluster scatter plot
        """
        from sklearn.preprocessing import StandardScaler
        
        # Get feature columns (same as used in clustering)
        exclude = ['date', 'symbol', 'side', 'price', 'quantity', 'rsi_14', 'ema_20', 
                  'ema_50', 'Close', 'Date', 'date_only', 'trend_regime', 'volatility_regime']
        
        feature_cols = [col for col in features.columns 
                       if col not in exclude and features[col].dtype in [np.float64, np.int64]]
        
        # Limit to top 20 features
        priority_features = [
            'trades_per_day', 'trades_per_rolling_7days', 'trades_per_rolling_30days',
            'position_size_dollar_value', 'position_size_normalized_by_volatility',
            'holding_duration_days', 'holding_duration_vs_volatility',
            'time_gap_hours_since_last_trade', 'time_gap_days_since_last_trade',
            'entry_price_distance_from_ema20', 'entry_price_distance_from_ema50',
            'exit_price_distance_from_ema20', 'exit_price_distance_from_ema50'
        ]
        
        ordered = [f for f in priority_features if f in feature_cols]
        ordered.extend([f for f in feature_cols if f not in ordered])
        feature_cols = ordered[:20]
        
        # Prepare feature matrix
        X = features[feature_cols].fillna(0).values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_scaled)
        
        # Transform cluster centers to 2D
        cluster_centers_scaled = scaler.transform(cluster_centers)
        centers_2d = pca.transform(cluster_centers_scaled)
        
        # Shift all points to positive quadrant (first quadrant only)
        # Find minimum values and shift
        min_x = np.min(X_2d[:, 0])
        min_y = np.min(X_2d[:, 1])
        min_center_x = np.min(centers_2d[:, 0])
        min_center_y = np.min(centers_2d[:, 1])
        
        # Use the most negative value to shift everything to positive
        shift_x = abs(min(min_x, min_center_x)) + 0.1  # Add small padding
        shift_y = abs(min(min_y, min_center_y)) + 0.1  # Add small padding
        
        # Shift all points to positive quadrant
        X_2d[:, 0] = X_2d[:, 0] + shift_x
        X_2d[:, 1] = X_2d[:, 1] + shift_y
        centers_2d[:, 0] = centers_2d[:, 0] + shift_x
        centers_2d[:, 1] = centers_2d[:, 1] + shift_y
        
        # Define colors for clusters
        cluster_colors = {
            0: '#FF6B6B',  # Red
            1: '#4ECDC4',  # Blue
            2: '#95E1D3',  # Green
            3: '#FFE66D',  # Yellow
            4: '#A8E6CF'   # Light Green
        }
        
        fig = go.Figure()
        
        # Calculate circle radii for each cluster
        unique_clusters = np.unique(clusters)
        cluster_radii = {}
        
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_points = X_2d[cluster_mask]
            
            if len(cluster_points) > 0:
                center = centers_2d[cluster_id]
                # Calculate maximum distance from center to any point in cluster
                distances = np.sqrt(np.sum((cluster_points - center) ** 2, axis=1))
                max_distance = np.max(distances) if len(distances) > 0 else 1.0
                # Add 10% padding
                cluster_radii[cluster_id] = max_distance * 1.1
        
        # Draw cluster circles (background layer)
        for cluster_id in unique_clusters:
            center = centers_2d[cluster_id]
            radius = cluster_radii.get(cluster_id, 1.0)
            color = cluster_colors.get(cluster_id % len(cluster_colors), '#CCCCCC')
            
            # Create circle using scatter plot with many points
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = center[0] + radius * np.cos(theta)
            circle_y = center[1] + radius * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='lines',
                name=f'Cluster {cluster_id} Area',
                line=dict(color=color, width=2, dash='dash'),
                fill='toself',
                fillcolor=color,
                opacity=0.15,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Plot data points colored by cluster
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_points = X_2d[cluster_mask]
            cluster_features = features.loc[cluster_mask]
            
            color = cluster_colors.get(cluster_id % len(cluster_colors), '#CCCCCC')
            
            # Get cluster info
            cluster_info = cluster_analysis.get(f'cluster_{cluster_id}', {})
            cluster_size = cluster_info.get('size', cluster_mask.sum())
            avg_pnl = cluster_info.get('avg_pnl', 0)
            win_rate = cluster_info.get('win_rate', 0) * 100
            
            # Create hover text
            hover_texts = []
            for idx, (_, row) in enumerate(cluster_features.iterrows()):
                hover_text = f"<b>Cluster {cluster_id}</b><br>"
                if 'date' in row:
                    hover_text += f"Date: {row['date']}<br>"
                if 'symbol' in row:
                    hover_text += f"Symbol: {row.get('symbol', 'N/A')}<br>"
                if 'realized_pnl' in row:
                    hover_text += f"P&L: ${row['realized_pnl']:.2f}<br>"
                hover_text += f"Cluster: {cluster_id}"
                hover_texts.append(hover_text)
            
            fig.add_trace(go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                name=f'Cluster {cluster_id} ({cluster_size} trades)',
                marker=dict(
                    size=8,
                    color=color,
                    line=dict(width=1, color='white'),
                    opacity=0.7
                ),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
                legendgroup=f'cluster_{cluster_id}'
            ))
        
        # Plot cluster centers (large, distinct markers)
        for cluster_id in unique_clusters:
            center = centers_2d[cluster_id]
            color = cluster_colors.get(cluster_id % len(cluster_colors), '#CCCCCC')
            
            cluster_info = cluster_analysis.get(f'cluster_{cluster_id}', {})
            avg_pnl = cluster_info.get('avg_pnl', 0)
            win_rate = cluster_info.get('win_rate', 0) * 100
            
            fig.add_trace(go.Scatter(
                x=[center[0]],
                y=[center[1]],
                mode='markers+text',
                name=f'Cluster {cluster_id} Center',
                marker=dict(
                    size=25,
                    color=color,
                    symbol='star',
                    line=dict(width=2, color='white'),
                    opacity=0.9
                ),
                text=[f'C{cluster_id}'],
                textposition='middle center',
                textfont=dict(size=12, color='white', family='Arial Black'),
                hovertemplate=(
                    f'<b>Cluster {cluster_id} Center</b><br>'
                    f'Avg P&L: ${avg_pnl:.2f}<br>'
                    f'Win Rate: {win_rate:.1f}%<br>'
                    f'Size: {cluster_info.get("size", 0)} trades<extra></extra>'
                ),
                legendgroup=f'cluster_{cluster_id}',
                showlegend=False
            ))
        
        # Calculate explained variance
        explained_var = pca.explained_variance_ratio_
        total_variance = explained_var.sum() * 100
        
        # Update layout - ensure axes start at 0 (positive quadrant only)
        fig.update_layout(
            title=dict(
                text=f'Behavioral Cluster Scatter Plot (Positive Quadrant)<br><sub>2D projection (explains {total_variance:.1f}% of variance)</sub>',
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='Principal Component 1',
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                range=[0, None]  # Start at 0 (positive quadrant)
            ),
            yaxis=dict(
                title='Principal Component 2',
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                range=[0, None]  # Start at 0 (positive quadrant)
            ),
            hovermode='closest',
            height=800,
            width=1200,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        self.figures['cluster_scatter'] = fig
        return fig
    
    def create_volatility_trade_timeline(self, features: pd.DataFrame, 
                                        market_data_fetcher=None) -> go.Figure:
        """
        Create volatility timeline visualization showing volatility over time with buy/sell markers.
        
        For each stock traded:
        - Line graph showing volatility over time (full timeline)
        - Green dots: Buy trades
        - Red dots: Sell trades
        
        Args:
            features: DataFrame with trade features including date, symbol, side, volatility
            market_data_fetcher: Optional MarketDataFetcher instance to get full volatility data
            
        Returns:
            Plotly figure with volatility timeline
        """
        if 'symbol' not in features.columns or 'date' not in features.columns:
            logger.warning("Missing 'symbol' or 'date' columns. Cannot create volatility timeline.")
            return go.Figure()
        
        if 'volatility_rolling_std' not in features.columns:
            logger.warning("Missing 'volatility_rolling_std' column. Cannot create volatility timeline.")
            return go.Figure()
        
        # Get unique symbols
        symbols = features['symbol'].dropna().unique()
        symbols = sorted([s for s in symbols if pd.notna(s)])
        
        if len(symbols) == 0:
            logger.warning("No symbols found. Cannot create volatility timeline.")
            return go.Figure()
        
        # Calculate number of rows for subplots (max 3 per row)
        n_symbols = len(symbols)
        n_rows = (n_symbols + 2) // 3  # Round up
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows,
            cols=3,
            subplot_titles=[f'{symbol} - Volatility & Trades' for symbol in symbols],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Color palette for different symbols (if needed for line)
        colors = px.colors.qualitative.Set3
        
        for idx, symbol in enumerate(symbols):
            row = (idx // 3) + 1
            col = (idx % 3) + 1
            
            # Get data for this symbol
            symbol_data = features[features['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date')
            
            if len(symbol_data) == 0:
                continue
            
            # Get date range for this symbol
            min_date = symbol_data['date'].min()
            max_date = symbol_data['date'].max()
            
            # Try to get full volatility data if market_data_fetcher is available
            volatility_timeline = None
            if market_data_fetcher is not None:
                try:
                    # Fetch full OHLCV data for the symbol
                    ohlcv = market_data_fetcher.fetch_ohlcv(
                        symbol, 
                        start_date=min_date.strftime('%Y-%m-%d') if hasattr(min_date, 'strftime') else str(min_date)[:10],
                        end_date=max_date.strftime('%Y-%m-%d') if hasattr(max_date, 'strftime') else str(max_date)[:10]
                    )
                    if 'volatility_rolling_std' in ohlcv.columns:
                        volatility_timeline = ohlcv[['Date', 'volatility_rolling_std']].copy()
                        volatility_timeline = volatility_timeline.rename(columns={'Date': 'date'})
                except Exception as e:
                    logger.warning(f"Could not fetch full volatility data for {symbol}: {e}")
            
            # If we don't have full timeline, use trade dates only
            if volatility_timeline is None or len(volatility_timeline) == 0:
                volatility_timeline = symbol_data[['date', 'volatility_rolling_std']].dropna()
            
            if len(volatility_timeline) == 0:
                continue
            
            # Sort by date
            volatility_timeline = volatility_timeline.sort_values('date')
            
            # Plot volatility line
            fig.add_trace(
                go.Scatter(
                    x=volatility_timeline['date'],
                    y=volatility_timeline['volatility_rolling_std'],
                    mode='lines',
                    name=f'{symbol} Volatility',
                    line=dict(color='#2E86AB', width=2),
                    hovertemplate=f'<b>{symbol}</b><br>Date: %{{x}}<br>Volatility: %{{y:.4f}}<extra></extra>',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Separate buy and sell trades
            buy_trades = symbol_data[symbol_data.get('side', '') == 'buy']
            sell_trades = symbol_data[symbol_data.get('side', '') == 'sell']
            
            # Plot buy trades (green dots)
            if len(buy_trades) > 0:
                buy_volatility = buy_trades[['date', 'volatility_rolling_std', 'price', 'quantity']].dropna(subset=['volatility_rolling_std'])
                
                if len(buy_volatility) > 0:
                    hover_texts_buy = []
                    for _, trade in buy_volatility.iterrows():
                        hover_text = f"<b>BUY {symbol}</b><br>"
                        hover_text += f"Date: {trade['date']}<br>"
                        hover_text += f"Volatility: {trade['volatility_rolling_std']:.4f}<br>"
                        if 'price' in trade:
                            hover_text += f"Price: ${trade['price']:.2f}<br>"
                        if 'quantity' in trade:
                            hover_text += f"Quantity: {trade['quantity']}"
                        hover_texts_buy.append(hover_text)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=buy_volatility['date'],
                            y=buy_volatility['volatility_rolling_std'],
                            mode='markers',
                            name=f'{symbol} Buy',
                            marker=dict(
                                color='#28A745',  # Green
                                size=10,
                                symbol='circle',
                                line=dict(width=1, color='white')
                            ),
                            text=hover_texts_buy,
                            hovertemplate='%{text}<extra></extra>',
                            showlegend=False
                        ),
                        row=row, col=col
                    )
            
            # Plot sell trades (red dots)
            if len(sell_trades) > 0:
                sell_volatility = sell_trades[['date', 'volatility_rolling_std', 'price', 'quantity', 'realized_pnl']].dropna(subset=['volatility_rolling_std'])
                
                if len(sell_volatility) > 0:
                    hover_texts_sell = []
                    for _, trade in sell_volatility.iterrows():
                        hover_text = f"<b>SELL {symbol}</b><br>"
                        hover_text += f"Date: {trade['date']}<br>"
                        hover_text += f"Volatility: {trade['volatility_rolling_std']:.4f}<br>"
                        if 'price' in trade:
                            hover_text += f"Price: ${trade['price']:.2f}<br>"
                        if 'quantity' in trade:
                            hover_text += f"Quantity: {trade['quantity']}<br>"
                        if 'realized_pnl' in trade and pd.notna(trade['realized_pnl']):
                            hover_text += f"P&L: ${trade['realized_pnl']:.2f}"
                        hover_texts_sell.append(hover_text)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=sell_volatility['date'],
                            y=sell_volatility['volatility_rolling_std'],
                            mode='markers',
                            name=f'{symbol} Sell',
                            marker=dict(
                                color='#DC3545',  # Red
                                size=10,
                                symbol='circle',
                                line=dict(width=1, color='white')
                            ),
                            text=hover_texts_sell,
                            hovertemplate='%{text}<extra></extra>',
                            showlegend=False
                        ),
                        row=row, col=col
                    )
            
            # Update axes for this subplot
            fig.update_xaxes(title_text="Date", row=row, col=col)
            fig.update_yaxes(title_text="Volatility", row=row, col=col)
        
        # Add legend for buy/sell markers (only once)
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                name='Buy',
                marker=dict(color='#28A745', size=10),
                showlegend=True
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                name='Sell',
                marker=dict(color='#DC3545', size=10),
                showlegend=True
            )
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Volatility Timeline with Trade Markers<br><sub>Green dots = Buy, Red dots = Sell</sub>',
                x=0.5,
                xanchor='center'
            ),
            height=300 * n_rows,
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        self.figures['volatility_timeline'] = fig
        return fig
    
    def create_sp500_volatility_timeline(self, features: pd.DataFrame, 
                                        market_data_fetcher=None) -> go.Figure:
        """
        Create S&P 500 volatility timeline with buy/sell trade markers.
        
        Shows:
        - S&P 500 volatility over time (line graph, normalized 0-100)
        - Green dots: Buy trades with stock name labels
        - Red dots: Sell trades with stock name labels
        
        Args:
            features: DataFrame with trade features including date, symbol, side
            market_data_fetcher: Optional MarketDataFetcher instance to get S&P 500 data
            
        Returns:
            Plotly figure with S&P 500 volatility timeline
        """
        if 'date' not in features.columns or 'symbol' not in features.columns:
            logger.warning("Missing 'date' or 'symbol' columns. Cannot create S&P 500 volatility timeline.")
            return go.Figure()
        
        # Get date range from trades
        min_date = features['date'].min()
        max_date = features['date'].max()
        
        # Convert dates to strings if needed
        if hasattr(min_date, 'strftime'):
            start_date = min_date.strftime('%Y-%m-%d')
            end_date = max_date.strftime('%Y-%m-%d')
        else:
            start_date = str(min_date)[:10]
            end_date = str(max_date)[:10]
        
        # Fetch S&P 500 data
        sp500_data = None
        if market_data_fetcher is not None:
            try:
                # Fetch S&P 500 data (^GSPC is the Yahoo Finance symbol)
                sp500_ohlcv = market_data_fetcher.fetch_ohlcv('^GSPC', start_date, end_date)
                
                # Calculate volatility
                sp500_ohlcv = market_data_fetcher.calculate_market_indicators(sp500_ohlcv)
                
                if 'volatility_rolling_std' in sp500_ohlcv.columns:
                    # Handle date column - yfinance returns date as index
                    sp500_data = sp500_ohlcv.copy()
                    if sp500_data.index.name == 'Date' or isinstance(sp500_data.index, pd.DatetimeIndex):
                        sp500_data = sp500_data.reset_index()
                        # Find date column (could be 'Date' or index name)
                        date_col = None
                        for col in ['Date', 'date', sp500_data.index.name]:
                            if col in sp500_data.columns:
                                date_col = col
                                break
                        if date_col:
                            sp500_data = sp500_data.rename(columns={date_col: 'date'})
                        else:
                            # If no date column found, use index
                            sp500_data['date'] = sp500_data.index
                    elif 'Date' in sp500_data.columns:
                        sp500_data = sp500_data.rename(columns={'Date': 'date'})
                    
                    sp500_data = sp500_data[['date', 'volatility_rolling_std']].copy()
                    sp500_data = sp500_data.dropna()
            except Exception as e:
                logger.warning(f"Could not fetch S&P 500 data: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        if sp500_data is None or len(sp500_data) == 0:
            logger.warning("No S&P 500 data available. Cannot create visualization.")
            return go.Figure()
        
        # Normalize volatility to 0-100 range
        vol_min = sp500_data['volatility_rolling_std'].min()
        vol_max = sp500_data['volatility_rolling_std'].max()
        vol_range = vol_max - vol_min
        
        if vol_range > 0:
            sp500_data['volatility_normalized'] = (
                (sp500_data['volatility_rolling_std'] - vol_min) / vol_range * 100
            )
        else:
            sp500_data['volatility_normalized'] = 50.0  # Default to middle if no range
        
        # Sort by date
        sp500_data = sp500_data.sort_values('date')
        
        # Create figure
        fig = go.Figure()
        
        # Plot S&P 500 volatility line
        fig.add_trace(
            go.Scatter(
                x=sp500_data['date'],
                y=sp500_data['volatility_normalized'],
                mode='lines',
                name='S&P 500 Volatility',
                line=dict(
                    color='#1E88E5',
                    width=3,
                    shape='spline'  # Smooth line
                ),
                fill='tozeroy',
                fillcolor='rgba(30, 136, 229, 0.1)',
                hovertemplate=(
                    '<b>S&P 500 Volatility</b><br>'
                    'Date: %{x}<br>'
                    'Volatility: %{y:.2f}/100<br>'
                    'Raw: %{customdata:.4f}<extra></extra>'
                ),
                customdata=sp500_data['volatility_rolling_std']
            )
        )
        
        # Prepare trade data
        trades_sorted = features.sort_values('date').copy()
        
        # Separate buy and sell trades
        if 'side' in trades_sorted.columns:
            # Handle case where side might have NaN values
            side_col = trades_sorted['side'].astype(str).str.lower().str.strip()
            # Filter out 'nan' strings that come from actual NaN values
            buy_trades = trades_sorted[(side_col == 'buy') & (side_col != 'nan')].copy()
            sell_trades = trades_sorted[(side_col == 'sell') & (side_col != 'nan')].copy()
            
            logger.info(f"S&P 500 visualization: Found {len(buy_trades)} buy trades and {len(sell_trades)} sell trades")
        else:
            logger.warning("'side' column not found in features. Cannot separate buy/sell trades.")
            buy_trades = pd.DataFrame()
            sell_trades = pd.DataFrame()
        
        # For each trade, find the closest S&P 500 volatility value
        def get_volatility_for_date(trade_date, sp500_df):
            """Get volatility value for a trade date (find closest date)."""
            # Find closest date in S&P 500 data
            sp500_df = sp500_df.copy()
            
            # Ensure dates are datetime and handle timezone
            if not pd.api.types.is_datetime64_any_dtype(sp500_df['date']):
                sp500_df['date'] = pd.to_datetime(sp500_df['date'])
            
            # Convert trade_date to datetime if needed
            if not isinstance(trade_date, pd.Timestamp):
                trade_date = pd.to_datetime(trade_date)
            
            # Remove timezone from both if one is tz-aware and the other is not
            if sp500_df['date'].dt.tz is not None and trade_date.tz is None:
                sp500_df['date'] = sp500_df['date'].dt.tz_localize(None)
            elif sp500_df['date'].dt.tz is None and trade_date.tz is not None:
                trade_date = trade_date.tz_localize(None)
            elif sp500_df['date'].dt.tz is not None and trade_date.tz is not None:
                # Both have timezone, normalize to same timezone
                sp500_df['date'] = sp500_df['date'].dt.tz_convert(None)
                trade_date = trade_date.tz_localize(None)
            
            # Calculate time differences
            sp500_df['date_diff'] = abs((sp500_df['date'] - trade_date).dt.total_seconds())
            closest_idx = sp500_df['date_diff'].idxmin()
            
            # Check if the closest date is within 7 days (reasonable tolerance)
            closest_date_diff = sp500_df.loc[closest_idx, 'date_diff'] / (24 * 3600)  # Convert to days
            if closest_date_diff > 7:
                logger.debug(f"Closest S&P 500 date is {closest_date_diff:.1f} days away from trade date {trade_date}")
            
            return sp500_df.loc[closest_idx, 'volatility_normalized']
        
        # Plot buy trades (green dots with stock names)
        logger.info(f"Attempting to plot {len(buy_trades)} buy trades on S&P 500 chart")
        if len(buy_trades) > 0:
            buy_volatilities = []
            buy_dates = []
            buy_symbols = []
            buy_hover_texts = []
            
            for _, trade in buy_trades.iterrows():
                trade_date = trade['date']
                symbol = trade.get('symbol', 'UNKNOWN')
                
                try:
                    vol_value = get_volatility_for_date(trade_date, sp500_data)
                    buy_volatilities.append(vol_value)
                    buy_dates.append(trade_date)
                    buy_symbols.append(symbol)
                    
                    # Create hover text
                    hover_text = f"<b>BUY {symbol}</b><br>"
                    hover_text += f"Date: {trade_date}<br>"
                    hover_text += f"S&P 500 Volatility: {vol_value:.2f}/100<br>"
                    if 'price' in trade and pd.notna(trade['price']):
                        hover_text += f"Price: ${trade['price']:.2f}<br>"
                    if 'quantity' in trade and pd.notna(trade['quantity']):
                        hover_text += f"Quantity: {trade['quantity']}"
                    buy_hover_texts.append(hover_text)
                except Exception as e:
                    logger.warning(f"Could not get volatility for buy trade {symbol} on {trade_date}: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(buy_volatilities)} buy trades for plotting")
            if len(buy_volatilities) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_dates,
                        y=buy_volatilities,
                        mode='markers+text',
                        name='Buy Trades',
                        marker=dict(
                            color='#28A745',  # Green
                            size=12,
                            symbol='circle',
                            line=dict(width=2, color='white'),
                            opacity=0.9
                        ),
                        text=buy_symbols,
                        textposition='top center',
                        textfont=dict(
                            size=10,
                            color='#28A745',
                            family='Arial Black'
                        ),
                        customdata=buy_hover_texts,
                        hovertemplate='%{customdata}<extra></extra>'
                    )
                )
        
        # Plot sell trades (red dots with stock names)
        logger.info(f"Attempting to plot {len(sell_trades)} sell trades on S&P 500 chart")
        if len(sell_trades) > 0:
            sell_volatilities = []
            sell_dates = []
            sell_symbols = []
            sell_hover_texts = []
            
            for _, trade in sell_trades.iterrows():
                trade_date = trade['date']
                symbol = trade.get('symbol', 'UNKNOWN')
                
                try:
                    vol_value = get_volatility_for_date(trade_date, sp500_data)
                    sell_volatilities.append(vol_value)
                    sell_dates.append(trade_date)
                    sell_symbols.append(symbol)
                    
                    # Create hover text
                    hover_text = f"<b>SELL {symbol}</b><br>"
                    hover_text += f"Date: {trade_date}<br>"
                    hover_text += f"S&P 500 Volatility: {vol_value:.2f}/100<br>"
                    if 'price' in trade and pd.notna(trade['price']):
                        hover_text += f"Price: ${trade['price']:.2f}<br>"
                    if 'quantity' in trade and pd.notna(trade['quantity']):
                        hover_text += f"Quantity: {trade['quantity']}<br>"
                    if 'realized_pnl' in trade and pd.notna(trade['realized_pnl']):
                        pnl_color = 'green' if trade['realized_pnl'] > 0 else 'red'
                        hover_text += f"P&L: <span style='color:{pnl_color}'>${trade['realized_pnl']:.2f}</span>"
                    sell_hover_texts.append(hover_text)
                except Exception as e:
                    logger.warning(f"Could not get volatility for sell trade {symbol} on {trade_date}: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(sell_volatilities)} sell trades for plotting")
            if len(sell_volatilities) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sell_dates,
                        y=sell_volatilities,
                        mode='markers+text',
                        name='Sell Trades',
                        marker=dict(
                            color='#DC3545',  # Red
                            size=12,
                            symbol='circle',
                            line=dict(width=2, color='white'),
                            opacity=0.9
                        ),
                        text=sell_symbols,
                        textposition='bottom center',
                        textfont=dict(
                            size=10,
                            color='#DC3545',
                            family='Arial Black'
                        ),
                        customdata=sell_hover_texts,
                        hovertemplate='%{customdata}<extra></extra>'
                    )
                )
        
        # Update layout with beautiful styling
        fig.update_layout(
            title=dict(
                text='S&P 500 Volatility Timeline with Trade Markers<br>'
                     '<sub>Green dots = Buy trades, Red dots = Sell trades (with stock names)</sub>',
                x=0.5,
                xanchor='center',
                font=dict(size=20, color='#1a1a1a')
            ),
            xaxis=dict(
                title=dict(text='Date', font=dict(size=14, color='#1a1a1a')),
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                showline=True,
                linecolor='rgba(128, 128, 128, 0.5)'
            ),
            yaxis=dict(
                title=dict(text='S&P 500 Volatility (Normalized 0-100)', font=dict(size=14, color='#1a1a1a')),
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                showline=True,
                linecolor='rgba(128, 128, 128, 0.5)',
                range=[0, 100]
            ),
            hovermode='closest',
            height=700,
            width=1400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1,
                font=dict(size=12)
            ),
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        self.figures['sp500_volatility'] = fig
        return fig
    
    def save_all_figures(self, output_dir: str = "output/"):
        """Save all figures to HTML files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in self.figures.items():
            if isinstance(fig, dict):
                for sub_name, sub_fig in fig.items():
                    filepath = os.path.join(output_dir, f"{name}_{sub_name}.html")
                    sub_fig.write_html(filepath)
                    logger.info(f"Saved {name}_{sub_name} to {filepath}")
            else:
                filepath = os.path.join(output_dir, f"{name}.html")
                fig.write_html(filepath)
                logger.info(f"Saved {name} to {filepath}")
    
    def create_unified_dashboard(self, output_dir: str = "output/", 
                                xai_file: str = "xai_explanation.txt",
                                report_file: str = "behavioral_report.txt") -> str:
        """
        Create a unified dark-themed dashboard HTML page with all visualizations and reports.
        
        Args:
            output_dir: Directory containing visualization files and reports
            xai_file: Filename of XAI explanation text file
            report_file: Filename of behavioral report text file
            
        Returns:
            Path to the generated dashboard HTML file
        """
        import os
        
        # Visualization order and descriptions
        viz_order = [
            ('regime_timeline', 'Behavioral Regime Timeline', 'Shows how your trading behavior clusters changed over time, with market regime overlays'),
            ('cluster_timeline', 'Cluster Distribution Timeline', 'Distribution of behavioral clusters across your trading period'),
            ('cluster_scatter', 'Cluster Scatter Plot', '2D visualization of behavioral clusters showing cluster areas and data point grouping (positive quadrant only)'),
            ('sp500_volatility', 'S&P 500 Volatility with Trades', 'S&P 500 volatility timeline (0-100) with buy (green) and sell (red) trade markers labeled with stock names'),
            ('volatility_timeline', 'Volatility Timeline with Trades', 'Volatility over time for each stock with buy (green) and sell (red) trade markers'),
            ('trade_journey', 'Trade Journey Timeline', 'Individual trade journeys from entry to exit with P&L'),
            ('signal_scorecard', 'Signal Following Scorecard', 'How well you followed technical indicators (RSI, MACD, EMA)'),
            ('stability_scorecard', 'Behavioral Stability Score', 'Consistency of your trading behavior over time (does not measure skill or profitability)'),
            ('post_event', 'Post-Loss Behavior Analysis', 'Trading behavior after losses (revenge trading patterns)'),
            ('performance_matrix', 'Performance Matrix', 'Performance across different market regimes and behavioral clusters'),
            ('deviation_plots', 'Behavioral Deviation Plots', 'How your trading deviates from baseline patterns')
        ]
        
        # Read text files
        xai_path = os.path.join(output_dir, xai_file)
        report_path = os.path.join(output_dir, report_file)
        
        xai_content = ""
        if os.path.exists(xai_path):
            try:
                with open(xai_path, 'r', encoding='utf-8') as f:
                    xai_content = f.read()
            except UnicodeDecodeError:
                with open(xai_path, 'r', encoding='utf-8', errors='replace') as f:
                    xai_content = f.read()
        
        report_content = ""
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
            except UnicodeDecodeError:
                with open(report_path, 'r', encoding='utf-8', errors='replace') as f:
                    report_content = f.read()
        
        # Generate HTML
        html_parts = []
        
        # HTML Header with dark theme CSS
        html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Behavioral Trading Analysis Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            background: #000000;
            color: #ffffff;
            line-height: 1.7;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            padding: 40px 20px;
            border-bottom: 2px solid #333333;
            margin-bottom: 40px;
            background: #1a1a1a;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .header h1 {
            font-size: 2.2em;
            color: #ffffff;
            margin-bottom: 12px;
            font-weight: 600;
        }
        
        .header p {
            color: #e9ecef;
            font-size: 1.05em;
        }
        
        .section {
            background: #1a1a1a;
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        }
        
        .section-title {
            font-size: 1.6em;
            color: #ffffff;
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 2px solid #333333;
            font-weight: 600;
        }
        
        .section-description {
            color: #e9ecef;
            font-size: 0.95em;
            margin-bottom: 24px;
            line-height: 1.6;
        }
        
        .viz-container {
            background: #0a0a0a;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #333333;
        }
        
        .viz-container iframe {
            width: 100%;
            height: 700px;
            border: none;
            border-radius: 4px;
            background: #ffffff;
        }
        
        .text-content {
            background: #0a0a0a;
            border-radius: 6px;
            padding: 24px;
            margin: 20px 0;
            border: 1px solid #333333;
            white-space: pre-wrap;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', monospace;
            font-size: 0.9em;
            line-height: 1.8;
            color: #ffffff;
            max-height: 600px;
            overflow-y: auto;
        }
        
        .text-content::-webkit-scrollbar {
            width: 8px;
        }
        
        .text-content::-webkit-scrollbar-track {
            background: #1a1a1a;
            border-radius: 4px;
        }
        
        .text-content::-webkit-scrollbar-thumb {
            background: #6c757d;
            border-radius: 4px;
        }
        
        .text-content::-webkit-scrollbar-thumb:hover {
            background: #868e96;
        }
        
        .download-btn {
            display: inline-block;
            background: #6c757d;
            color: #ffffff;
            padding: 10px 20px;
            text-decoration: none;
            border: none;
            border-radius: 5px;
            font-weight: 500;
            margin: 10px 10px 10px 0;
            transition: background-color 0.2s ease;
            cursor: pointer;
            font-size: 0.95em;
            font-family: inherit;
        }
        
        .download-btn:hover {
            background: #868e96;
        }
        
        .download-btn:active {
            background: #5a6268;
        }
        
        .scale-info {
            background: #0a0a0a;
            border-left: 3px solid #6c757d;
            padding: 14px 16px;
            margin: 16px 0;
            border-radius: 4px;
            font-size: 0.9em;
            color: #e9ecef;
        }
        
        .scale-info strong {
            color: #ffffff;
        }
        
        .nav-menu {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #1a1a1a;
            border: 1px solid #333333;
            border-radius: 6px;
            padding: 16px;
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
        }
        
        .nav-menu h3 {
            color: #ffffff;
            font-size: 0.95em;
            margin-bottom: 10px;
            font-weight: 600;
        }
        
        .nav-menu a {
            display: block;
            color: #e9ecef;
            text-decoration: none;
            padding: 6px 0;
            font-size: 0.9em;
            transition: color 0.2s;
        }
        
        .nav-menu a:hover {
            color: #ffffff;
        }
        
        @media (max-width: 768px) {
            body {
                padding: 12px;
            }
            
            .nav-menu {
                position: relative;
                top: 0;
                right: 0;
                margin-bottom: 20px;
            }
            
            .header {
                padding: 30px 15px;
            }
            
            .header h1 {
                font-size: 1.6em;
            }
            
            .section {
                padding: 20px;
            }
            
            .viz-container iframe {
                height: 500px;
            }
        }
    </style>
    <script>
        function downloadFile(filename) {
            // Create a temporary anchor element to trigger download
            const link = document.createElement('a');
            link.href = filename;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    </script>
</head>
<body>
    <div class="nav-menu">
        <h3>Navigation</h3>
        <a href="#visualizations">All Visualizations</a>
        <a href="#regime-timeline">Regime Timeline</a>
        <a href="#cluster-timeline">Cluster Timeline</a>
        <a href="#cluster-scatter">Cluster Scatter</a>
        <a href="#sp500-volatility">S&P 500 Volatility</a>
        <a href="#volatility-timeline">Volatility Timeline</a>
        <a href="#trade-journey">Trade Journey</a>
        <a href="#signal-scorecard">Signal Scorecard</a>
        <a href="#stability-scorecard">Stability Scorecard</a>
        <a href="#post-event">Post-Loss Analysis</a>
        <a href="#performance-matrix">Performance Matrix</a>
        <a href="#deviation-plots">Deviation Plots</a>
        <a href="#macd-stock-charts">MACD Stock Charts</a>
        <a href="#xai-explanation">XAI Summary</a>
        <a href="#behavioral-report">Full Report</a>
        <a href="#stock-performance">Stock Performance</a>
    </div>
    
    <div class="container">
        <div class="header">
            <h1>Behavioral Trading Analysis Dashboard</h1>
            <p>Comprehensive analysis of your trading behavior, patterns, and performance</p>
        </div>
        
        <div id="visualizations">
            <div class="section">
                <h2 class="section-title">Interactive Visualizations</h2>
                <p class="section-description">Explore your trading patterns through interactive charts. Hover over data points for detailed information.</p>
""")
        
        # Add visualizations
        for viz_name, viz_title, viz_desc in viz_order:
            viz_file = os.path.join(output_dir, f"{viz_name}.html")
            if os.path.exists(viz_file):
                anchor_id = viz_name.replace('_', '-')
                html_parts.append(f"""
                <div class="section" id="{anchor_id}">
                    <h3 class="section-title">{viz_title}</h3>
                    <p class="section-description">{viz_desc}</p>
                    <div class="scale-info">
                        <strong>Note:</strong> Interactive charts - zoom, hover for details, toggle series via legend.
                    </div>
                    <div class="viz-container">
                        <iframe src="{viz_name}.html" title="{viz_title}"></iframe>
                    </div>
                </div>
""")
        
        # Add MACD stock charts section
        macd_files = []
        for file in os.listdir(output_dir):
            if file.startswith('macd_stock_charts_') and file.endswith('.html'):
                macd_files.append(file)
        
        if macd_files:
            html_parts.append("""
                <div class="section" id="macd-stock-charts">
                    <h2 class="section-title">MACD Charts by Stock</h2>
                    <p class="section-description">MACD Fast and Slow lines for each stock with buy (green) and sell (red) trade markers on the fast line.</p>
""")
            for macd_file in sorted(macd_files):
                stock_symbol = macd_file.replace('macd_stock_charts_', '').replace('.html', '')
                anchor_id = f"macd-{stock_symbol.lower()}"
                html_parts.append(f"""
                    <div class="section" id="{anchor_id}">
                        <h3 class="section-title">{stock_symbol} - MACD Chart</h3>
                        <p class="section-description">MACD Fast Line (blue) and Slow Line (purple) with trade markers</p>
                        <div class="scale-info">
                            <strong>Note:</strong> X-axis shows days (positive), Y-axis shows MACD values (positive and negative). Green dots = Buy trades, Red dots = Sell trades.
                        </div>
                        <div class="viz-container">
                            <iframe src="{macd_file}" title="{stock_symbol} MACD Chart"></iframe>
                        </div>
                    </div>
""")
            html_parts.append("""
                </div>
""")
        
        html_parts.append("""
        </div>
        
        <div id="xai-explanation">
            <div class="section">
                <h2 class="section-title">Explainable AI Summary</h2>
                <p class="section-description">AI-generated explanation of your trading behavior using rule-based Natural Language Generation and feature contribution ranking.</p>
                <button onclick="downloadFile('xai_explanation.txt')" class="download-btn">Download XAI Explanation</button>
                <div class="text-content">""")
        
        # Add XAI content (escape HTML)
        xai_escaped = html.escape(xai_content) if xai_content else "XAI explanation file not found."
        html_parts.append(xai_escaped)
        
        html_parts.append("""
                </div>
            </div>
        </div>
        
        <div id="stock-performance">
            <div class="section">
                <h2 class="section-title">Stock Performance Analysis</h2>
                <p class="section-description">Detailed analysis of best and worst performing stocks with explanations.</p>
                <button onclick="downloadFile('stock_performance_analysis.txt')" class="download-btn">Download Stock Performance Report</button>
                <div class="text-content">""")
        
        stock_perf_path = os.path.join(output_dir, "stock_performance_analysis.txt")
        stock_perf_content = ""
        if os.path.exists(stock_perf_path):
            try:
                with open(stock_perf_path, 'r', encoding='utf-8') as f:
                    stock_perf_content = f.read()
            except UnicodeDecodeError:
                with open(stock_perf_path, 'r', encoding='utf-8', errors='replace') as f:
                    stock_perf_content = f.read()
        
        stock_perf_escaped = html.escape(stock_perf_content) if stock_perf_content else "Stock performance analysis file not found."
        html_parts.append(stock_perf_escaped)
        
        html_parts.append("""
                </div>
            </div>
        </div>
        
        <div id="behavioral-report">
            <div class="section">
                <h2 class="section-title">Detailed Behavioral Report</h2>
                <p class="section-description">Comprehensive analysis including statistics, baselines, clusters, change points, and anomalies.</p>
                <button onclick="downloadFile('behavioral_report.txt')" class="download-btn">Download Full Report</button>
                <div class="text-content">""")
        
        report_escaped = html.escape(report_content) if report_content else "Behavioral report file not found."
        html_parts.append(report_escaped)
        
        html_parts.append("""
                </div>
            </div>
        </div>
        
        <div class="section" style="text-align: center; padding: 30px; background: #0a0a0a;">
            <p style="color: #e9ecef; font-size: 0.9em;">
                Generated by Behavioral Trading Analysis System<br>
                Using Explainable AI (XAI) and Advanced Pattern Recognition
            </p>
        </div>
    </div>
</body>
</html>""")
        
        # Write HTML file
        dashboard_path = os.path.join(output_dir, "dashboard.html")
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(''.join(html_parts))
        
        logger.info(f"Unified dashboard saved to {dashboard_path}")
        return dashboard_path
    
    def create_macd_stock_charts(self, features: pd.DataFrame, market_data_fetcher) -> Dict[str, go.Figure]:
        """
        Create MACD charts for each stock showing fast and slow lines with trade markers.
        
        Args:
            features: DataFrame with trade features
            market_data_fetcher: MarketDataFetcher instance for fetching OHLCV data
            
        Returns:
            Dictionary mapping stock symbols to their MACD chart figures
        """
        if 'symbol' not in features.columns:
            logger.warning("No 'symbol' column found. Cannot create stock-specific MACD charts.")
            return {}
        
        stock_figures = {}
        unique_symbols = features['symbol'].dropna().unique()
        
        for symbol in unique_symbols:
            try:
                symbol_trades = features[features['symbol'] == symbol].copy()
                if len(symbol_trades) == 0:
                    continue
                
                symbol_trades = symbol_trades.sort_values('date')
                min_date = symbol_trades['date'].min()
                max_date = symbol_trades['date'].max()
                
                if not hasattr(min_date, 'strftime'):
                    min_date = pd.to_datetime(min_date)
                if not hasattr(max_date, 'strftime'):
                    max_date = pd.to_datetime(max_date)
                
                start_date = (min_date - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
                end_date = (max_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                
                ohlcv = market_data_fetcher.fetch_ohlcv(symbol, start_date, end_date)
                if ohlcv is None or len(ohlcv) == 0:
                    logger.warning(f"Could not fetch OHLCV data for {symbol}")
                    continue
                
                if 'Date' in ohlcv.columns:
                    ohlcv['date'] = pd.to_datetime(ohlcv['Date'])
                elif 'date' in ohlcv.columns:
                    ohlcv['date'] = pd.to_datetime(ohlcv['date'])
                else:
                    ohlcv['date'] = pd.to_datetime(ohlcv.index)
                
                ohlcv = ohlcv.sort_values('date')
                
                if 'Close' not in ohlcv.columns:
                    logger.warning(f"No 'Close' column in OHLCV data for {symbol}")
                    continue
                
                if ohlcv['date'].dt.tz is not None:
                    ohlcv['date'] = ohlcv['date'].dt.tz_localize(None)
                
                macd_data = market_data_fetcher.compute_macd(ohlcv['Close'], fast_period=12, slow_period=26, signal_period=9)
                ohlcv['macd_line'] = macd_data['macd_line']
                ohlcv['macd_signal'] = macd_data['macd_signal']
                
                ohlcv = ohlcv.dropna(subset=['macd_line', 'macd_signal'])
                
                if len(ohlcv) == 0:
                    logger.warning(f"No MACD data available for {symbol}")
                    continue
                
                ohlcv = ohlcv.reset_index(drop=True)
                ohlcv['day'] = range(1, len(ohlcv) + 1)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=ohlcv['day'].values,
                    y=ohlcv['macd_line'].values,
                    mode='lines',
                    name='MACD Fast Line',
                    line=dict(color='#2E86AB', width=2),
                    hovertemplate='Day: %{x}<br>MACD Fast: %{y:.4f}<extra></extra>'
                ))
                
                fig.add_trace(go.Scatter(
                    x=ohlcv['day'].values,
                    y=ohlcv['macd_signal'].values,
                    mode='lines',
                    name='MACD Slow Line',
                    line=dict(color='#A23B72', width=2),
                    hovertemplate='Day: %{x}<br>MACD Slow: %{y:.4f}<extra></extra>'
                ))
                
                if 'side' in symbol_trades.columns:
                    side_col = symbol_trades['side'].astype(str).str.lower().str.strip()
                    buy_trades = symbol_trades[(side_col == 'buy') & (side_col != 'nan')].copy()
                    sell_trades = symbol_trades[(side_col == 'sell') & (side_col != 'nan')].copy()
                    
                    buy_days = []
                    buy_macd_values = []
                    for _, trade in buy_trades.iterrows():
                        trade_date = pd.to_datetime(trade['date'])
                        if trade_date.tz is not None:
                            trade_date = trade_date.tz_localize(None)
                        
                        date_diffs = abs(ohlcv['date'] - trade_date)
                        closest_idx = date_diffs.idxmin()
                        if date_diffs.loc[closest_idx] <= pd.Timedelta(days=7):
                            buy_days.append(ohlcv.loc[closest_idx, 'day'])
                            buy_macd_values.append(ohlcv.loc[closest_idx, 'macd_line'])
                    
                    if len(buy_days) > 0:
                        fig.add_trace(go.Scatter(
                            x=buy_days,
                            y=buy_macd_values,
                            mode='markers',
                            name='Buy Trades',
                            marker=dict(
                                color='#28A745',
                                size=12,
                                symbol='circle',
                                line=dict(width=2, color='white'),
                                opacity=0.9
                            ),
                            hovertemplate=f'<b>BUY {symbol}</b><br>Day: %{{x}}<br>MACD Fast: %{{y:.4f}}<extra></extra>'
                        ))
                    
                    sell_days = []
                    sell_macd_values = []
                    for _, trade in sell_trades.iterrows():
                        trade_date = pd.to_datetime(trade['date'])
                        if trade_date.tz is not None:
                            trade_date = trade_date.tz_localize(None)
                        
                        date_diffs = abs(ohlcv['date'] - trade_date)
                        closest_idx = date_diffs.idxmin()
                        if date_diffs.loc[closest_idx] <= pd.Timedelta(days=7):
                            sell_days.append(ohlcv.loc[closest_idx, 'day'])
                            sell_macd_values.append(ohlcv.loc[closest_idx, 'macd_line'])
                    
                    if len(sell_days) > 0:
                        fig.add_trace(go.Scatter(
                            x=sell_days,
                            y=sell_macd_values,
                            mode='markers',
                            name='Sell Trades',
                            marker=dict(
                                color='#DC3545',
                                size=12,
                                symbol='circle',
                                line=dict(width=2, color='white'),
                                opacity=0.9
                            ),
                            hovertemplate=f'<b>SELL {symbol}</b><br>Day: %{{x}}<br>MACD Fast: %{{y:.4f}}<extra></extra>'
                        ))
                
                fig.update_layout(
                    title=dict(
                        text=f'{symbol} - MACD Fast & Slow Lines with Trade Markers',
                        x=0.5,
                        xanchor='center',
                        font=dict(size=18)
                    ),
                    xaxis=dict(
                        title='Days',
                        showgrid=True,
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        range=[0, None]
                    ),
                    yaxis=dict(
                        title='MACD Value',
                        showgrid=True,
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        zeroline=True,
                        zerolinecolor='rgba(128, 128, 128, 0.5)'
                    ),
                    hovermode='closest',
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    ),
                    template='plotly_dark',
                    height=600
                )
                
                stock_figures[symbol] = fig
                logger.info(f"Created MACD chart for {symbol}")
                
            except Exception as e:
                logger.warning(f"Could not create MACD chart for {symbol}: {e}")
                continue
        
        return stock_figures

    def create_probability_distribution_plots(self, features: pd.DataFrame) -> Optional[go.Figure]:
        """
        Create violin/density plots for key behavioral features.
        Features: holding_duration_days, position_size_dollar_value, trades_per_day, time_gap_days_since_last_trade.
        """
        try:
            cols = [
                'holding_duration_days',
                'position_size_dollar_value',
                'trades_per_day',
                'time_gap_days_since_last_trade'
            ]
            available = [c for c in cols if c in features.columns]
            if not available:
                logger.warning("create_probability_distribution_plots: none of the required feature columns found.")
                return None
            data = features[available].dropna()
            if len(data) == 0:
                logger.warning("create_probability_distribution_plots: no non-null data for selected features.")
                return None
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=available,
                vertical_spacing=0.12,
                horizontal_spacing=0.08
            )
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            for idx, col in enumerate(available):
                r, c = positions[idx]
                y = data[col].values
                fig.add_trace(
                    go.Violin(
                        y=y,
                        name=col,
                        box_visible=True,
                        line_color='#2E86AB',
                        fillcolor='rgba(46, 134, 171, 0.4)'
                    ),
                    row=r, col=c
                )
                fig.update_yaxes(title_text=col, row=r, col=c)
            fig.update_layout(
                title=dict(text="Behavioral Feature Probability Distributions", x=0.5, xanchor='center'),
                template='plotly_dark',
                height=600,
                showlegend=False
            )
            fig.update_xaxes(showticklabels=False)
            self.figures['probability_distributions'] = fig
            return fig
        except Exception as e:
            logger.warning(f"create_probability_distribution_plots failed: {e}")
            return None

    def create_confidence_band_timeline(self, features: pd.DataFrame) -> Optional[go.Figure]:
        """
        Create a timeline of behavioral metrics with 95% CI bands (rolling mean ± 1.96 * rolling std).
        """
        try:
            cols = ['holding_duration_days', 'position_size_dollar_value', 'trades_per_day']
            available = [c for c in cols if c in features.columns]
            if not available:
                logger.warning("create_confidence_band_timeline: none of the required feature columns found.")
                return None
            if 'date' not in features.columns:
                logger.warning("create_confidence_band_timeline: 'date' column not found.")
                return None
            df = features.sort_values('date').copy()
            window = min(30, max(5, len(df) // 10))
            fig = make_subplots(
                rows=len(available), cols=1,
                subplot_titles=available,
                vertical_spacing=0.08,
                row_heights=[1.0 / len(available)] * len(available)
            )
            for i, col in enumerate(available):
                s = df[col].dropna()
                if len(s) < window:
                    continue
                roll_mean = s.rolling(window, min_periods=1).mean()
                roll_std = s.rolling(window, min_periods=1).std().fillna(0)
                upper = roll_mean + 1.96 * roll_std
                lower = roll_mean - 1.96 * roll_std
                x = df.loc[s.index, 'date'].values
                fig.add_trace(
                    go.Scatter(x=x, y=upper, mode='lines', line=dict(width=0), showlegend=False),
                    row=i + 1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=x, y=lower, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(46, 134, 171, 0.3)', name=f'{col} 95% CI'),
                    row=i + 1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=x, y=roll_mean.values, mode='lines', name=col, line=dict(color='#2E86AB', width=2)),
                    row=i + 1, col=1
                )
                fig.update_yaxes(title_text=col, row=i + 1, col=1)
            fig.update_xaxes(title_text="Date", row=len(available), col=1)
            fig.update_layout(
                title=dict(text="Behavioral Metrics with 95% Confidence Bands", x=0.5, xanchor='center'),
                template='plotly_dark',
                height=250 * len(available),
                hovermode='x unified'
            )
            self.figures['confidence_bands'] = fig
            return fig
        except Exception as e:
            logger.warning(f"create_confidence_band_timeline failed: {e}")
            return None

    def create_regime_transition_heatmap(self, transition_matrix: np.ndarray, n_regimes: int = 3) -> Optional[go.Figure]:
        """
        Create a heatmap of HMM transition probabilities (From Regime -> To Regime).
        """
        try:
            n = transition_matrix.shape[0] if transition_matrix.ndim >= 1 else n_regimes
            if transition_matrix.size == 0:
                logger.warning("create_regime_transition_heatmap: empty transition matrix.")
                return None
            mat = np.asarray(transition_matrix)
            if mat.shape[0] != mat.shape[1]:
                mat = mat[:n, :n] if mat.shape[0] >= n and mat.shape[1] >= n else mat
            n = mat.shape[0]
            from_labels = [f"From Regime {i}" for i in range(n)]
            to_labels = [f"To Regime {i}" for i in range(n)]
            fig = go.Figure(data=go.Heatmap(
                z=mat,
                x=to_labels,
                y=from_labels,
                colorscale='Viridis',
                colorbar=dict(title="Probability"),
                hovertemplate="From %{y}<br>To %{x}<br>P: %{z:.3f}<extra></extra>"
            ))
            fig.update_layout(
                title=dict(text="Regime Transition Probability Heatmap", x=0.5, xanchor='center'),
                xaxis=dict(title="To Regime"),
                yaxis=dict(title="From Regime"),
                template='plotly_dark',
                height=400 + n * 40
            )
            self.figures['regime_transition_heatmap'] = fig
            return fig
        except Exception as e:
            logger.warning(f"create_regime_transition_heatmap failed: {e}")
            return None

    def create_counterfactual_comparison(self, actual_pnl: float, counterfactual_results: Dict[str, float]) -> Optional[go.Figure]:
        """
        Side-by-side bar chart comparing actual vs hypothetical performance (e.g. actual P&L vs ATR-based vs equal-weight).
        """
        try:
            names = ["Actual"] + list(counterfactual_results.keys())
            values = [actual_pnl] + list(counterfactual_results.values())
            colors = ['#2E86AB'] + ['#A23B72', '#28A745', '#FFC107'][:len(names)]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=names,
                y=values,
                marker_color=colors,
                text=[f"{v:,.2f}" for v in values],
                textposition='outside'
            ))
            fig.update_layout(
                title=dict(text="Counterfactual Performance: Actual vs Hypothetical P&L", x=0.5, xanchor='center'),
                xaxis=dict(title="Strategy"),
                yaxis=dict(title="Total P&L"),
                template='plotly_dark',
                height=450
            )
            self.figures['counterfactual_comparison'] = fig
            return fig
        except Exception as e:
            logger.warning(f"create_counterfactual_comparison failed: {e}")
            return None

    def create_sankey_diagram(self, features: pd.DataFrame, clusters: np.ndarray) -> Optional[go.Figure]:
        """
        Sankey diagram: Entry regime -> Behavioral cluster -> Outcome (win/loss).
        """
        try:
            n = len(features)
            if n == 0 or len(clusters) != n:
                logger.warning("create_sankey_diagram: features and clusters length mismatch or empty.")
                return None
            regime_col = None
            for c in ['regime', 'behavioral_regime', 'market_regime']:
                if c in features.columns:
                    regime_col = c
                    break
            if regime_col is None:
                regime_col = 'regime'
                features = features.copy()
                features['regime'] = 0
            outcomes = []
            for c in ['outcome', 'pnl', 'realized_pnl', 'win']:
                if c in features.columns:
                    if c == 'win':
                        outcomes = (features[c].astype(float) > 0).map({True: 'Win', False: 'Loss'}).values
                    elif c == 'pnl' or c == 'realized_pnl':
                        outcomes = (features[c].astype(float) > 0).map({True: 'Win', False: 'Loss'}).fillna('Unknown').values
                    else:
                        outcomes = features[c].astype(str).values
                    break
            if len(outcomes) == 0:
                outcomes = np.array(['Outcome'] * n)
            unique_regimes = sorted(features[regime_col].dropna().unique().astype(int).tolist())
            unique_clusters = sorted(np.unique(clusters).tolist())
            unique_outcomes = sorted(np.unique(outcomes).tolist())
            node_labels = (
                [f"Regime {r}" for r in unique_regimes] +
                [f"Cluster {c}" for c in unique_clusters] +
                list(unique_outcomes)
            )
            n_reg = len(unique_regimes)
            n_clu = len(unique_clusters)
            n_out = len(unique_outcomes)
            reg_ix = {r: i for i, r in enumerate(unique_regimes)}
            clu_ix = {c: i for i, c in enumerate(unique_clusters)}
            out_ix = {o: i for i, o in enumerate(unique_outcomes)}
            source = []
            target = []
            value = []
            for i in range(n):
                r = int(features[regime_col].iloc[i]) if regime_col in features.columns else 0
                c = clusters[i]
                o = outcomes[i] if i < len(outcomes) else 'Outcome'
                if r not in reg_ix or c not in clu_ix or o not in out_ix:
                    continue
                source.append(reg_ix[r])
                target.append(n_reg + clu_ix[c])
                value.append(1)
            for i in range(n):
                c = clusters[i]
                o = outcomes[i] if i < len(outcomes) else 'Outcome'
                if c not in clu_ix or o not in out_ix:
                    continue
                source.append(n_reg + clu_ix[c])
                target.append(n_reg + n_clu + out_ix[o])
                value.append(1)
            from collections import Counter
            link_sources = []
            link_targets = []
            link_values = []
            for (s, t), v in Counter(zip(source, target)).items():
                link_sources.append(s)
                link_targets.append(t)
                link_values.append(v)
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='rgba(128,128,128,0.5)', width=0.5),
                    label=node_labels,
                    color=['#2E86AB'] * n_reg + ['#A23B72'] * n_clu + (['#28A745', '#DC3545', '#6c757d'] + ['#FFC107'] * max(0, n_out - 3))[:n_out]
                ),
                link=dict(source=link_sources, target=link_targets, value=link_values)
            )])
            fig.update_layout(
                title=dict(text="Trade Flow: Entry Regime → Behavioral Cluster → Outcome", x=0.5, xanchor='center'),
                template='plotly_dark',
                height=500
            )
            self.figures['sankey_diagram'] = fig
            return fig
        except Exception as e:
            logger.warning(f"create_sankey_diagram failed: {e}")
            return None

    def create_decision_tree_explainer(self, features: pd.DataFrame, clusters: np.ndarray) -> Optional[go.Figure]:
        """
        Visual explanation of pattern classification: fit DecisionTree on features to predict clusters, show feature importances.
        """
        try:
            numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols or len(np.unique(clusters)) < 2:
                logger.warning("create_decision_tree_explainer: need numeric features and at least 2 clusters.")
                return None
            X = features[numeric_cols].fillna(0).values
            y = np.asarray(clusters).ravel()
            if len(y) != len(X):
                logger.warning("create_decision_tree_explainer: features and clusters length mismatch.")
                return None
            clf = DecisionTreeClassifier(max_depth=10, random_state=42, min_samples_leaf=2)
            clf.fit(X, y)
            imp = clf.feature_importances_
            names = numeric_cols
            order = np.argsort(imp)[::-1]
            imp = imp[order]
            names = [names[i] for i in order]
            fig = go.Figure(go.Bar(
                x=imp,
                y=names,
                orientation='h',
                marker=dict(color='#2E86AB'),
                hovertemplate='%{y}: %{x:.3f}<extra></extra>'
            ))
            fig.update_layout(
                title=dict(text="Decision Tree Feature Importance (Cluster Classification)", x=0.5, xanchor='center'),
                xaxis=dict(title="Importance"),
                yaxis=dict(title="Feature", autorange='reversed'),
                template='plotly_dark',
                height=400 + len(names) * 18
            )
            self.figures['decision_tree_explainer'] = fig
            return fig
        except Exception as e:
            logger.warning(f"create_decision_tree_explainer failed: {e}")
            return None

