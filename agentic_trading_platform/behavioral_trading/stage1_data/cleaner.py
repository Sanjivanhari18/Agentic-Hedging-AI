"""Tradebook data cleaning and position reconstruction."""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TradebookCleaner:
    """Cleans tradebook data and reconstructs positions."""
    
    def __init__(self):
        self.cleaned_trades: Optional[pd.DataFrame] = None
    
    def clean(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Clean tradebook data:
        - Remove duplicates
        - Handle partial fills
        - Reconstruct positions
        - Calculate derived primitives
        """
        df = trades.copy()
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate trades")
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Aggregate partial fills (same symbol, side, date, price)
        df = self._aggregate_partial_fills(df)
        
        # Reconstruct positions
        df = self._reconstruct_positions(df)
        
        # Calculate derived primitives
        df = self._calculate_primitives(df)
        
        self.cleaned_trades = df
        return df
    
    def _aggregate_partial_fills(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate partial fills into single trades."""
        # Validate required columns
        required_cols = ['date', 'symbol', 'side', 'quantity', 'price']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for aggregation: {missing}")
        
        # Group by symbol, side, date, and price (within tolerance)
        df['date_only'] = df['date'].dt.date
        
        # Round price to nearest cent for grouping
        df['price_rounded'] = (df['price'] * 100).round() / 100
        
        # Build aggregation dictionary dynamically based on available columns
        # Note: Columns used in groupby are automatically included, but we need to aggregate others
        agg_dict = {
            'quantity': 'sum',
            'date': 'first',  # Keep first timestamp
        }
        
        # Add required columns (price is used in groupby via price_rounded, but we need to keep original)
        if 'price' in df.columns:
            agg_dict['price'] = 'first'  # Keep first price (they should be same after grouping)
        # Note: 'symbol' and 'side' are in groupby, so pandas keeps them automatically with as_index=False
        
        # Add optional columns if they exist
        if 'order_type' in df.columns:
            agg_dict['order_type'] = 'first'
        if 'realized_pnl' in df.columns:
            agg_dict['realized_pnl'] = 'sum'
        if 'commission' in df.columns:
            agg_dict['commission'] = 'sum'
        if 'fees' in df.columns:
            agg_dict['fees'] = 'sum'
        
        grouped = df.groupby(['symbol', 'side', 'date_only', 'price_rounded'], as_index=False).agg(agg_dict)
        
        # Drop helper columns
        grouped = grouped.drop(columns=['date_only', 'price_rounded'])
        
        return grouped.sort_values('date').reset_index(drop=True)
    
    def _reconstruct_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct position state for each trade."""
        df = df.copy()
        df['position_id'] = None
        df['entry_date'] = None
        df['entry_price'] = None
        df['holding_duration'] = None
        
        # Track positions per symbol
        positions = {}  # symbol -> list of (entry_date, entry_price, quantity)
        
        for idx, row in df.iterrows():
            symbol = row['symbol']
            side = row['side']
            quantity = row['quantity']
            price = row['price']
            date = row['date']
            
            if symbol not in positions:
                positions[symbol] = []
            
            if side == 'buy':
                # Add to positions
                positions[symbol].append({
                    'entry_date': date,
                    'entry_price': price,
                    'quantity': quantity
                })
                df.at[idx, 'position_id'] = len(positions[symbol]) - 1
                df.at[idx, 'entry_date'] = date
                df.at[idx, 'entry_price'] = price
                
            elif side == 'sell':
                # Match with oldest position (FIFO)
                remaining_qty = quantity
                matched = False
                
                for pos_idx, pos in enumerate(positions[symbol]):
                    if remaining_qty <= 0:
                        break
                    
                    if remaining_qty <= pos['quantity']:
                        # Full or partial match
                        df.at[idx, 'position_id'] = pos_idx
                        df.at[idx, 'entry_date'] = pos['entry_date']
                        df.at[idx, 'entry_price'] = pos['entry_price']
                        df.at[idx, 'holding_duration'] = (date - pos['entry_date']).days
                        
                        pos['quantity'] -= remaining_qty
                        if pos['quantity'] == 0:
                            positions[symbol].pop(pos_idx)
                        remaining_qty = 0
                        matched = True
                        break
                    else:
                        # Partial match, continue
                        remaining_qty -= pos['quantity']
                        positions[symbol].pop(pos_idx)
                
                if not matched and remaining_qty > 0:
                    # Short sale or unmatched sell
                    logger.warning(f"Unmatched sell at index {idx}: {remaining_qty} shares")
        
        return df
    
    def _calculate_primitives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived behavioral primitives."""
        df = df.copy()
        
        # Trade sequence index (order of trades)
        df['trade_sequence'] = range(len(df))
        
        # Calculate P&L if not present
        if 'realized_pnl' not in df.columns or df['realized_pnl'].isna().all():
            df['realized_pnl'] = 0.0
            # Estimate P&L for sells
            sell_mask = df['side'] == 'sell'
            if 'entry_price' in df.columns:
                df.loc[sell_mask, 'realized_pnl'] = (
                    (df.loc[sell_mask, 'price'] - df.loc[sell_mask, 'entry_price']) *
                    df.loc[sell_mask, 'quantity']
                )
        
        # Entry vs exit efficiency (for sells)
        sell_mask = df['side'] == 'sell'
        if 'entry_price' in df.columns:
            df.loc[sell_mask, 'entry_exit_efficiency'] = (
                df.loc[sell_mask, 'price'] / (df.loc[sell_mask, 'entry_price'] + 1e-6)
            )
        
        # Trade value
        df['trade_value'] = df['quantity'] * df['price']
        
        # Cumulative P&L
        df['cumulative_pnl'] = df['realized_pnl'].cumsum()
        
        return df
    
    def get_cleaned_trades(self) -> pd.DataFrame:
        """Get cleaned trades."""
        if self.cleaned_trades is None:
            raise ValueError("No cleaned trades available. Call clean() first.")
        return self.cleaned_trades

