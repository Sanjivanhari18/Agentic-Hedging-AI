"""CSV tradebook loader for prototype stage."""

import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CSVTradebookLoader:
    """Loads tradebook data from CSV files."""
    
    REQUIRED_COLUMNS = ['date', 'symbol', 'side', 'quantity', 'price']
    OPTIONAL_COLUMNS = ['order_type', 'realized_pnl', 'commission', 'fees']
    
    def __init__(self):
        self.trades: Optional[pd.DataFrame] = None
    
    def load(self, filepath: str) -> pd.DataFrame:
        """
        Load tradebook from CSV file.
        
        Expected columns:
        - date: Trade date
        - symbol: Stock symbol
        - side: 'buy' or 'sell'
        - quantity: Number of shares
        - price: Trade price
        - order_type: (optional) Market, Limit, etc.
        - realized_pnl: (optional) Realized profit/loss
        - commission: (optional) Commission paid
        - fees: (optional) Other fees
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            
            # Normalize column names (case-insensitive)
            df.columns = df.columns.str.lower().str.strip()
            
            # Validate required columns
            missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            # Ensure date is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Normalize side column
            df['side'] = df['side'].str.lower().str.strip()
            if not df['side'].isin(['buy', 'sell']).all():
                raise ValueError("'side' column must contain only 'buy' or 'sell'")
            
            # Ensure numeric columns
            numeric_cols = ['quantity', 'price']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            self.trades = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def get_trades(self) -> pd.DataFrame:
        """Get loaded trades."""
        if self.trades is None:
            raise ValueError("No trades loaded. Call load() first.")
        return self.trades

