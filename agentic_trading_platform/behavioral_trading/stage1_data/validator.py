"""Tradebook data validation."""

import pandas as pd
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class TradebookValidator:
    """Validates tradebook data quality."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, trades: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate tradebook data.
        
        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        self.errors = []
        self.warnings = []
        
        # Required columns check
        required = ['date', 'symbol', 'side', 'quantity', 'price']
        missing = set(required) - set(trades.columns)
        if missing:
            self.errors.append(f"Missing required columns: {missing}")
            return {'errors': self.errors, 'warnings': self.warnings}
        
        # Check for duplicates
        duplicates = trades.duplicated().sum()
        if duplicates > 0:
            self.warnings.append(f"Found {duplicates} duplicate rows")
        
        # Check for missing values
        for col in required:
            null_count = trades[col].isna().sum()
            if null_count > 0:
                self.errors.append(f"Column '{col}' has {null_count} missing values")
        
        # Check timestamp consistency
        if 'date' in trades.columns:
            invalid_dates = trades['date'].isna().sum()
            if invalid_dates > 0:
                self.errors.append(f"Found {invalid_dates} invalid dates")
        
        # Check side values
        if 'side' in trades.columns:
            invalid_sides = ~trades['side'].isin(['buy', 'sell'])
            if invalid_sides.any():
                self.errors.append(f"Found {invalid_sides.sum()} invalid side values")
        
        # Check numeric values
        for col in ['quantity', 'price']:
            if col in trades.columns:
                negative = (trades[col] < 0).sum()
                if negative > 0:
                    self.warnings.append(f"Found {negative} negative values in '{col}'")
        
        # Check for zero quantities/prices
        if 'quantity' in trades.columns:
            zero_qty = (trades['quantity'] == 0).sum()
            if zero_qty > 0:
                self.errors.append(f"Found {zero_qty} trades with zero quantity")
        
        if 'price' in trades.columns:
            zero_price = (trades['price'] == 0).sum()
            if zero_price > 0:
                self.errors.append(f"Found {zero_price} trades with zero price")
        
        return {'errors': self.errors, 'warnings': self.warnings}
    
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

