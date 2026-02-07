"""PDF tradebook loader for production stage."""

import pdfplumber
import pandas as pd
from typing import Optional, List
import logging
import re

logger = logging.getLogger(__name__)


class PDFTradebookLoader:
    """Loads tradebook data from PDF broker statements."""
    
    def __init__(self):
        self.trades: Optional[pd.DataFrame] = None
    
    def load(self, filepath: str) -> pd.DataFrame:
        """
        Load tradebook from PDF file.
        
        Uses pdfplumber for table extraction with rule-based column detection.
        """
        try:
            tables = []
            
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    # Extract tables from page
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
            
            if not tables:
                raise ValueError("No tables found in PDF")
            
            # Convert tables to DataFrames
            dfs = []
            for table in tables:
                if len(table) < 2:  # Need at least header + 1 row
                    continue
                
                # First row as header
                header = [str(cell).strip().lower() if cell else '' for cell in table[0]]
                rows = table[1:]
                
                # Create DataFrame
                df = pd.DataFrame(rows, columns=header)
                dfs.append(df)
            
            if not dfs:
                raise ValueError("No valid tables found in PDF")
            
            # Combine all DataFrames
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Normalize column names (common broker formats)
            combined_df.columns = combined_df.columns.str.lower().str.strip()
            
            # Map common column name variations
            column_mapping = self._detect_column_mapping(combined_df.columns)
            combined_df = combined_df.rename(columns=column_mapping)
            
            # Clean and parse data
            combined_df = self._clean_data(combined_df)
            
            self.trades = combined_df
            logger.info(f"Loaded {len(combined_df)} rows from PDF")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise
    
    def _detect_column_mapping(self, columns: List[str]) -> dict:
        """Detect and map common broker column name variations."""
        mapping = {}
        
        # Date variations
        date_patterns = ['date', 'trade date', 'execution date', 'settlement date']
        for col in columns:
            if any(pattern in col for pattern in date_patterns):
                mapping[col] = 'date'
                break
        
        # Symbol variations
        symbol_patterns = ['symbol', 'ticker', 'security', 'instrument']
        for col in columns:
            if any(pattern in col for pattern in symbol_patterns):
                mapping[col] = 'symbol'
                break
        
        # Side variations
        side_patterns = ['side', 'buy/sell', 'action', 'transaction type']
        for col in columns:
            if any(pattern in col for pattern in side_patterns):
                mapping[col] = 'side'
                break
        
        # Quantity variations
        qty_patterns = ['quantity', 'qty', 'shares', 'units']
        for col in columns:
            if any(pattern in col for pattern in qty_patterns):
                mapping[col] = 'quantity'
                break
        
        # Price variations
        price_patterns = ['price', 'execution price', 'fill price', 'trade price']
        for col in columns:
            if any(pattern in col for pattern in price_patterns):
                mapping[col] = 'price'
                break
        
        return mapping
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and parse extracted data."""
        # Remove empty rows
        df = df.dropna(how='all')
        
        # Clean date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Clean side column
        if 'side' in df.columns:
            df['side'] = df['side'].astype(str).str.lower().str.strip()
            # Map common variations
            df['side'] = df['side'].replace({
                'b': 'buy', 's': 'sell',
                'buy': 'buy', 'sell': 'sell',
                'purchase': 'buy', 'sale': 'sell'
            })
        
        # Clean numeric columns
        numeric_cols = ['quantity', 'price']
        for col in numeric_cols:
            if col in df.columns:
                # Remove currency symbols and commas
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def get_trades(self) -> pd.DataFrame:
        """Get loaded trades."""
        if self.trades is None:
            raise ValueError("No trades loaded. Call load() first.")
        return self.trades

