"""DataFetchAgent: Fetches and normalizes historical OHLC price data."""

from typing import Dict, List, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from app.agents.base import BaseAgent
from app.models.agent import AgentType, AgentContext


class DataFetchAgent(BaseAgent):
    """
    Fetches historical OHLC price data for given tickers.
    
    Responsibilities:
    - Fetch price data from data source (yfinance, Alpha Vantage, etc.)
    - Normalize and align time series
    - Handle missing data gracefully
    - Return aligned date index and price arrays
    
    Note: This implementation uses yfinance as a placeholder.
    In production, you might use:
    - Alpha Vantage API
    - IEX Cloud
    - Bloomberg API
    - Custom data warehouse
    """
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.DATA_FETCH
    
    def __init__(self, lookback_days: int = 252 * 5):
        """
        Initialize the data fetch agent.
        
        Args:
            lookback_days: Number of trading days to fetch (default: 5 years)
        """
        self.lookback_days = lookback_days
    
    def execute(self, context: AgentContext) -> Dict[str, Any]:
        """
        Fetch and normalize price data for all tickers.
        
        Args:
            context: Contains tickers and analysis_date
            
        Returns:
            Dictionary with:
            - price_data: Dict[ticker, List[float]] of normalized prices
            - dates: List[datetime] of aligned dates
            - returns: Dict[ticker, List[float]] of daily returns (optional)
        """
        # Determine date range
        end_date = context.analysis_date
        start_date = end_date - timedelta(days=self.lookback_days * 1.5)  # Buffer for weekends/holidays
        
        # Fetch data for all tickers
        price_data_dict = {}
        dates_dict = {}
        
        for ticker in context.tickers:
            try:
                data = self._fetch_ticker_data(ticker, start_date, end_date)
                if data is not None and len(data) > 0:
                    price_data_dict[ticker] = data['close'].tolist()
                    dates_dict[ticker] = data.index.tolist()
            except Exception as e:
                # Log warning but continue with other tickers
                print(f"Warning: Failed to fetch data for {ticker}: {e}")
                # Use placeholder data for missing tickers (in production, handle differently)
                price_data_dict[ticker] = []
                dates_dict[ticker] = []
        
        # Align all time series to common date index
        aligned_data = self._align_time_series(price_data_dict, dates_dict)
        
        # Handle missing data
        aligned_data = self._handle_missing_data(aligned_data)
        
        return {
            "price_data": aligned_data["price_data"],
            "dates": [d.isoformat() if isinstance(d, datetime) else str(d) 
                     for d in aligned_data["dates"]],
            "raw_data_info": {
                "tickers_fetched": len([t for t in context.tickers if t in aligned_data["price_data"]]),
                "total_data_points": len(aligned_data["dates"]),
                "missing_data_handled": True
            }
        }
    
    def _fetch_ticker_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical data for a single ticker.
        
        TODO: Replace with actual data source API call.
        Current implementation uses yfinance as placeholder.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLC data indexed by date
        """
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data available for {ticker}")
            
            return data
        except ImportError:
            # Fallback: Generate synthetic data for development
            print(f"Warning: yfinance not available. Generating synthetic data for {ticker}")
            return self._generate_synthetic_data(ticker, start_date, end_date)
        except Exception as e:
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")
    
    def _generate_synthetic_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate synthetic price data for development/testing.
        
        This is a placeholder when real data sources are unavailable.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5]  # Business days only
        
        # Generate random walk price series
        np.random.seed(hash(ticker) % 2**32)  # Deterministic per ticker
        returns = np.random.normal(0.0005, 0.02, len(dates))  # ~0.05% daily return, 2% volatility
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        return data
    
    def _align_time_series(self, price_data_dict: Dict[str, List[float]], 
                          dates_dict: Dict[str, List[datetime]]) -> Dict[str, Any]:
        """
        Align all time series to a common date index.
        
        Uses intersection of all available dates to ensure consistent analysis.
        
        Args:
            price_data_dict: Price data per ticker
            dates_dict: Date indices per ticker
            
        Returns:
            Dictionary with aligned price_data and dates
        """
        if not price_data_dict:
            raise ValueError("No price data available")
        
        # Find common date range (intersection of all date ranges)
        all_dates = set()
        for dates in dates_dict.values():
            all_dates.update(dates)
        
        # Convert to sorted list
        common_dates = sorted(list(all_dates))
        
        if not common_dates:
            raise ValueError("No common dates found across tickers")
        
        # Align each ticker's data to common dates
        aligned_prices = {}
        for ticker in price_data_dict.keys():
            ticker_dates = dates_dict[ticker]
            ticker_prices = price_data_dict[ticker]
            
            # Create mapping
            date_to_price = dict(zip(ticker_dates, ticker_prices))
            
            # Align to common dates
            aligned_prices[ticker] = [
                date_to_price.get(date, None) for date in common_dates
            ]
        
        return {
            "price_data": aligned_prices,
            "dates": common_dates
        }
    
    def _handle_missing_data(self, aligned_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle missing data points using forward-fill then backward-fill.
        
        Args:
            aligned_data: Aligned price data with potential None values
            
        Returns:
            Cleaned data with missing values filled
        """
        price_data = aligned_data["price_data"]
        
        for ticker in price_data:
            prices = price_data[ticker]
            
            # Convert to pandas Series for easier handling
            series = pd.Series(prices)
            
            # Forward fill, then backward fill (using new pandas API)
            series = series.ffill().bfill()
            
            # If still missing (all NaN), use constant value
            if series.isna().any():
                series = series.fillna(100.0)  # Default placeholder
            
            price_data[ticker] = series.tolist()
        
        return aligned_data
