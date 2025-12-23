"""
Binance Public API Client for Historical Data Fetching.

This module provides a client for fetching historical market data from Binance
using the public API (no authentication required). Designed specifically for
data collection and training purposes.

Key Features:
- No API keys required (uses public endpoints)
- Automatic pagination for large date ranges
- Returns standardized DataFrame format
- Rate limiting to respect exchange limits

Usage:
    >>> from bitcoin_scalper.connectors.binance_public import BinancePublicClient
    >>> 
    >>> client = BinancePublicClient()
    >>> df = client.fetch_history(
    ...     symbol="BTC/USDT",
    ...     timeframe="1m",
    ...     start_date="2024-01-01",
    ...     end_date="2024-01-31"
    ... )
    >>> print(f"Fetched {len(df)} candles")
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)


class BinancePublicClient:
    """
    Public API client for fetching historical data from Binance.
    
    This client uses CCXT's public endpoints to fetch historical OHLCV data
    without requiring API credentials. It handles pagination automatically
    when the requested date range exceeds Binance's 5000-candle limit per request.
    
    Attributes:
        exchange: CCXT Binance exchange instance (public mode)
        rate_limit_delay: Delay between requests in seconds (default: 1.0)
        
    Example:
        >>> client = BinancePublicClient()
        >>> df = client.fetch_history("BTC/USDT", "1h", "2024-01-01", "2024-12-31")
        >>> print(df.head())
    """
    
    def __init__(self, rate_limit_delay: float = 1.0):
        """
        Initialize Binance public API client.
        
        Args:
            rate_limit_delay: Delay between requests in seconds to respect rate limits
        """
        self.rate_limit_delay = rate_limit_delay
        
        logger.info("Initializing BinancePublicClient (public API, no auth)")
        
        try:
            # Initialize CCXT Binance exchange in public mode
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                }
            })
            
            # Load markets
            self.exchange.load_markets()
            logger.info("Successfully connected to Binance public API")
            
        except Exception as e:
            logger.error(f"Failed to initialize Binance public client: {e}")
            raise
    
    def fetch_history(
        self,
        symbol: str,
        timeframe: str = "1m",
        start_date: str = None,
        end_date: str = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance.
        
        This method automatically handles pagination when the date range exceeds
        Binance's limit of 5000 candles per request. It will make multiple requests
        to fetch all data in the specified range.
        
        **CRITICAL**: Returns a DataFrame with standardized lowercase column names:
        ['date', 'open', 'high', 'low', 'close', 'volume']
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT", "ETH/USDT")
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h", "1d")
            start_date: Start date as string (e.g., "2024-01-01" or "2024-01-01 00:00:00")
            end_date: End date as string (e.g., "2024-12-31" or "2024-12-31 23:59:59")
            limit: Maximum number of candles to fetch (overrides date range if specified)
        
        Returns:
            DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume']
            The 'date' column is set as index with datetime objects.
            
        Raises:
            Exception: If fetching data fails
            
        Example:
            >>> client = BinancePublicClient()
            >>> # Fetch 1 year of hourly data
            >>> df = client.fetch_history("BTC/USDT", "1h", "2024-01-01", "2024-12-31")
            >>> print(f"Fetched {len(df)} candles")
            >>> 
            >>> # Fetch last 5000 candles (no dates needed)
            >>> df = client.fetch_history("BTC/USDT", "1m", limit=5000)
        """
        try:
            logger.info(f"Fetching historical data for {symbol} ({timeframe})")
            
            # If limit is specified, use simple fetch
            if limit is not None:
                logger.info(f"Fetching last {limit} candles")
                return self._fetch_simple(symbol, timeframe, limit)
            
            # Convert date strings to timestamps
            if start_date is None:
                # Default to 1 year ago
                start_dt = datetime.now() - timedelta(days=365)
            else:
                start_dt = pd.to_datetime(start_date)
            
            if end_date is None:
                # Default to now
                end_dt = datetime.now()
            else:
                end_dt = pd.to_datetime(end_date)
            
            logger.info(f"Date range: {start_dt} to {end_dt}")
            
            # Fetch data with pagination
            df = self._fetch_paginated(symbol, timeframe, start_dt, end_dt)
            
            if df.empty:
                logger.warning("No data fetched")
                return df
            
            logger.info(f"Successfully fetched {len(df)} candles")
            logger.info(f"Data range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            raise
    
    def _fetch_simple(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> pd.DataFrame:
        """
        Fetch a simple batch of candles (no pagination).
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=min(limit, 5000)  # Binance limit is 5000
            )
            
            if not ohlcv:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = self._ohlcv_to_dataframe(ohlcv)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise
    
    def _fetch_paginated(
        self,
        symbol: str,
        timeframe: str,
        start_dt: datetime,
        end_dt: datetime
    ) -> pd.DataFrame:
        """
        Fetch data with automatic pagination for large date ranges.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            DataFrame with all OHLCV data in the range
        """
        all_data = []
        current_time = start_dt
        end_timestamp = int(end_dt.timestamp() * 5000)
        
        request_count = 0
        max_requests = 5000  # Safety limit to prevent infinite loops
        
        logger.info("Starting paginated fetch...")
        
        while current_time < end_dt and request_count < max_requests:
            try:
                # Convert to timestamp (milliseconds)
                since = int(current_time.timestamp() * 5000)
                
                # Fetch batch
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=5000  # Binance max limit
                )
                
                if not ohlcv:
                    logger.info("No more data available")
                    break
                
                # Add to results
                all_data.extend(ohlcv)
                
                # Update current time to last candle timestamp
                last_timestamp = ohlcv[-1][0]
                
                # If we've reached or passed the end time, stop
                if last_timestamp >= end_timestamp:
                    break
                
                # Move to next batch (add 1ms to avoid fetching same candle)
                current_time = datetime.fromtimestamp((last_timestamp + 1) / 5000)
                
                request_count += 1
                
                # Log progress every 10 requests
                if request_count % 10 == 0:
                    logger.info(f"Progress: {request_count} requests, {len(all_data)} candles fetched")
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error during paginated fetch: {e}")
                if all_data:
                    logger.warning("Returning partial data due to error")
                    break
                else:
                    raise
        
        if request_count >= max_requests:
            logger.warning(f"Hit maximum request limit ({max_requests})")
        
        if not all_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = self._ohlcv_to_dataframe(all_data)
        
        # Filter to exact date range
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        
        # Remove duplicates (can occur at boundaries)
        df = df[~df.index.duplicated(keep='first')]
        
        logger.info(f"Completed paginated fetch: {request_count} requests, {len(df)} candles")
        
        return df
    
    def _ohlcv_to_dataframe(self, ohlcv: list) -> pd.DataFrame:
        """
        Convert OHLCV data to standardized DataFrame.
        
        Args:
            ohlcv: List of OHLCV data from CCXT
            
        Returns:
            DataFrame with standardized columns
        """
        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Drop timestamp column
        df = df.drop(columns=['timestamp'])
        
        # Set date as index
        df = df.set_index('date')
        
        # Ensure all numeric columns are float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df
