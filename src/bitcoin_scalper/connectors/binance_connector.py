"""
Binance Connector using CCXT - Exchange Integration for Live Trading.

This module implements a Binance exchange connector using the CCXT library.
It provides a standardized interface for:
- Fetching OHLCV market data
- Executing market orders
- Retrieving account balance

The connector returns data in a standardized format with lowercase column names:
['date', 'open', 'high', 'low', 'close', 'volume']

This ensures compatibility with the feature engineering pipeline while
migrating away from MT5 to a modern crypto exchange.

Usage:
    >>> from bitcoin_scalper.connectors.binance_connector import BinanceConnector
    >>> 
    >>> # Initialize connector
    >>> connector = BinanceConnector(
    ...     api_key="your_api_key",
    ...     api_secret="your_api_secret",
    ...     testnet=True  # Use testnet for testing
    ... )
    >>> 
    >>> # Fetch market data
    >>> df = connector.fetch_ohlcv("BTC/USDT", timeframe="1m", limit=100)
    >>> print(df.head())
    >>> 
    >>> # Execute order
    >>> result = connector.execute_order("BTC/USDT", "buy", 0.001)
    >>> 
    >>> # Get balance
    >>> balance = connector.get_balance("USDT")
"""

import ccxt
import pandas as pd
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class BinanceConnector:
    """
    Binance exchange connector using CCXT library.
    
    Provides a clean interface for interacting with Binance exchange:
    - Market data fetching with standardized DataFrame output
    - Order execution with proper error handling
    - Account balance retrieval
    
    The connector supports both live trading and testnet mode for safe testing.
    All data is returned in a standardized format compatible with the
    feature engineering pipeline.
    
    **Compatibility Methods:**
    This connector implements both native methods and compatibility methods
    to work with the MT5RestClient interface:
    - fetch_ohlcv() / get_ohlcv() - Market data
    - execute_order() / send_order() - Order execution
    - get_account_info() / _request() - Account info
    
    Attributes:
        api_key: Binance API key
        api_secret: Binance API secret
        testnet: Whether to use testnet (default: True for safety)
        market_type: Market type - "spot" or "future" (default: "spot")
        exchange: CCXT Binance exchange instance
        
    Example:
        >>> connector = BinanceConnector(api_key="key", api_secret="secret", testnet=True)
        >>> df = connector.fetch_ohlcv("BTC/USDT", "1m", 100)
        >>> print(df.columns)
        Index(['date', 'open', 'high', 'low', 'close', 'volume'])
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        market_type: str = "spot"  # "spot" or "future"
    ):
        """
        Initialize Binance connector.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet if True (default: True for safety)
            market_type: Market type - "spot" or "future" (default: "spot")
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.market_type = market_type
        
        logger.info(f"Initializing BinanceConnector (testnet={testnet}, market_type={market_type})")
        
        try:
            # Initialize CCXT Binance exchange
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,  # Enable rate limiting
                'options': {
                    'defaultType': market_type,
                }
            })
            
            # Set testnet URLs if testnet mode
            if testnet:
                self.exchange.set_sandbox_mode(True)
                logger.info("Binance testnet mode enabled")
            
            # Load markets
            self.exchange.load_markets()
            logger.info(f"Successfully connected to Binance ({'testnet' if testnet else 'mainnet'})")
            
        except Exception as e:
            logger.error(f"Failed to initialize Binance connector: {e}")
            raise
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance.
        
        **CRITICAL**: Returns a DataFrame with standardized lowercase column names:
        ['date', 'open', 'high', 'low', 'close', 'volume']
        
        The 'date' column is set as the index and contains datetime objects.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h")
            limit: Number of candles to fetch (default: 100)
        
        Returns:
            DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume']
            The 'date' column is set as index with datetime objects.
            
        Raises:
            Exception: If fetching data fails
            
        Example:
            >>> df = connector.fetch_ohlcv("BTC/USDT", "1m", 100)
            >>> print(df.columns)
            Index(['open', 'high', 'low', 'close', 'volume'])
            >>> print(df.index.name)
            date
        """
        try:
            logger.debug(f"Fetching OHLCV: {symbol}, {timeframe}, limit={limit}")
            
            # Fetch OHLCV data from Binance
            # CCXT returns: [[timestamp, open, high, low, close, volume], ...]
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            if not ohlcv:
                logger.warning(f"No OHLCV data returned for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame with standardized column names
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp (milliseconds) to datetime
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Drop the raw timestamp column
            df = df.drop(columns=['timestamp'])
            
            # Set date as index
            df = df.set_index('date')
            
            # Ensure all numeric columns are float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            logger.info(f"Fetched {len(df)} candles for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data: {e}")
            raise
    
    def fetch_ohlcv_historical(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 5000
    ) -> pd.DataFrame:
        """
        Fetch large amounts of historical OHLCV data by making multiple requests.
        
        Binance limits single requests to ~1000-1500 candles. This method
        makes multiple requests and concatenates them to get more history.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h")
            limit: Total number of candles desired (will fetch in batches)
        
        Returns:
            DataFrame with historical OHLCV data
        """
        try:
            # Binance typical max limit per request
            max_per_request = 1000
            
            if limit <= max_per_request:
                # Single request is sufficient
                return self.fetch_ohlcv(symbol, timeframe, limit)
            
            # Calculate number of batches needed
            num_batches = (limit + max_per_request - 1) // max_per_request
            
            all_data = []
            since = None  # Start from most recent
            
            logger.info(f"Fetching {limit} candles in {num_batches} batches...")
            
            for batch in range(num_batches):
                # Fetch batch
                batch_limit = min(max_per_request, limit - len(all_data))
                
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=batch_limit,
                    since=since
                )
                
                if not ohlcv:
                    break
                
                # Collect data (will reverse at the end for chronological order)
                all_data.append(ohlcv)
                
                # Set 'since' to the timestamp of the first candle (oldest)
                # for the next batch to fetch older data
                since = ohlcv[0][0] - 1
                
                # Stop if we got less than requested (no more history available)
                if len(ohlcv) < batch_limit:
                    break
                
                logger.debug(f"Batch {batch + 1}/{num_batches}: Fetched {len(ohlcv)} candles")
                
                # Add small delay between batches to avoid rate limiting
                if batch < num_batches - 1:  # Don't delay after last batch
                    import time
                    time.sleep(0.1)
            
            if not all_data:
                logger.warning(f"No historical OHLCV data returned for {symbol}")
                return pd.DataFrame()
            
            # Flatten and reverse to get chronological order (oldest first)
            # We fetched newest first, then went backwards in time
            flat_data = []
            for batch_data in reversed(all_data):
                flat_data.extend(batch_data)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                flat_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop(columns=['timestamp'])
            df = df.set_index('date')
            
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Remove duplicates (keep first occurrence) - data should already be mostly sorted
            # but duplicates may occur at batch boundaries
            df = df[~df.index.duplicated(keep='first')]
            
            # Ensure chronological order
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
            
            logger.info(f"Fetched total of {len(df)} historical candles for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical OHLCV data: {e}")
            raise
    
    def execute_order(
        self,
        symbol: str,
        side: str,
        volume: float,
        order_type: str = "market",
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a trading order on Binance.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            side: Order side ("buy" or "sell")
            volume: Order size (amount in base currency)
            order_type: Order type ("market" or "limit")
            price: Limit price (required for limit orders)
            **kwargs: Additional order parameters (e.g., stopPrice, timeInForce)
        
        Returns:
            Dict with order result containing:
            - id: Order ID
            - symbol: Trading pair
            - type: Order type
            - side: Order side
            - price: Execution price
            - amount: Order amount
            - filled: Filled amount
            - remaining: Remaining amount
            - status: Order status
            - timestamp: Execution timestamp
            
        Raises:
            Exception: If order execution fails
            
        Example:
            >>> result = connector.execute_order("BTC/USDT", "buy", 0.001)
            >>> print(f"Order ID: {result['id']}")
        """
        try:
            logger.info(f"Executing order: {side} {volume} {symbol} ({order_type})")
            
            # Validate side
            if side not in ['buy', 'sell']:
                raise ValueError(f"Invalid order side: {side}. Must be 'buy' or 'sell'")
            
            # Create order parameters
            params = kwargs.copy()
            
            # Execute order via CCXT
            if order_type == "market":
                order = self.exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=volume,
                    params=params
                )
            elif order_type == "limit":
                if price is None:
                    raise ValueError("Limit orders require a price")
                order = self.exchange.create_limit_order(
                    symbol=symbol,
                    side=side,
                    amount=volume,
                    price=price,
                    params=params
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            logger.info(
                f"Order executed successfully: ID={order.get('id')}, "
                f"Status={order.get('status')}, Filled={order.get('filled', 'N/A')}"
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to execute order: {e}")
            raise
    
    def get_balance(self, currency: str = "USDT") -> float:
        """
        Get free balance for a specific currency.
        
        Args:
            currency: Currency code (default: "USDT")
        
        Returns:
            Free balance amount
            
        Raises:
            Exception: If fetching balance fails
            
        Example:
            >>> balance = connector.get_balance("USDT")
            >>> print(f"Free USDT: {balance}")
        """
        try:
            logger.debug(f"Fetching balance for {currency}")
            
            # Fetch account balance
            balance = self.exchange.fetch_balance()
            
            # Get free balance for the currency
            free_balance = balance.get('free', {}).get(currency, 0.0)
            
            logger.info(f"Free {currency} balance: {free_balance}")
            
            return float(free_balance)
            
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            raise
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict with account information including balances for all currencies
            
        Example:
            >>> info = connector.get_account_info()
            >>> print(info['free'])  # All free balances
        """
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Failed to fetch account info: {e}")
            raise
    
    # Compatibility methods to match MT5RestClient interface
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Generic request method for compatibility with MT5RestClient interface.
        
        This method provides a unified interface for internal requests.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional parameters
            
        Returns:
            Dict with response data
        """
        if endpoint == "/account":
            # Return account info in expected format
            balance_info = self.exchange.fetch_balance()
            total = balance_info.get('total', {})
            free = balance_info.get('free', {})
            
            # Get USDT balance or first available currency
            usdt_balance = total.get('USDT', 0.0)
            usdt_free = free.get('USDT', 0.0)
            
            return {
                'balance': float(usdt_balance),
                'equity': float(usdt_balance),  # For crypto, equity = balance
                'free_margin': float(usdt_free),
                'margin_level': 100.0,  # Spot trading doesn't use margin
            }
        else:
            raise NotImplementedError(f"Endpoint {endpoint} not implemented")
    
    def get_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV data and return as list of dicts for compatibility with MT5RestClient.
        
        This is a compatibility wrapper that intelligently handles large requests
        by using fetch_ohlcv_historical when needed.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h")
            limit: Number of candles to fetch
            
        Returns:
            List of dicts with OHLCV data
        """
        # Use historical fetcher for large requests (>1000 candles)
        if limit > 1000:
            df = self.fetch_ohlcv_historical(symbol, timeframe, limit)
        else:
            df = self.fetch_ohlcv(symbol, timeframe, limit)
        
        if df.empty:
            return []
        
        # Convert DataFrame to list of dicts
        df_reset = df.reset_index()
        return df_reset.to_dict('records')
    
    def send_order(self, symbol: str, action: str, volume: float, 
                   price: Optional[float] = None, order_type: str = "market", **kwargs) -> Dict[str, Any]:
        """
        Send order for compatibility with MT5RestClient interface.
        
        This is a compatibility wrapper around execute_order.
        
        Args:
            symbol: Trading pair symbol
            action: Order action ("buy" or "sell")
            volume: Order size
            price: Limit price (optional)
            order_type: Order type
            **kwargs: Additional parameters
            
        Returns:
            Dict with order result
        """
        return self.execute_order(symbol, action, volume, order_type, price, **kwargs)
    
    def close(self):
        """Close the exchange connection."""
        try:
            if hasattr(self.exchange, 'close'):
                self.exchange.close()
            logger.info("Binance connector closed")
        except Exception as e:
            logger.warning(f"Error closing connector: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
