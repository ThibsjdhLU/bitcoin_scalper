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
    >>> # Fetch market data (default: 1500 candles for proper indicator calculation)
    >>> df = connector.fetch_ohlcv("BTC/USDT", timeframe="1m", limit=1500)
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

from bitcoin_scalper.core.data_requirements import DEFAULT_FETCH_LIMIT

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
        limit: int = DEFAULT_FETCH_LIMIT
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance.
        
        **CRITICAL**: Returns a DataFrame with standardized lowercase column names:
        ['date', 'open', 'high', 'low', 'close', 'volume']
        
        The 'date' column is set as the index and contains datetime objects.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "1m", "5m", "1h")
            limit: Number of candles to fetch (default: 1500 for proper feature engineering)
        
        Returns:
            DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume']
            The 'date' column is set as index with datetime objects.
            
        Raises:
            Exception: If fetching data fails
            
        Example:
            >>> df = connector.fetch_ohlcv("BTC/USDT", "1m", 1500)
            >>> print(df.columns)
            Index(['open', 'high', 'low', 'close', 'volume'])
            >>> print(df.index.name)
            date
        """
        try:
            logger.debug(f"Fetching OHLCV: {symbol}, {timeframe}, limit={limit}")
            max_per_request = 1000
            if limit is not None and int(limit) > max_per_request:
                logger.debug(
                    f"Requested limit ({limit}) > per-request cap ({max_per_request}), "
                    "delegating to fetch_ohlcv_historical()"
                )
                return self.fetch_ohlcv_historical(symbol, timeframe, limit)

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
    
    def fetch_ohlcv_historical( self, symbol: str, timeframe: str = "1m", limit: int = 5000) -> pd.DataFrame:
        """
        Robust paginated fetch for historical OHLCV.
    
        Fetches in batches (max_per_request) and pages backward in time until
        `limit` rows are collected or no more history is available.
    
        Returns a DataFrame indexed by datetime (date) with columns:
        ['open','high','low','close','volume']
        """
        try:
            max_per_request = 1000  # Binance typical per-request cap
            limit = int(limit or max_per_request)
    
            # Convert timeframe to milliseconds
            try:
                timeframe_ms = int(self.exchange.parse_timeframe(timeframe) * 1000)
            except Exception:
                # Fallback (1m)
                timeframe_ms = 60 * 1000
    
            all_rows = []
            requests = 0
    
            logger.info(f"Starting paginated fetch for {symbol} {timeframe} (target={limit})")
    
            # We start by fetching the most recent batch, then page older batches
            since = None  # None -> most recent
            while len(all_rows) < limit:
                batch_limit = min(max_per_request, limit - len(all_rows))
                requests += 1
                logger.debug(f"Paginated fetch: request #{requests}, batch_limit={batch_limit}, since={since}")
    
                try:
                    batch = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=since,
                        limit=batch_limit
                    )
                except Exception as e:
                    logger.error(f"fetch_ohlcv failed on batch #{requests}: {e}")
                    raise
    
                if not batch:
                    logger.info("No more data returned by exchange during pagination.")
                    break
    
                # batch is typically ordered oldest -> newest
                logger.info(f"Batch #{requests}: received {len(batch)} candles (ts range: {batch[0][0]} -> {batch[-1][0]})")
                all_rows.extend(batch)
    
                # If returned less than requested, likely no more history.
                if len(batch) < batch_limit:
                    logger.info("Received fewer candles than requested for this batch; stopping pagination.")
                    break
    
                # Prepare since for next batch to fetch older candles:
                # Use oldest timestamp in current batch and shift backwards by batch_limit * timeframe_ms
                oldest_ts = int(batch[0][0])
                next_since = oldest_ts - (batch_limit * timeframe_ms)
                # Avoid negative since
                if next_since < 0:
                    logger.info("Reached beginning of epoch; stopping pagination.")
                    break
                since = next_since
    
                # Respect exchange rateLimit (ccxt exposes in ms)
                sleep_ms = getattr(self.exchange, "rateLimit", 200)
                time_sleep = max(0.05, sleep_ms / 1000.0)
                logger.debug(f"Sleeping {time_sleep:.3f}s to respect rateLimit")
                import time
                time.sleep(time_sleep)
    
            if not all_rows:
                logger.warning(f"No historical OHLCV data returned for {symbol}")
                return pd.DataFrame()
    
            # Deduplicate by timestamp and sort ascending (oldest first)
            # Flattened rows may already be chronological, but ensure correctness
            # Convert to dict keyed by timestamp to dedupe
            unique_map = {}
            for row in all_rows:
                ts = int(row[0])
                unique_map[ts] = row  # last assignment keeps latest occurrence (shouldn't duplicate)
            unique_rows = [unique_map[k] for k in sorted(unique_map.keys())]
    
            # If we collected more than needed, take the most recent `limit` rows
            if len(unique_rows) > limit:
                unique_rows = unique_rows[-limit:]
    
            # Build DataFrame
            df = pd.DataFrame(unique_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            df = df.drop(columns=['timestamp'])
            df = df.set_index('date')
    
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
            # Final dedupe & sorting just in case
            df = df[~df.index.duplicated(keep='first')]
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
    
            logger.info(f"Fetched total of {len(df)} historical candles for {symbol} (requested {limit})")
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
    
    def get_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = DEFAULT_FETCH_LIMIT) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV data and return as list of dicts for compatibility with MT5RestClient.

        This wrapper will automatically use the paginated historical fetcher when the
        requested `limit` exceeds the typical per-request cap for Binance (~1000).
        """
        try:
            # Normalize limit
            limit = int(limit or DEFAULT_FETCH_LIMIT)

            # Typical per-request cap on Binance/CCXT
            max_per_request = 1000

            # Debug/logging
            logger.debug(f"get_ohlcv called: symbol={symbol}, timeframe={timeframe}, limit={limit}")

            # Use historical fetcher for large requests (> max_per_request)
            if limit > max_per_request:
                logger.info(
                    f"Requested limit ({limit}) > per-request cap ({max_per_request}), "
                    "delegating to fetch_ohlcv_historical() to page through history."
                )
                df = self.fetch_ohlcv_historical(symbol, timeframe, limit)
            else:
                df = self.fetch_ohlcv(symbol, timeframe, limit)

            # Ensure a DataFrame was returned and convert to list of dicts
            if df is None:
                logger.warning("get_ohlcv: underlying fetch returned None")
                return []
            if getattr(df, "empty", False):
                logger.warning("get_ohlcv: underlying fetch returned an empty DataFrame")
                return []

            # Convert DataFrame to list of dicts (compatibility format)
            df_reset = df.reset_index()

            # Ensure 'date' column is present and serializable (if index name different, leave as-is)
            # Return as list[dict]
            return df_reset.to_dict("records")

        except Exception as e:
            logger.error(f"Failed to get OHLCV in get_ohlcv(): {e}")
            raise
    
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
