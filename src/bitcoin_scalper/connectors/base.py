"""
Abstract base class for data source connectors.

This module defines the interface that all data source connectors must implement,
ensuring consistent API across different data providers (CoinAPI, Kaiko, Glassnode, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """
    Abstract base class for market data connectors.
    
    All data source implementations must inherit from this class and implement
    the required abstract methods. This ensures a consistent interface for:
    - Level 2 order book data
    - Trade data
    - On-chain metrics
    
    The abstract methods serve as a strict blueprint for connector implementations,
    ensuring that the data pipeline can work with any provider without modifications.
    
    Attributes:
        name: Human-readable name of the data source.
        api_key: API key for authentication (optional).
        base_url: Base URL for API endpoints.
        rate_limit: Maximum requests per second (for rate limiting).
        
    Example:
        >>> class MyCustomConnector(DataSource):
        ...     def __init__(self, api_key: str):
        ...         super().__init__(name="MyProvider", api_key=api_key)
        ...     
        ...     def fetch_l2_data(self, symbol: str, **kwargs):
        ...         # Implementation here
        ...         pass
    """
    
    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        rate_limit: float = 10.0
    ):
        """
        Initialize data source connector.
        
        Args:
            name: Name of the data source provider.
            api_key: API authentication key (if required).
            base_url: Base URL for API endpoints.
            rate_limit: Maximum requests per second (default 10).
        """
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limit = rate_limit
        self._logger = logger.getChild(name)
        
    @abstractmethod
    def fetch_l2_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        depth: int = 50,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch Level 2 order book data.
        
        Level 2 data provides aggregated order book snapshots at various price levels,
        essential for analyzing market depth and liquidity.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC-USD", "BTCUSDT").
            start_time: Start of time range (None for most recent).
            end_time: End of time range (None for most recent).
            depth: Number of price levels to fetch per side (default 50).
            **kwargs: Additional provider-specific parameters.
        
        Returns:
            DataFrame with order book snapshots containing:
            - timestamp: Snapshot timestamp
            - bids: List of {'price': float, 'volume': float} dicts
            - asks: List of {'price': float, 'volume': float} dicts
            - symbol: Trading pair
            
        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError(
            f"{self.name} must implement fetch_l2_data method"
        )
    
    @abstractmethod
    def fetch_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch trade (tick) data.
        
        Trade data contains individual executed transactions, providing the finest
        granularity for market analysis and feature engineering.
        
        Args:
            symbol: Trading pair symbol.
            start_time: Start of time range.
            end_time: End of time range.
            limit: Maximum number of trades to fetch (if supported).
            **kwargs: Additional provider-specific parameters.
        
        Returns:
            DataFrame with trade data containing:
            - timestamp: Trade execution time
            - price: Execution price
            - volume: Trade volume (size)
            - side: Trade side ('buy' or 'sell', if available)
            - trade_id: Unique trade identifier (if available)
            - symbol: Trading pair
            
        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError(
            f"{self.name} must implement fetch_trades method"
        )
    
    @abstractmethod
    def fetch_onchain_metrics(
        self,
        metric: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch on-chain metrics data.
        
        On-chain metrics provide fundamental insights into network activity,
        holder behavior, and macro trends that complement market microstructure data.
        
        Common metrics:
        - MVRV (Market Value to Realized Value)
        - SOPR (Spent Output Profit Ratio)
        - Exchange netflows (inflows/outflows)
        - Active addresses
        - Transaction volume
        
        Args:
            metric: Name of the metric to fetch (provider-specific).
            start_time: Start of time range.
            end_time: End of time range.
            **kwargs: Additional provider-specific parameters.
        
        Returns:
            DataFrame with metric data containing:
            - timestamp: Metric timestamp
            - value: Metric value
            - metric_name: Name of the metric
            Additional columns may vary by metric and provider.
            
        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError(
            f"{self.name} must implement fetch_onchain_metrics method"
        )
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate that a symbol is supported by this data source.
        
        Args:
            symbol: Trading pair symbol to validate.
        
        Returns:
            True if symbol is valid, False otherwise.
            
        Notes:
            Default implementation returns True. Override in subclass
            to implement actual validation logic.
        """
        return True
    
    def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported trading pair symbols.
        
        Returns:
            List of supported symbol strings.
            
        Notes:
            Default implementation returns empty list. Override in subclass
            to provide actual list of supported symbols.
        """
        return []
    
    def check_connection(self) -> bool:
        """
        Check if connection to data source is working.
        
        Returns:
            True if connection is successful, False otherwise.
            
        Notes:
            Default implementation returns True. Override in subclass
            to implement actual connection check (e.g., API health endpoint).
        """
        return True
    
    def __repr__(self) -> str:
        """String representation of the data source."""
        return f"{self.__class__.__name__}(name='{self.name}')"
