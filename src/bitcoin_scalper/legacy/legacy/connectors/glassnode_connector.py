"""
Glassnode connector for Bitcoin on-chain analytics and metrics.

Glassnode provides comprehensive on-chain analytics for Bitcoin and other
cryptocurrencies, offering insights into network activity, holder behavior,
and fundamental metrics that complement market microstructure data.

Website: https://glassnode.com/
Features:
- Extensive on-chain metrics (MVRV, SOPR, etc.)
- Exchange flow data
- Network health indicators
- Holder and entity analytics
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd
import logging

from .base import DataSource

logger = logging.getLogger(__name__)


class GlassnodeConnector(DataSource):
    """
    Connector for Glassnode on-chain analytics.
    
    Glassnode specializes in on-chain metrics that provide fundamental insights
    into Bitcoin network activity and holder behavior. These metrics are crucial
    for regime detection and fundamental analysis in crypto trading strategies.
    
    Key metrics available:
    - MVRV (Market Value to Realized Value) Z-Score
    - SOPR (Spent Output Profit Ratio)
    - Exchange netflows (inflows/outflows)
    - Active addresses and entities
    - Supply distribution metrics
    
    Attributes:
        api_key: Glassnode API key.
        base_url: Base URL for Glassnode API.
        
    Example:
        >>> connector = GlassnodeConnector(api_key="your-api-key")
        >>> mvrv = connector.fetch_onchain_metrics(
        ...     metric="mvrv_z_score",
        ...     start_time=datetime(2024, 1, 1)
        ... )
        
    References:
        Glassnode API Documentation: https://docs.glassnode.com/
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.glassnode.com",
        rate_limit: float = 1.0
    ):
        """
        Initialize Glassnode connector.
        
        Args:
            api_key: Glassnode API key.
                    Sign up at https://glassnode.com/ for API access.
            base_url: Base URL for API endpoints.
            rate_limit: Maximum requests per second (Glassnode has strict limits).
        """
        super().__init__(
            name="Glassnode",
            api_key=api_key,
            base_url=base_url,
            rate_limit=rate_limit
        )
    
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
        
        Args:
            symbol: Trading pair symbol.
            start_time: Start of time range.
            end_time: End of time range.
            depth: Number of price levels.
            **kwargs: Additional parameters.
        
        Returns:
            DataFrame with order book snapshots.
            
        Raises:
            NotImplementedError: Glassnode does not provide order book data.
            
        Notes:
            Glassnode focuses on on-chain metrics, not exchange market data.
            Use CoinApiConnector or KaikoConnector for order book data.
        """
        raise NotImplementedError(
            "Glassnode does not provide order book data. "
            "Use CoinApiConnector or KaikoConnector for L2 market data."
        )
    
    def fetch_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch trade data.
        
        Args:
            symbol: Trading pair symbol.
            start_time: Start of time range.
            end_time: End of time range.
            limit: Maximum number of trades.
            **kwargs: Additional parameters.
        
        Returns:
            DataFrame with trade data.
            
        Raises:
            NotImplementedError: Glassnode does not provide trade data.
            
        Notes:
            Glassnode focuses on on-chain metrics, not exchange market data.
            Use CoinApiConnector or KaikoConnector for trade data.
        """
        raise NotImplementedError(
            "Glassnode does not provide trade data. "
            "Use CoinApiConnector or KaikoConnector for trade data."
        )
    
    def fetch_onchain_metrics(
        self,
        metric: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch on-chain metrics from Glassnode.
        
        Args:
            metric: Metric name from Glassnode's catalog. Common metrics:
                - "mvrv": Market Value to Realized Value ratio
                - "mvrv_z_score": MVRV Z-Score (normalized)
                - "sopr": Spent Output Profit Ratio
                - "exchange_net_flow": Net flow to/from exchanges
                - "active_addresses": Number of active addresses
                - "transaction_volume": On-chain transaction volume
                - "nvt": Network Value to Transaction ratio
                - "difficulty_ribbon": Mining difficulty metrics
            start_time: Start of time range.
            end_time: End of time range.
            **kwargs: Additional parameters including:
                - asset: Asset symbol (default "BTC")
                - interval: Data interval ("24h", "1h", "10m")
                - currency: Quote currency for value metrics (default "USD")
        
        Returns:
            DataFrame with metric data containing:
            - timestamp: Metric timestamp (Unix seconds)
            - value: Metric value
            - metric_name: Name of the metric
            
        Raises:
            NotImplementedError: Implementation pending API key configuration.
            
        Notes:
            This is a blueprint method. Implementation requires:
            1. Valid Glassnode API key
            2. HTTP client with API key authentication
            3. Rate limiting (strict limits on free/lower tiers)
            4. Proper timestamp conversion (Unix to datetime)
            5. Metric name validation and normalization
            
            Endpoint: GET /v1/metrics/{category}/{metric}
            
            Example metric paths:
            - /v1/metrics/market/mvrv_z_score
            - /v1/metrics/indicators/sopr
            - /v1/metrics/transactions/transfers_volume_sum
            
            Key on-chain metrics for trading:
            
            1. MVRV Z-Score:
               - High values (>3) indicate overvaluation/bubble territory
               - Low values (<0) indicate undervaluation/accumulation zone
               - Use for regime detection and macro positioning
            
            2. SOPR:
               - >1: Holders selling at profit (potential resistance)
               - <1: Holders selling at loss (potential capitulation)
               - Trend reversals at 1.0 are significant in bull markets
            
            3. Exchange Netflows:
               - Positive (inflows): Potential selling pressure
               - Negative (outflows): Accumulation, less supply on exchanges
               - Large sudden inflows often precede volatility
        """
        raise NotImplementedError(
            "Glassnode connector requires valid API key and implementation. "
            "To implement:\n"
            "1. Configure API key authentication (URL parameter: ?api_key=...)\n"
            "2. Map metric name to Glassnode endpoint path\n"
            "3. Make request to /v1/metrics/{category}/{metric}\n"
            "4. Convert Unix timestamps to datetime\n"
            "5. Parse JSON response to DataFrame format\n"
            "6. Implement rate limiting (respect tier limits)\n"
            "7. Handle pagination if needed\n"
            f"Required metric: {metric}"
        )
    
    def get_available_metrics(self) -> List[str]:
        """
        Get list of available on-chain metrics.
        
        Returns:
            List of metric names available through this connector.
            
        Notes:
            This is a helper method to discover available metrics.
            Implementation should query Glassnode's metric catalog or
            return a predefined list of commonly used metrics.
        """
        # Common metrics that should be implemented
        common_metrics = [
            "mvrv",
            "mvrv_z_score",
            "sopr",
            "sopr_adjusted",
            "exchange_net_flow",
            "exchange_balance",
            "active_addresses",
            "transaction_volume",
            "nvt",
            "nvt_signal",
            "difficulty_ribbon",
            "puell_multiple",
            "reserve_risk",
            "hodl_waves"
        ]
        
        logger.info(
            f"Common Glassnode metrics: {len(common_metrics)} metrics available. "
            "Full catalog requires API call to /v1/metrics/endpoints"
        )
        
        return common_metrics
