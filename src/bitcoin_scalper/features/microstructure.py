"""
Microstructure features for order book and trade flow analysis.

This module implements advanced market microstructure features used in high-frequency
and algorithmic trading strategies. These features capture price formation dynamics,
liquidity conditions, and order flow information.

References:
    - Cont, R., Kukanov, A., & Stoikov, S. (2014). "The Price Impact of Order Book Events"
    - Cartea, Á., Jaimungal, S., & Penalva, J. (2015). "Algorithmic and High-Frequency Trading"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class OrderFlowImbalance:
    """
    Calculate Order Flow Imbalance (OFI) from order book updates.
    
    OFI measures the net buying or selling pressure at the best bid/ask levels.
    Research shows OFI has >80% feature importance in short-term price prediction
    models, making it one of the most predictive microstructure signals.
    
    The OFI at time t is calculated as:
        OFI_t = e_t * q_t
    
    Where:
        - e_t captures the direction of the order flow change
        - q_t represents the quantity involved in the change
    
    More specifically, for best bid/ask changes:
        OFI_t = (ΔBid_volume * I(ΔBid_price ≥ 0)) - (ΔAsk_volume * I(ΔAsk_price ≤ 0))
    
    Positive OFI indicates buying pressure, negative indicates selling pressure.
    
    Example:
        >>> ofi_calc = OrderFlowImbalance()
        >>> # Feed order book snapshots sequentially
        >>> ofi_values = []
        >>> for snapshot in orderbook_stream:
        ...     ofi = ofi_calc.calculate(snapshot)
        ...     if ofi is not None:
        ...         ofi_values.append(ofi)
    
    References:
        Cont, R., Kukanov, A., & Stoikov, S. (2014). 
        "The Price Impact of Order Book Events". Journal of Financial Econometrics.
    """
    
    def __init__(self):
        """Initialize Order Flow Imbalance calculator."""
        self.prev_snapshot: Optional[Dict[str, Any]] = None
        
    def calculate(
        self,
        current_snapshot: Dict[str, Any],
        bid_key: str = 'best_bid',
        ask_key: str = 'best_ask',
        bid_volume_key: str = 'best_bid_volume',
        ask_volume_key: str = 'best_ask_volume'
    ) -> Optional[float]:
        """
        Calculate OFI from consecutive order book snapshots.
        
        Args:
            current_snapshot: Current order book state with structure:
                {
                    'best_bid': float,
                    'best_ask': float,
                    'best_bid_volume': float,
                    'best_ask_volume': float,
                    ...
                }
            bid_key: Key for best bid price.
            ask_key: Key for best ask price.
            bid_volume_key: Key for best bid volume.
            ask_volume_key: Key for best ask volume.
        
        Returns:
            OFI value (float) or None if this is the first snapshot or data is invalid.
            Positive values indicate buying pressure, negative indicate selling pressure.
            
        Notes:
            - Returns None on first call (need previous state for comparison)
            - Stores current snapshot for next calculation
            - Handles missing data gracefully by returning None
        """
        # Extract current values
        try:
            curr_bid_price = current_snapshot.get(bid_key)
            curr_ask_price = current_snapshot.get(ask_key)
            curr_bid_volume = current_snapshot.get(bid_volume_key)
            curr_ask_volume = current_snapshot.get(ask_volume_key)
            
            # Validate current snapshot
            if any(v is None for v in [curr_bid_price, curr_ask_price, 
                                       curr_bid_volume, curr_ask_volume]):
                logger.debug("Current snapshot has missing values")
                return None
            
            # Need previous snapshot for comparison
            if self.prev_snapshot is None:
                self.prev_snapshot = current_snapshot.copy()
                return None
            
            # Extract previous values
            prev_bid_price = self.prev_snapshot.get(bid_key)
            prev_ask_price = self.prev_snapshot.get(ask_key)
            prev_bid_volume = self.prev_snapshot.get(bid_volume_key)
            prev_ask_volume = self.prev_snapshot.get(ask_volume_key)
            
            # Validate previous snapshot
            if any(v is None for v in [prev_bid_price, prev_ask_price,
                                       prev_bid_volume, prev_ask_volume]):
                logger.debug("Previous snapshot has missing values")
                self.prev_snapshot = current_snapshot.copy()
                return None
            
            # Calculate price changes
            bid_price_change = curr_bid_price - prev_bid_price
            ask_price_change = curr_ask_price - prev_ask_price
            
            # Calculate volume changes
            bid_volume_change = curr_bid_volume - prev_bid_volume
            ask_volume_change = curr_ask_volume - prev_ask_volume
            
            # Calculate OFI components
            # Bid side: positive contribution if bid price increased or stayed same
            bid_ofi = 0.0
            if bid_price_change >= 0:
                bid_ofi = bid_volume_change
            elif bid_price_change < 0:
                # Bid price decreased: negative contribution
                bid_ofi = -prev_bid_volume
            
            # Ask side: negative contribution if ask price decreased or stayed same
            ask_ofi = 0.0
            if ask_price_change <= 0:
                ask_ofi = -ask_volume_change
            elif ask_price_change > 0:
                # Ask price increased: positive contribution (less selling pressure)
                ask_ofi = prev_ask_volume
            
            # Total OFI
            ofi = bid_ofi + ask_ofi
            
            # Store current snapshot for next iteration
            self.prev_snapshot = current_snapshot.copy()
            
            return float(ofi)
            
        except Exception as e:
            logger.error(f"Error calculating OFI: {e}")
            return None
    
    def calculate_from_series(
        self,
        snapshots: List[Dict[str, Any]],
        **kwargs
    ) -> pd.Series:
        """
        Calculate OFI for a series of order book snapshots.
        
        Args:
            snapshots: List of order book snapshots in chronological order.
            **kwargs: Additional arguments passed to calculate().
        
        Returns:
            Series of OFI values with same length as input.
            First value is NaN (requires previous snapshot).
            
        Example:
            >>> ofi_calc = OrderFlowImbalance()
            >>> snapshots = [snapshot1, snapshot2, snapshot3, ...]
            >>> ofi_series = ofi_calc.calculate_from_series(snapshots)
        """
        self.prev_snapshot = None  # Reset state
        ofi_values = []
        
        for snapshot in snapshots:
            ofi = self.calculate(snapshot, **kwargs)
            ofi_values.append(ofi)
        
        return pd.Series(ofi_values)
    
    def reset(self):
        """Reset the calculator state (clear previous snapshot)."""
        self.prev_snapshot = None


class OrderBookDepthAnalyzer:
    """
    Analyze order book depth and liquidity distribution.
    
    Provides metrics for understanding liquidity concentration and depth beyond
    the best bid/ask. This is crucial for:
    - Estimating price impact of large orders
    - Detecting support/resistance levels
    - Assessing market stability
    
    Key metrics:
    - Liquidity concentration ratios (e.g., top 5 vs top 50 levels)
    - Weighted average depth
    - Imbalance at various depths
    
    Example:
        >>> depth_analyzer = OrderBookDepthAnalyzer(levels=50)
        >>> orderbook = {
        ...     'bids': [{'price': 100, 'volume': 10}, ...],
        ...     'asks': [{'price': 101, 'volume': 8}, ...]
        ... }
        >>> metrics = depth_analyzer.analyze(orderbook)
        >>> print(f"Liquidity concentration: {metrics['concentration_ratio']:.2%}")
    
    References:
        Cartea, Á., Jaimungal, S., & Penalva, J. (2015).
        "Algorithmic and High-Frequency Trading"
    """
    
    def __init__(self, levels: int = 50):
        """
        Initialize Order Book Depth Analyzer.
        
        Args:
            levels: Number of price levels to analyze on each side.
                   Typical values: 10 (shallow), 50 (standard), 100 (deep).
        """
        self.levels = levels
    
    def analyze(
        self,
        orderbook: Dict[str, List[Dict[str, float]]],
        bids_key: str = 'bids',
        asks_key: str = 'asks',
        top_levels: int = 5
    ) -> Dict[str, float]:
        """
        Analyze order book depth and calculate liquidity metrics.
        
        Args:
            orderbook: Order book data with structure:
                {
                    'bids': [
                        {'price': float, 'volume': float},
                        ...
                    ],
                    'asks': [
                        {'price': float, 'volume': float},
                        ...
                    ]
                }
                Bids should be sorted descending (best first)
                Asks should be sorted ascending (best first)
            bids_key: Key for bids list.
            asks_key: Key for asks list.
            top_levels: Number of top levels for concentration ratio (default 5).
        
        Returns:
            Dictionary containing:
            - 'bid_depth': Total bid volume across all levels
            - 'ask_depth': Total ask volume across all levels
            - 'total_depth': Total volume on both sides
            - 'depth_imbalance': (bid_depth - ask_depth) / total_depth
            - 'concentration_ratio': Volume in top N levels / total volume
            - 'weighted_bid_price': Volume-weighted average bid price
            - 'weighted_ask_price': Volume-weighted average ask price
            - 'spread_to_depth_ratio': Spread as fraction of total depth
            - 'bid_levels': Number of valid bid levels
            - 'ask_levels': Number of valid ask levels
            
        Raises:
            ValueError: If orderbook structure is invalid.
        """
        try:
            bids = orderbook.get(bids_key, [])[:self.levels]
            asks = orderbook.get(asks_key, [])[:self.levels]
            
            if not bids or not asks:
                raise ValueError("Order book must have both bids and asks")
            
            # Extract volumes and prices
            bid_volumes = np.array([b.get('volume', 0) for b in bids])
            ask_volumes = np.array([a.get('volume', 0) for a in asks])
            bid_prices = np.array([b.get('price', 0) for b in bids])
            ask_prices = np.array([a.get('price', 0) for a in asks])
            
            # Calculate total depths
            bid_depth = np.sum(bid_volumes)
            ask_depth = np.sum(ask_volumes)
            total_depth = bid_depth + ask_depth
            
            # Depth imbalance
            depth_imbalance = 0.0
            if total_depth > 0:
                depth_imbalance = (bid_depth - ask_depth) / total_depth
            
            # Concentration ratio (liquidity in top N levels vs all levels)
            top_bid_volume = np.sum(bid_volumes[:top_levels])
            top_ask_volume = np.sum(ask_volumes[:top_levels])
            concentration_ratio = 0.0
            if total_depth > 0:
                concentration_ratio = (top_bid_volume + top_ask_volume) / total_depth
            
            # Volume-weighted average prices
            weighted_bid_price = 0.0
            if bid_depth > 0:
                weighted_bid_price = np.average(bid_prices, weights=bid_volumes)
            
            weighted_ask_price = 0.0
            if ask_depth > 0:
                weighted_ask_price = np.average(ask_prices, weights=ask_volumes)
            
            # Spread to depth ratio
            best_bid = bid_prices[0] if len(bid_prices) > 0 else 0
            best_ask = ask_prices[0] if len(ask_prices) > 0 else 0
            spread = best_ask - best_bid
            spread_to_depth_ratio = 0.0
            if total_depth > 0:
                spread_to_depth_ratio = spread / total_depth
            
            return {
                'bid_depth': float(bid_depth),
                'ask_depth': float(ask_depth),
                'total_depth': float(total_depth),
                'depth_imbalance': float(depth_imbalance),
                'concentration_ratio': float(concentration_ratio),
                'weighted_bid_price': float(weighted_bid_price),
                'weighted_ask_price': float(weighted_ask_price),
                'spread_to_depth_ratio': float(spread_to_depth_ratio),
                'bid_levels': len(bids),
                'ask_levels': len(asks)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing order book depth: {e}")
            return {}
    
    def calculate_cumulative_depth(
        self,
        orderbook: Dict[str, List[Dict[str, float]]],
        bids_key: str = 'bids',
        asks_key: str = 'asks'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate cumulative depth profiles for visualization and analysis.
        
        Args:
            orderbook: Order book data.
            bids_key: Key for bids list.
            asks_key: Key for asks list.
        
        Returns:
            Tuple of (bid_profile, ask_profile) DataFrames with columns:
            - 'price': Price level
            - 'volume': Volume at this level
            - 'cumulative_volume': Cumulative volume from best price
            - 'distance_from_mid': Distance from mid price in basis points
        """
        bids = orderbook.get(bids_key, [])[:self.levels]
        asks = orderbook.get(asks_key, [])[:self.levels]
        
        best_bid = bids[0]['price'] if bids else 0
        best_ask = asks[0]['price'] if asks else 0
        mid_price = (best_bid + best_ask) / 2 if (best_bid and best_ask) else 0
        
        # Bid profile
        bid_data = []
        cumulative = 0.0
        for bid in bids:
            price = bid['price']
            volume = bid['volume']
            cumulative += volume
            distance_bps = ((price - mid_price) / mid_price * 10000) if mid_price else 0
            bid_data.append({
                'price': price,
                'volume': volume,
                'cumulative_volume': cumulative,
                'distance_from_mid': distance_bps
            })
        
        # Ask profile
        ask_data = []
        cumulative = 0.0
        for ask in asks:
            price = ask['price']
            volume = ask['volume']
            cumulative += volume
            distance_bps = ((price - mid_price) / mid_price * 10000) if mid_price else 0
            ask_data.append({
                'price': price,
                'volume': volume,
                'cumulative_volume': cumulative,
                'distance_from_mid': distance_bps
            })
        
        return pd.DataFrame(bid_data), pd.DataFrame(ask_data)


class VWAPSpreadCalculator:
    """
    Calculate Volume-Weighted Average Price (VWAP) Spread.
    
    VWAP Spread measures the cost of liquidity by comparing the volume-weighted
    average prices on both sides of the order book. This is more informative than
    simple bid-ask spread for understanding execution costs.
    
    VWAP Spread captures:
    - Effective transaction costs
    - Liquidity conditions
    - Implicit volatility expectations
    
    The VWAP spread can be calculated as:
        VWAP_Spread = (VWAP_Ask - VWAP_Bid) / Mid_Price
    
    Example:
        >>> vwap_calc = VWAPSpreadCalculator(levels=10)
        >>> orderbook = {...}  # Order book snapshot
        >>> spread_metrics = vwap_calc.calculate(orderbook)
        >>> print(f"VWAP Spread: {spread_metrics['vwap_spread_bps']:.2f} bps")
    
    References:
        Kissell, R. (2013). "The Science of Algorithmic Trading and Portfolio Management"
    """
    
    def __init__(self, levels: int = 10):
        """
        Initialize VWAP Spread Calculator.
        
        Args:
            levels: Number of order book levels to include in VWAP calculation.
                   More levels give better representation but may include noise.
        """
        self.levels = levels
    
    def calculate(
        self,
        orderbook: Dict[str, List[Dict[str, float]]],
        bids_key: str = 'bids',
        asks_key: str = 'asks'
    ) -> Dict[str, float]:
        """
        Calculate VWAP-based spread metrics from order book.
        
        Args:
            orderbook: Order book data with structure:
                {
                    'bids': [{'price': float, 'volume': float}, ...],
                    'asks': [{'price': float, 'volume': float}, ...]
                }
            bids_key: Key for bids list.
            asks_key: Key for asks list.
        
        Returns:
            Dictionary containing:
            - 'vwap_bid': Volume-weighted average bid price
            - 'vwap_ask': Volume-weighted average ask price
            - 'vwap_spread': Absolute VWAP spread (vwap_ask - vwap_bid)
            - 'vwap_spread_bps': VWAP spread in basis points (0.01%)
            - 'mid_price': Mid price (best_bid + best_ask) / 2
            - 'simple_spread': Simple bid-ask spread
            - 'simple_spread_bps': Simple spread in basis points
            - 'spread_ratio': VWAP spread / Simple spread
            
        Raises:
            ValueError: If orderbook is invalid or empty.
        """
        try:
            bids = orderbook.get(bids_key, [])[:self.levels]
            asks = orderbook.get(asks_key, [])[:self.levels]
            
            if not bids or not asks:
                raise ValueError("Order book must have both bids and asks")
            
            # Extract data
            bid_prices = np.array([b.get('price', 0) for b in bids])
            bid_volumes = np.array([b.get('volume', 0) for b in bids])
            ask_prices = np.array([a.get('price', 0) for a in asks])
            ask_volumes = np.array([a.get('volume', 0) for a in asks])
            
            # Calculate VWAPs
            total_bid_volume = np.sum(bid_volumes)
            total_ask_volume = np.sum(ask_volumes)
            
            if total_bid_volume == 0 or total_ask_volume == 0:
                raise ValueError("Zero volume in order book")
            
            vwap_bid = np.average(bid_prices, weights=bid_volumes)
            vwap_ask = np.average(ask_prices, weights=ask_volumes)
            
            # Best prices
            best_bid = bid_prices[0]
            best_ask = ask_prices[0]
            mid_price = (best_bid + best_ask) / 2
            
            # Calculate spreads
            vwap_spread = vwap_ask - vwap_bid
            vwap_spread_bps = (vwap_spread / mid_price) * 10000 if mid_price > 0 else 0
            
            simple_spread = best_ask - best_bid
            simple_spread_bps = (simple_spread / mid_price) * 10000 if mid_price > 0 else 0
            
            # Spread ratio (how much wider is VWAP spread vs simple spread)
            spread_ratio = vwap_spread / simple_spread if simple_spread > 0 else 1.0
            
            return {
                'vwap_bid': float(vwap_bid),
                'vwap_ask': float(vwap_ask),
                'vwap_spread': float(vwap_spread),
                'vwap_spread_bps': float(vwap_spread_bps),
                'mid_price': float(mid_price),
                'simple_spread': float(simple_spread),
                'simple_spread_bps': float(simple_spread_bps),
                'spread_ratio': float(spread_ratio)
            }
            
        except Exception as e:
            logger.error(f"Error calculating VWAP spread: {e}")
            return {}
    
    def calculate_time_series(
        self,
        orderbook_snapshots: List[Dict[str, Any]],
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate VWAP spread metrics for a time series of order book snapshots.
        
        Args:
            orderbook_snapshots: List of order book snapshots in chronological order.
            **kwargs: Additional arguments passed to calculate().
        
        Returns:
            DataFrame with one row per snapshot containing all VWAP spread metrics.
            
        Example:
            >>> vwap_calc = VWAPSpreadCalculator(levels=10)
            >>> snapshots = [snapshot1, snapshot2, ...]
            >>> spread_series = vwap_calc.calculate_time_series(snapshots)
            >>> spread_series['vwap_spread_bps'].plot()
        """
        results = []
        
        for snapshot in orderbook_snapshots:
            metrics = self.calculate(snapshot, **kwargs)
            results.append(metrics)
        
        return pd.DataFrame(results)
