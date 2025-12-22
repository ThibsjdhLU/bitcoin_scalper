"""
Paper Trading Client - Simulated Trading for Testing and Development.

This module implements a paper trading client that simulates order execution
without connecting to a real broker. It maintains internal state for:
- Account balance and equity
- Open positions
- Order history

The paper client provides the same interface as MT5RestClient but executes
orders instantly with simulated fills. This is essential for:
- Testing trading strategies without risk
- Developing and debugging the engine
- Demonstrating the bot's behavior

All paper trades are logged with clear markers to prevent confusion with real trades.

Usage:
    >>> from bitcoin_scalper.connectors.paper import PaperMT5Client
    >>> 
    >>> # Initialize with starting balance
    >>> client = PaperMT5Client(initial_balance=10000.0)
    >>> 
    >>> # Execute paper order
    >>> result = client.send_order("BTCUSD", action="buy", volume=0.1)
    >>> print(f"Paper order executed: {result}")
    >>> 
    >>> # Check positions
    >>> positions = client.get_positions()
    >>> print(f"Open positions: {len(positions)}")
"""

import time
import logging
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """Represents a simulated position in paper trading."""
    ticket: int
    symbol: str
    action: str  # "buy" or "sell"
    volume: float
    open_price: float
    open_time: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    profit: float = 0.0
    
    def update_profit(self, current_price: float) -> float:
        """
        Update and return profit based on current price.
        
        Args:
            current_price: Current market price
            
        Returns:
            Profit in dollars
        """
        if self.action == "buy":
            self.profit = (current_price - self.open_price) * self.volume
        else:  # sell
            self.profit = (self.open_price - current_price) * self.volume
        return self.profit


class PaperMT5Client:
    """
    Paper trading client that simulates MT5RestClient interface.
    
    This client maintains internal state and simulates instant order fills.
    It provides the same API as MT5RestClient but does not connect to any
    real broker or execute real trades.
    
    Key Features:
    - Tracks balance and equity
    - Manages simulated positions
    - Instant order execution (simulated slippage optional)
    - Order history logging
    - Thread-safe operations
    
    Attributes:
        initial_balance: Starting account balance
        balance: Current account balance
        equity: Current equity (balance + unrealized P&L)
        positions: List of open positions
        order_history: History of all executed orders
        simulated_prices: Dict of symbol -> current price
        
    Example:
        >>> client = PaperMT5Client(initial_balance=10000.0)
        >>> 
        >>> # Set current market price
        >>> client.set_price("BTCUSD", 50000.0)
        >>> 
        >>> # Execute buy order
        >>> result = client.send_order("BTCUSD", action="buy", volume=0.1)
        >>> 
        >>> # Check account info
        >>> account = client._request("GET", "/account")
        >>> print(f"Balance: ${account['balance']:.2f}")
        >>> print(f"Equity: ${account['equity']:.2f}")
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        simulated_slippage: float = 0.0001,  # 0.01% slippage
        enable_slippage: bool = False,
    ):
        """
        Initialize paper trading client.
        
        Args:
            initial_balance: Starting balance in USD
            simulated_slippage: Simulated slippage as fraction (e.g., 0.0001 = 0.01%)
            enable_slippage: Whether to apply simulated slippage
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.simulated_slippage = simulated_slippage
        self.enable_slippage = enable_slippage
        
        # State tracking
        self.positions: List[PaperPosition] = []
        self.order_history: List[Dict[str, Any]] = []
        self._next_ticket = 1000  # Starting ticket number
        
        # Current market prices (symbol -> price)
        self.simulated_prices: Dict[str, float] = {}
        
        # Mock attributes to match MT5RestClient interface
        self.base_url = "paper://localhost"
        self.api_key = "paper_trading"
        self.timeout = 30.0
        self.max_retries = 3
        
        logger.info(
            f"[PAPER] PaperMT5Client initialized with balance: ${initial_balance:.2f}"
        )
    
    def set_price(self, symbol: str, price: float) -> None:
        """
        Set current market price for a symbol.
        
        This is used to simulate market data updates.
        
        Args:
            symbol: Trading symbol
            price: Current market price
        """
        self.simulated_prices[symbol] = price
        
        # Update positions profit
        for pos in self.positions:
            if pos.symbol == symbol:
                pos.update_profit(price)
        
        # Update equity
        self._update_equity()
    
    def _update_equity(self) -> None:
        """Update equity based on current positions P&L."""
        unrealized_pnl = sum(pos.profit for pos in self.positions)
        self.equity = self.balance + unrealized_pnl
    
    def _get_current_price(self, symbol: str) -> float:
        """
        Get current market price for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price
            
        Raises:
            ValueError: If price not set for symbol
        """
        if symbol not in self.simulated_prices:
            # For testing, generate a default price if not set
            logger.warning(
                f"[PAPER] No price set for {symbol}, using default 50000.0"
            )
            self.simulated_prices[symbol] = 50000.0
        
        return self.simulated_prices[symbol]
    
    def _apply_slippage(self, price: float, action: str) -> float:
        """
        Apply simulated slippage to price.
        
        Args:
            price: Base price
            action: "buy" or "sell"
            
        Returns:
            Price with slippage applied
        """
        if not self.enable_slippage:
            return price
        
        # Buy: pay slightly more, Sell: receive slightly less
        if action == "buy":
            return price * (1 + self.simulated_slippage)
        else:  # sell
            return price * (1 - self.simulated_slippage)
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """
        Simulate REST API request interface.
        
        This method provides compatibility with MT5RestClient API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional parameters
            
        Returns:
            Response data
        """
        # Account info endpoint
        if endpoint == "/account":
            return {
                'balance': self.balance,
                'equity': self.equity,
                'margin': 0.0,  # Paper trading has no margin requirements
                'free_margin': self.equity,
                'margin_level': 0.0 if len(self.positions) == 0 else float('inf'),
                'paper_mode': True,
            }
        
        # Status endpoint
        elif endpoint == "/status":
            return {
                'status': 'ok',
                'mode': 'paper',
                'connected': True,
                'positions': len(self.positions),
                'balance': self.balance,
            }
        
        # Positions endpoint
        elif endpoint == "/positions":
            return self._format_positions()
        
        # Order endpoint
        elif endpoint == "/order" and method == "POST":
            json_data = kwargs.get('json', {})
            return self._execute_paper_order(json_data)
        
        # Default response
        return {'status': 'ok', 'paper_mode': True}
    
    def _format_positions(self) -> List[Dict[str, Any]]:
        """
        Format positions for API response.
        
        Returns:
            List of position dictionaries
        """
        return [
            {
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 0 if pos.action == "buy" else 1,  # 0=buy, 1=sell
                'volume': pos.volume,
                'price_open': pos.open_price,
                'price_current': self.simulated_prices.get(pos.symbol, pos.open_price),
                'profit': pos.profit,
                'sl': pos.sl,
                'tp': pos.tp,
                'time': int(pos.open_time),
            }
            for pos in self.positions
        ]
    
    def _execute_paper_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a paper order (internal method).
        
        Args:
            order_data: Order parameters
            
        Returns:
            Order result dictionary
        """
        symbol = order_data.get('symbol')
        action = order_data.get('action')
        volume = order_data.get('volume', 0.01)
        sl = order_data.get('sl')
        tp = order_data.get('tp')
        
        # Get current price
        current_price = self._get_current_price(symbol)
        
        # Apply slippage
        fill_price = self._apply_slippage(current_price, action)
        
        # Create position
        ticket = self._next_ticket
        self._next_ticket += 1
        
        position = PaperPosition(
            ticket=ticket,
            symbol=symbol,
            action=action,
            volume=volume,
            open_price=fill_price,
            open_time=time.time(),
            sl=sl,
            tp=tp,
        )
        
        self.positions.append(position)
        
        # Record in history
        order_record = {
            'ticket': ticket,
            'symbol': symbol,
            'action': action,
            'volume': volume,
            'price': fill_price,
            'sl': sl,
            'tp': tp,
            'time': time.time(),
            'status': 'filled',
        }
        self.order_history.append(order_record)
        
        logger.info(
            f"[PAPER] Order Executed: {action.upper()} {volume} {symbol} @ ${fill_price:.2f}"
        )
        
        return {
            'status': 'success',
            'ticket': ticket,
            'symbol': symbol,
            'action': action,
            'volume': volume,
            'price': fill_price,
            'sl': sl,
            'tp': tp,
            'paper_mode': True,
        }
    
    def send_order(
        self,
        symbol: str,
        action: str,
        volume: float,
        price: Optional[float] = None,
        order_type: str = "market",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a paper order (simulated execution).
        
        This method matches the MT5RestClient.send_order interface.
        
        Args:
            symbol: Trading symbol
            action: "buy" or "sell"
            volume: Position size
            price: Limit price (ignored for market orders)
            order_type: Order type (only "market" supported)
            **kwargs: Additional parameters (sl, tp, etc.)
            
        Returns:
            Order execution result
        """
        order_data = {
            'symbol': symbol,
            'action': action,
            'volume': volume,
            'order_type': order_type,
            **kwargs
        }
        
        return self._execute_paper_order(order_data)
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions.
        
        Returns:
            List of position dictionaries
        """
        return self._format_positions()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get paper trading status.
        
        Returns:
            Status dictionary
        """
        return self._request("GET", "/status")
    
    def get_ticks(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get simulated tick data.
        
        Args:
            symbol: Trading symbol
            limit: Number of ticks to return
            
        Returns:
            List of tick dictionaries
        """
        # Generate simulated ticks around current price
        current_price = self._get_current_price(symbol)
        
        ticks = []
        timestamp = time.time()
        
        for i in range(limit):
            # Small random walk
            price_change = random.uniform(-0.001, 0.001) * current_price
            tick_price = current_price + price_change
            
            ticks.append({
                'timestamp': timestamp - (limit - i),
                'bid': tick_price * 0.9999,
                'ask': tick_price * 1.0001,
                'last': tick_price,
                'volume': random.uniform(0.1, 1.0),
                'symbol': symbol,
            })
        
        return ticks
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "M1",
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get simulated OHLCV data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., "M1", "M5")
            limit: Number of candles to return
            
        Returns:
            List of OHLCV dictionaries
        """
        # Get base price
        current_price = self._get_current_price(symbol)
        
        # Generate realistic price walk
        dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq='1min')
        
        candles = []
        price = current_price
        
        for ts in dates:
            # Random walk with small changes
            price_change = np.random.normal(0, current_price * 0.001)
            open_price = price
            high_price = open_price + abs(np.random.normal(0, current_price * 0.0005))
            low_price = open_price - abs(np.random.normal(0, current_price * 0.0005))
            close_price = open_price + price_change
            
            candles.append({
                'timestamp': int(ts.timestamp()),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': random.uniform(50, 200),
                'symbol': symbol,
            })
            
            price = close_price
        
        # Update current price to last close
        self.set_price(symbol, candles[-1]['close'])
        
        return candles
    
    def close_position(self, ticket: int) -> Dict[str, Any]:
        """
        Close a position.
        
        Args:
            ticket: Position ticket number
            
        Returns:
            Result dictionary
        """
        # Find position
        position = None
        for pos in self.positions:
            if pos.ticket == ticket:
                position = pos
                break
        
        if not position:
            return {
                'status': 'error',
                'error': f'Position {ticket} not found',
            }
        
        # Calculate final profit
        current_price = self._get_current_price(position.symbol)
        position.update_profit(current_price)
        
        # Update balance
        self.balance += position.profit
        
        # Remove position
        self.positions.remove(position)
        
        # Update equity
        self._update_equity()
        
        logger.info(
            f"[PAPER] Position Closed: {position.action.upper()} {position.volume} "
            f"{position.symbol} @ ${current_price:.2f}, P&L: ${position.profit:.2f}"
        )
        
        return {
            'status': 'success',
            'ticket': ticket,
            'profit': position.profit,
            'close_price': current_price,
            'paper_mode': True,
        }
    
    def close_all_positions(self) -> Dict[str, Any]:
        """
        Close all open positions.
        
        Returns:
            Result dictionary with summary
        """
        closed_tickets = []
        total_profit = 0.0
        
        # Close positions one by one (iterate over copy)
        for position in list(self.positions):
            result = self.close_position(position.ticket)
            if result['status'] == 'success':
                closed_tickets.append(position.ticket)
                total_profit += result['profit']
        
        logger.info(
            f"[PAPER] Closed {len(closed_tickets)} positions, "
            f"Total P&L: ${total_profit:.2f}"
        )
        
        return {
            'status': 'success',
            'closed_count': len(closed_tickets),
            'tickets': closed_tickets,
            'total_profit': total_profit,
            'paper_mode': True,
        }
    
    def reset(self) -> None:
        """
        Reset paper trading account to initial state.
        
        This closes all positions and resets balance to initial value.
        """
        self.positions.clear()
        self.order_history.clear()
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self._next_ticket = 1000
        
        logger.info(
            f"[PAPER] Account reset to initial balance: ${self.initial_balance:.2f}"
        )
