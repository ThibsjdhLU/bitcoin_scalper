"""
Paper Trading Client - Simulated Trading for Testing and Development.

FIXED VERSION: Implements Persistent Random Walk to avoid history rewriting.
This ensures technical indicators remain stable between calls.
"""

import time
import logging
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

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
        """Update and return profit based on current price."""
        if self.action == "buy":
            self.profit = (current_price - self.open_price) * self.volume
        else:  # sell
            self.profit = (self.open_price - current_price) * self.volume
        return self.profit

class PaperMT5Client:
    """
    Paper trading client that simulates MT5RestClient interface with STABLE history.
    
    Changes in this version:
    - Persistent History: Generates history once and appends to it (no regeneration).
    - Realistic Walk: Uses Geometric Brownian Motion properties for more realistic volatility.
    - Time Sync: Aligns candles with real system clock.
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        simulated_slippage: float = 0.0001,  # 0.01% slippage
        enable_slippage: bool = False,
        initial_price: float = 50000.0
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.simulated_slippage = simulated_slippage
        self.enable_slippage = enable_slippage
        
        # State tracking
        self.positions: List[PaperPosition] = []
        self.order_history: List[Dict[str, Any]] = []
        self._next_ticket = 1000
        
        # --- Persistent Market Data Engine ---
        self.current_symbol = "BTC/USDT" # Default tracking
        self.history: List[Dict[str, Any]] = []
        self.simulated_prices: Dict[str, float] = {}
        
        # Init History (Pre-load 5000 candles to stabilize indicators)
        logger.info("[PAPER] Generating 5000 minutes of stable history...")
        self._init_history(initial_price, count=5000)
        
        # Mock attributes
        self.base_url = "paper://localhost"
        self.api_key = "paper_trading"
        self.timeout = 30.0
        self.max_retries = 3
        
        logger.info(f"[PAPER] Ready. Current simulated price: ${self._get_current_price(self.current_symbol):.2f}")
    
    def _init_history(self, start_price: float, count: int):
        """Generates a stable baseline history ending 'now'."""
        now = int(time.time())
        # Snap to nearest minute
        now = now - (now % 60)
        
        timestamps = [now - (i * 60) for i in range(count)]
        timestamps.reverse() # Oldest first
        
        price = start_price
        for ts in timestamps:
            candle, price = self._generate_next_candle(ts, price)
            self.history.append(candle)
            
        # Set current price
        self.simulated_prices[self.current_symbol] = self.history[-1]['close']

    def _generate_next_candle(self, timestamp: int, open_price: float) -> tuple:
        """Generates a single realistic candle from an open price."""
        # Volatility parameters (tuned for BTC-like moves)
        volatility_min = 0.0005  # 0.05% per minute
        trend = random.uniform(-0.0001, 0.0001) # Slight drift
        
        change_pct = np.random.normal(trend, volatility_min)
        close_price = open_price * (1 + change_pct)
        
        # High/Low generation
        move = abs(close_price - open_price)
        noise_h = abs(np.random.normal(0, volatility_min/2)) * open_price
        noise_l = abs(np.random.normal(0, volatility_min/2)) * open_price
        
        high_price = max(open_price, close_price) + noise_h
        low_price = min(open_price, close_price) - noise_l
        
        candle = {
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': random.uniform(50, 500), # Random volume
            'symbol': self.current_symbol
        }
        return candle, close_price

    def _update_market(self):
        """Checks time and appends new candles if a minute has passed."""
        if not self.history:
            return

        last_ts = self.history[-1]['timestamp']
        now = int(time.time())
        # Snap to minute
        # We allow a new candle only if the minute is fully completed or we simulate live
        # Here we simulate 'closed' candles logic, so we wait for minute change
        current_minute_ts = now - (now % 60)
        
        while last_ts + 60 <= current_minute_ts:
            next_ts = last_ts + 60
            last_close = self.history[-1]['close']
            new_candle, new_close = self._generate_next_candle(next_ts, last_close)
            
            self.history.append(new_candle)
            self.simulated_prices[self.current_symbol] = new_close
            last_ts = next_ts
            
            # Keep memory clean (keep last 10k candles)
            if len(self.history) > 10000:
                self.history.pop(0)

    def set_price(self, symbol: str, price: float) -> None:
        """Force set price (used by external tools, rarely needed now)."""
        self.simulated_prices[symbol] = price
        self._update_positions_pnl(symbol, price)
        self._update_equity()
    
    def _update_positions_pnl(self, symbol: str, price: float):
        for pos in self.positions:
            if pos.symbol == symbol:
                pos.update_profit(price)

    def _update_equity(self) -> None:
        unrealized_pnl = sum(pos.profit for pos in self.positions)
        self.equity = self.balance + unrealized_pnl
    
    def _get_current_price(self, symbol: str) -> float:
        self._update_market() # Ensure we are up to date
        return self.simulated_prices.get(symbol, 50000.0)
    
    def _apply_slippage(self, price: float, action: str) -> float:
        if not self.enable_slippage: return price
        if action == "buy": return price * (1 + self.simulated_slippage)
        else: return price * (1 - self.simulated_slippage)

    # --- API Interface (MT5 Compatible) ---

    def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        self._update_market() # Tick the clock
        
        if endpoint == "/account":
            return {
                'balance': self.balance,
                'equity': self.equity,
                'margin': 0.0,
                'free_margin': self.equity,
                'margin_level': 0.0 if len(self.positions) == 0 else float('inf'),
                'paper_mode': True,
            }
        elif endpoint == "/status":
            return {
                'status': 'ok', 'mode': 'paper', 'connected': True,
                'positions': len(self.positions), 'balance': self.balance
            }
        elif endpoint == "/positions":
            return self._format_positions()
        elif endpoint == "/order" and method == "POST":
            return self._execute_paper_order(kwargs.get('json', {}))
        
        return {'status': 'ok', 'paper_mode': True}
    
    def _format_positions(self) -> List[Dict[str, Any]]:
        current_price = self._get_current_price(self.current_symbol)
        return [
            {
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 0 if pos.action == "buy" else 1,
                'volume': pos.volume,
                'price_open': pos.open_price,
                'price_current': current_price,
                'profit': pos.profit,
                'sl': pos.sl,
                'tp': pos.tp,
                'time': int(pos.open_time),
            }
            for pos in self.positions
        ]
    
    def _execute_paper_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        symbol = order_data.get('symbol', self.current_symbol)
        action = order_data.get('action')
        volume = float(order_data.get('volume', 0.01))
        sl = order_data.get('sl')
        tp = order_data.get('tp')
        
        current_price = self._get_current_price(symbol)
        fill_price = self._apply_slippage(current_price, action)
        
        ticket = self._next_ticket
        self._next_ticket += 1
        
        position = PaperPosition(
            ticket=ticket, symbol=symbol, action=action, volume=volume,
            open_price=fill_price, open_time=time.time(), sl=sl, tp=tp,
        )
        
        self.positions.append(position)
        
        self.order_history.append({
            'ticket': ticket, 'symbol': symbol, 'action': action,
            'volume': volume, 'price': fill_price, 'sl': sl, 'tp': tp,
            'time': time.time(), 'status': 'filled',
        })
        
        logger.info(f"[PAPER] Order Executed: {action.upper()} {volume} {symbol} @ ${fill_price:.2f}")
        return {
            'status': 'success', 'ticket': ticket, 'symbol': symbol,
            'action': action, 'volume': volume, 'price': fill_price,
            'sl': sl, 'tp': tp, 'paper_mode': True,
        }
    
    def send_order(self, symbol: str, action: str, volume: float, price: Optional[float] = None, order_type: str = "market", **kwargs) -> Dict[str, Any]:
        return self._execute_paper_order({
            'symbol': symbol, 'action': action, 'volume': volume, 'order_type': order_type, **kwargs
        })
    
    def get_positions(self) -> List[Dict[str, Any]]:
        return self._format_positions()
    
    def get_status(self) -> Dict[str, Any]:
        return self._request("GET", "/status")
    
    def get_ticks(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        # Generate micro-ticks around current candle close
        current_price = self._get_current_price(symbol)
        ticks = []
        timestamp = time.time()
        for i in range(limit):
            tick_price = current_price + np.random.normal(0, current_price * 0.0001)
            ticks.append({
                'timestamp': timestamp - (limit - i),
                'bid': tick_price * 0.9999,
                'ask': tick_price * 1.0001,
                'last': tick_price,
                'volume': random.uniform(0.1, 1.0),
                'symbol': symbol,
            })
        return ticks
    
    def get_ohlcv(self, symbol: str, timeframe: str = "M1", limit: int = 1000) -> List[Dict[str, Any]]:
        """Return the stable history slice."""
        self._update_market() # Sync with time
        
        # If we asked for more than we have, we might return less (or should we backfill more?)
        # For now, return what we have.
        if limit > len(self.history):
            return self.history
        return self.history[-limit:]
    
    def close_position(self, ticket: int) -> Dict[str, Any]:
        position = next((p for p in self.positions if p.ticket == ticket), None)
        if not position:
            return {'status': 'error', 'error': f'Position {ticket} not found'}
        
        current_price = self._get_current_price(position.symbol)
        position.update_profit(current_price)
        
        self.balance += position.profit
        self.positions.remove(position)
        self._update_equity()
        
        logger.info(f"[PAPER] Position Closed: {position.action.upper()} {position.volume} {position.symbol} @ ${current_price:.2f}, P&L: ${position.profit:.2f}")
        return {'status': 'success', 'ticket': ticket, 'profit': position.profit, 'close_price': current_price, 'paper_mode': True}
    
    def close_all_positions(self) -> Dict[str, Any]:
        closed_tickets = []
        total_profit = 0.0
        for position in list(self.positions):
            result = self.close_position(position.ticket)
            if result['status'] == 'success':
                closed_tickets.append(position.ticket)
                total_profit += result['profit']
        
        logger.info(f"[PAPER] Closed {len(closed_tickets)} positions, Total P&L: ${total_profit:.2f}")
        return {'status': 'success', 'closed_count': len(closed_tickets), 'tickets': closed_tickets, 'total_profit': total_profit, 'paper_mode': True}

    def reset(self) -> None:
        self.positions.clear()
        self.order_history.clear()
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self._next_ticket = 1000
        logger.info(f"[PAPER] Account reset to initial balance: ${self.initial_balance:.2f}")
