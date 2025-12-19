"""
Unit tests for Paper Trading Client.

Tests the PaperMT5Client functionality including:
- Order execution
- Position tracking
- Account balance management
- Simulated market data generation
"""

import pytest
import time
from src.bitcoin_scalper.connectors.paper import PaperMT5Client, PaperPosition


class TestPaperPosition:
    """Test suite for PaperPosition dataclass."""
    
    def test_position_creation(self):
        """Test creating a paper position."""
        pos = PaperPosition(
            ticket=1000,
            symbol="BTCUSD",
            action="buy",
            volume=0.1,
            open_price=50000.0,
            open_time=time.time(),
        )
        
        assert pos.ticket == 1000
        assert pos.symbol == "BTCUSD"
        assert pos.action == "buy"
        assert pos.volume == 0.1
        assert pos.open_price == 50000.0
        assert pos.profit == 0.0
    
    def test_position_profit_buy(self):
        """Test profit calculation for buy position."""
        pos = PaperPosition(
            ticket=1000,
            symbol="BTCUSD",
            action="buy",
            volume=0.1,
            open_price=50000.0,
            open_time=time.time(),
        )
        
        # Price goes up
        profit = pos.update_profit(51000.0)
        assert profit == 100.0  # (51000 - 50000) * 0.1
        assert pos.profit == 100.0
        
        # Price goes down
        profit = pos.update_profit(49000.0)
        assert profit == -100.0  # (49000 - 50000) * 0.1
        assert pos.profit == -100.0
    
    def test_position_profit_sell(self):
        """Test profit calculation for sell position."""
        pos = PaperPosition(
            ticket=1001,
            symbol="BTCUSD",
            action="sell",
            volume=0.1,
            open_price=50000.0,
            open_time=time.time(),
        )
        
        # Price goes down (profit for sell)
        profit = pos.update_profit(49000.0)
        assert profit == 100.0  # (50000 - 49000) * 0.1
        assert pos.profit == 100.0
        
        # Price goes up (loss for sell)
        profit = pos.update_profit(51000.0)
        assert profit == -100.0  # (50000 - 51000) * 0.1
        assert pos.profit == -100.0


class TestPaperMT5Client:
    """Test suite for PaperMT5Client."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = PaperMT5Client(initial_balance=5000.0)
        
        assert client.initial_balance == 5000.0
        assert client.balance == 5000.0
        assert client.equity == 5000.0
        assert len(client.positions) == 0
        assert len(client.order_history) == 0
    
    def test_set_price(self):
        """Test setting market price."""
        client = PaperMT5Client()
        
        client.set_price("BTCUSD", 50000.0)
        
        assert client.simulated_prices["BTCUSD"] == 50000.0
    
    def test_send_order_buy(self):
        """Test executing a buy order."""
        client = PaperMT5Client(initial_balance=10000.0)
        client.set_price("BTCUSD", 50000.0)
        
        result = client.send_order(
            symbol="BTCUSD",
            action="buy",
            volume=0.1
        )
        
        assert result['status'] == 'success'
        assert result['symbol'] == 'BTCUSD'
        assert result['action'] == 'buy'
        assert result['volume'] == 0.1
        assert result['paper_mode'] is True
        assert 'ticket' in result
        
        # Check position was created
        assert len(client.positions) == 1
        assert client.positions[0].action == 'buy'
        assert client.positions[0].volume == 0.1
    
    def test_send_order_sell(self):
        """Test executing a sell order."""
        client = PaperMT5Client(initial_balance=10000.0)
        client.set_price("BTCUSD", 50000.0)
        
        result = client.send_order(
            symbol="BTCUSD",
            action="sell",
            volume=0.2
        )
        
        assert result['status'] == 'success'
        assert result['action'] == 'sell'
        assert result['volume'] == 0.2
        
        # Check position was created
        assert len(client.positions) == 1
        assert client.positions[0].action == 'sell'
    
    def test_get_positions(self):
        """Test getting positions."""
        client = PaperMT5Client()
        client.set_price("BTCUSD", 50000.0)
        
        # No positions initially
        positions = client.get_positions()
        assert len(positions) == 0
        
        # Execute an order
        client.send_order("BTCUSD", action="buy", volume=0.1)
        
        # Check positions
        positions = client.get_positions()
        assert len(positions) == 1
        assert positions[0]['symbol'] == 'BTCUSD'
        assert positions[0]['volume'] == 0.1
        assert positions[0]['type'] == 0  # 0 = buy
    
    def test_account_info(self):
        """Test getting account information."""
        client = PaperMT5Client(initial_balance=10000.0)
        
        account = client._request("GET", "/account")
        
        assert account['balance'] == 10000.0
        assert account['equity'] == 10000.0
        assert account['paper_mode'] is True
    
    def test_close_position(self):
        """Test closing a position."""
        client = PaperMT5Client(initial_balance=10000.0)
        client.set_price("BTCUSD", 50000.0)
        
        # Open position
        order_result = client.send_order("BTCUSD", action="buy", volume=0.1)
        ticket = order_result['ticket']
        
        # Price goes up
        client.set_price("BTCUSD", 51000.0)
        
        # Close position
        close_result = client.close_position(ticket)
        
        assert close_result['status'] == 'success'
        assert close_result['ticket'] == ticket
        assert close_result['profit'] == 100.0  # (51000 - 50000) * 0.1
        
        # Check balance updated
        assert client.balance == 10100.0  # Initial + profit
        
        # Position should be removed
        assert len(client.positions) == 0
    
    def test_close_all_positions(self):
        """Test closing all positions."""
        client = PaperMT5Client(initial_balance=10000.0)
        client.set_price("BTCUSD", 50000.0)
        
        # Open multiple positions
        client.send_order("BTCUSD", action="buy", volume=0.1)
        client.send_order("BTCUSD", action="buy", volume=0.2)
        
        assert len(client.positions) == 2
        
        # Price goes up
        client.set_price("BTCUSD", 51000.0)
        
        # Close all
        result = client.close_all_positions()
        
        assert result['status'] == 'success'
        assert result['closed_count'] == 2
        assert result['total_profit'] == 300.0  # (0.1 + 0.2) * 1000
        
        # All positions should be closed
        assert len(client.positions) == 0
        
        # Balance should be updated
        assert client.balance == 10300.0
    
    def test_position_profit_tracking(self):
        """Test that position profits are tracked correctly."""
        client = PaperMT5Client(initial_balance=10000.0)
        client.set_price("BTCUSD", 50000.0)
        
        # Open buy position
        client.send_order("BTCUSD", action="buy", volume=0.1)
        
        # Price increases
        client.set_price("BTCUSD", 52000.0)
        
        # Check equity reflects unrealized profit
        assert client.equity == 10200.0  # 10000 + (52000 - 50000) * 0.1
        assert client.balance == 10000.0  # Balance unchanged until closed
    
    def test_get_ohlcv(self):
        """Test getting simulated OHLCV data."""
        client = PaperMT5Client()
        client.set_price("BTCUSD", 50000.0)
        
        ohlcv = client.get_ohlcv("BTCUSD", timeframe="M1", limit=50)
        
        assert len(ohlcv) == 50
        assert all('timestamp' in candle for candle in ohlcv)
        assert all('open' in candle for candle in ohlcv)
        assert all('high' in candle for candle in ohlcv)
        assert all('low' in candle for candle in ohlcv)
        assert all('close' in candle for candle in ohlcv)
        assert all('volume' in candle for candle in ohlcv)
    
    def test_get_ticks(self):
        """Test getting simulated tick data."""
        client = PaperMT5Client()
        client.set_price("BTCUSD", 50000.0)
        
        ticks = client.get_ticks("BTCUSD", limit=100)
        
        assert len(ticks) == 100
        assert all('timestamp' in tick for tick in ticks)
        assert all('bid' in tick for tick in ticks)
        assert all('ask' in tick for tick in ticks)
        assert all('last' in tick for tick in ticks)
    
    def test_reset(self):
        """Test resetting the paper account."""
        client = PaperMT5Client(initial_balance=10000.0)
        client.set_price("BTCUSD", 50000.0)
        
        # Execute orders
        client.send_order("BTCUSD", action="buy", volume=0.1)
        client.send_order("BTCUSD", action="buy", volume=0.2)
        
        # Change balance
        client.set_price("BTCUSD", 52000.0)
        
        assert len(client.positions) > 0
        assert len(client.order_history) > 0
        
        # Reset
        client.reset()
        
        # Everything should be cleared
        assert len(client.positions) == 0
        assert len(client.order_history) == 0
        assert client.balance == 10000.0
        assert client.equity == 10000.0
    
    def test_slippage_simulation(self):
        """Test that slippage is applied when enabled."""
        client = PaperMT5Client(
            initial_balance=10000.0,
            enable_slippage=True,
            simulated_slippage=0.001  # 0.1% slippage
        )
        client.set_price("BTCUSD", 50000.0)
        
        # Buy order - should get worse price (higher)
        result = client.send_order("BTCUSD", action="buy", volume=0.1)
        buy_price = result['price']
        
        assert buy_price > 50000.0  # Should pay more due to slippage
        assert buy_price <= 50000.0 * 1.001  # Within slippage range
    
    def test_multiple_symbols(self):
        """Test handling multiple trading symbols."""
        client = PaperMT5Client(initial_balance=10000.0)
        client.set_price("BTCUSD", 50000.0)
        client.set_price("ETHUSD", 3000.0)
        
        # Execute orders for different symbols
        client.send_order("BTCUSD", action="buy", volume=0.1)
        client.send_order("ETHUSD", action="buy", volume=1.0)
        
        positions = client.get_positions()
        assert len(positions) == 2
        
        symbols = {pos['symbol'] for pos in positions}
        assert 'BTCUSD' in symbols
        assert 'ETHUSD' in symbols


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
