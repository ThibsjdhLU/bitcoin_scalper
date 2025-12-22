"""
Tests for Binance Connector.

This module contains tests for the BinanceConnector class to ensure:
- fetch_ohlcv returns proper DataFrame with lowercase columns
- Column names match expected format
- Data types are correct
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from bitcoin_scalper.connectors.binance_connector import BinanceConnector


class TestBinanceConnector:
    """Test suite for BinanceConnector."""
    
    @patch('bitcoin_scalper.connectors.binance_connector.ccxt.binance')
    def test_initialization(self, mock_binance):
        """Test that BinanceConnector initializes correctly."""
        # Setup mock
        mock_exchange = Mock()
        mock_binance.return_value = mock_exchange
        
        # Initialize connector
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        # Verify
        assert connector.api_key == "test_key"
        assert connector.api_secret == "test_secret"
        assert connector.testnet is True
        mock_exchange.load_markets.assert_called_once()
    
    @patch('bitcoin_scalper.connectors.binance_connector.ccxt.binance')
    def test_fetch_ohlcv_returns_dataframe_with_correct_columns(self, mock_binance):
        """Test that fetch_ohlcv returns DataFrame with lowercase columns."""
        # Setup mock
        mock_exchange = Mock()
        mock_binance.return_value = mock_exchange
        
        # Mock OHLCV data from CCXT (timestamp, open, high, low, close, volume)
        mock_ohlcv_data = [
            [1609459200000, 29000.0, 29500.0, 28500.0, 29300.0, 100.0],
            [1609459260000, 29300.0, 29600.0, 29100.0, 29400.0, 150.0],
            [1609459320000, 29400.0, 29700.0, 29200.0, 29500.0, 120.0],
        ]
        mock_exchange.fetch_ohlcv.return_value = mock_ohlcv_data
        
        # Initialize connector
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        # Fetch OHLCV
        df = connector.fetch_ohlcv("BTC/USDT", "1m", 100)
        
        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == 'date'
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        
        # Verify data types
        assert df['open'].dtype == float
        assert df['high'].dtype == float
        assert df['low'].dtype == float
        assert df['close'].dtype == float
        assert df['volume'].dtype == float
        
        # Verify data
        assert len(df) == 3
        assert df['close'].iloc[0] == 29300.0
        assert df['close'].iloc[-1] == 29500.0
    
    @patch('bitcoin_scalper.connectors.binance_connector.ccxt.binance')
    def test_fetch_ohlcv_with_datetime_index(self, mock_binance):
        """Test that fetch_ohlcv returns DataFrame with datetime index."""
        # Setup mock
        mock_exchange = Mock()
        mock_binance.return_value = mock_exchange
        
        # Mock OHLCV data
        mock_ohlcv_data = [
            [1609459200000, 29000.0, 29500.0, 28500.0, 29300.0, 100.0],
        ]
        mock_exchange.fetch_ohlcv.return_value = mock_ohlcv_data
        
        # Initialize connector
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        # Fetch OHLCV
        df = connector.fetch_ohlcv("BTC/USDT", "1m", 100)
        
        # Verify index is datetime
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == 'date'
    
    @patch('bitcoin_scalper.connectors.binance_connector.ccxt.binance')
    def test_execute_order_buy(self, mock_binance):
        """Test executing a buy order."""
        # Setup mock
        mock_exchange = Mock()
        mock_binance.return_value = mock_exchange
        
        # Mock order response
        mock_order = {
            'id': '12345',
            'symbol': 'BTC/USDT',
            'type': 'market',
            'side': 'buy',
            'price': 29500.0,
            'amount': 0.001,
            'filled': 0.001,
            'remaining': 0.0,
            'status': 'closed',
            'timestamp': 1609459200000
        }
        mock_exchange.create_market_order.return_value = mock_order
        
        # Initialize connector
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        # Execute order
        result = connector.execute_order("BTC/USDT", "buy", 0.001)
        
        # Verify
        assert result['id'] == '12345'
        assert result['side'] == 'buy'
        assert result['status'] == 'closed'
        mock_exchange.create_market_order.assert_called_once_with(
            symbol="BTC/USDT",
            side="buy",
            amount=0.001,
            params={}
        )
    
    @patch('bitcoin_scalper.connectors.binance_connector.ccxt.binance')
    def test_get_balance(self, mock_binance):
        """Test getting account balance."""
        # Setup mock
        mock_exchange = Mock()
        mock_binance.return_value = mock_exchange
        
        # Mock balance response
        mock_balance = {
            'free': {
                'USDT': 10000.0,
                'BTC': 0.5
            },
            'used': {
                'USDT': 0.0,
                'BTC': 0.0
            },
            'total': {
                'USDT': 10000.0,
                'BTC': 0.5
            }
        }
        mock_exchange.fetch_balance.return_value = mock_balance
        
        # Initialize connector
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        # Get balance
        balance = connector.get_balance("USDT")
        
        # Verify
        assert balance == 10000.0
        mock_exchange.fetch_balance.assert_called_once()
    
    @patch('bitcoin_scalper.connectors.binance_connector.ccxt.binance')
    def test_fetch_ohlcv_empty_data(self, mock_binance):
        """Test fetch_ohlcv returns empty DataFrame when no data."""
        # Setup mock
        mock_exchange = Mock()
        mock_binance.return_value = mock_exchange
        
        # Mock empty OHLCV data
        mock_exchange.fetch_ohlcv.return_value = []
        
        # Initialize connector
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        # Fetch OHLCV
        df = connector.fetch_ohlcv("BTC/USDT", "1m", 100)
        
        # Verify
        assert isinstance(df, pd.DataFrame)
        assert df.empty
