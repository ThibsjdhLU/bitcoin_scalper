"""
Test for model warning behavior in demo mode.

This test ensures that when no model is loaded, the warning is only shown once
instead of being repeated every tick.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import logging

from src.bitcoin_scalper.core.engine import TradingEngine, TradingMode


class MockMT5Client:
    """Mock MT5 client for testing."""
    
    def __init__(self, base_url="http://test", api_key="test_key"):
        self.base_url = base_url
        self.api_key = api_key
        self.orders = []
    
    def _request(self, method, endpoint, **kwargs):
        """Mock request method."""
        if endpoint == "/account":
            return {
                'balance': 10000.0,
                'equity': 10000.0,
                'margin': 0.0,
                'free_margin': 10000.0
            }
        return {}
    
    def get_ohlcv(self, symbol, timeframe="M1", limit=100):
        """Return mock OHLCV data."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq='1min')
        data = []
        for ts in dates:
            data.append({
                'timestamp': int(ts.timestamp()),
                'open': 50000.0 + np.random.randn() * 100,
                'high': 50100.0 + np.random.randn() * 100,
                'low': 49900.0 + np.random.randn() * 100,
                'close': 50000.0 + np.random.randn() * 100,
                'volume': 100.0 + np.random.randn() * 10,
                'symbol': symbol
            })
        return data
    
    def send_order(self, symbol, action, volume, sl=None, tp=None, **kwargs):
        """Mock order sending."""
        order = {
            'symbol': symbol,
            'action': action,
            'volume': volume,
            'sl': sl,
            'tp': tp,
            'status': 'filled'
        }
        self.orders.append(order)
        return order


@pytest.fixture
def mock_mt5_client():
    """Provide a mock MT5 client."""
    return MockMT5Client()


class TestModelWarning:
    """Test suite for model warning behavior."""
    
    def test_ml_model_warning_shown_only_once(self, mock_mt5_client, tmp_path, caplog):
        """Test that ML model warning is only shown once, not every tick."""
        # Create engine without loading a model
        engine = TradingEngine(
            connector=mock_mt5_client,
            mode=TradingMode.ML,
            symbol="BTCUSD",
            timeframe="M1",
            log_dir=tmp_path,
            drift_detection=False,
        )
        
        # Ensure model is not loaded
        assert engine.ml_model is None
        assert engine.ml_warning_logged is False
        
        # Get mock market data
        market_data = mock_mt5_client.get_ohlcv("BTCUSD", limit=50)
        
        # Process multiple ticks
        with caplog.at_level(logging.WARNING):
            result1 = engine.process_tick(market_data)
            result2 = engine.process_tick(market_data)
            result3 = engine.process_tick(market_data)
        
        # Count how many times the warning was logged
        warning_count = sum(
            1 for record in caplog.records 
            if record.levelname == 'WARNING' and 'ML model not loaded' in record.message
        )
        
        # Should only be logged once
        assert warning_count == 1, f"Expected 1 warning, got {warning_count}"
        
        # Verify the ML flag is set
        assert engine.ml_warning_logged is True
        
        # All results should have no signal
        assert result1['signal'] is None
        assert result2['signal'] is None
        assert result3['signal'] is None
    
    def test_rl_agent_warning_shown_only_once(self, mock_mt5_client, tmp_path, caplog):
        """Test that RL agent warning is only shown once, not every tick."""
        # Create engine in RL mode without loading an agent
        engine = TradingEngine(
            connector=mock_mt5_client,
            mode=TradingMode.RL,
            symbol="BTCUSD",
            timeframe="M1",
            log_dir=tmp_path,
            drift_detection=False,
        )
        
        # Ensure agent is not loaded
        assert engine.rl_agent is None
        assert engine.rl_warning_logged is False
        
        # Get mock market data
        market_data = mock_mt5_client.get_ohlcv("BTCUSD", limit=50)
        
        # Process multiple ticks
        with caplog.at_level(logging.WARNING):
            result1 = engine.process_tick(market_data)
            result2 = engine.process_tick(market_data)
            result3 = engine.process_tick(market_data)
        
        # Count how many times the warning was logged
        warning_count = sum(
            1 for record in caplog.records 
            if record.levelname == 'WARNING' and 'RL agent not loaded' in record.message
        )
        
        # Should only be logged once
        assert warning_count == 1, f"Expected 1 warning, got {warning_count}"
        
        # Verify the RL flag is set
        assert engine.rl_warning_logged is True
    
    def test_model_loaded_no_warning(self, mock_mt5_client, tmp_path, caplog):
        """Test that no warning is shown when model is loaded."""
        # Create engine
        engine = TradingEngine(
            connector=mock_mt5_client,
            mode=TradingMode.ML,
            symbol="BTCUSD",
            timeframe="M1",
            log_dir=tmp_path,
            drift_detection=False,
        )
        
        # Mock a loaded model
        class MockModel:
            def predict(self, X):
                return [0]  # Return 'hold' signal
        
        engine.ml_model = MockModel()
        engine.features_list = ['close']
        
        # Get mock market data
        market_data = mock_mt5_client.get_ohlcv("BTCUSD", limit=50)
        
        # Process multiple ticks
        with caplog.at_level(logging.WARNING):
            result1 = engine.process_tick(market_data)
            result2 = engine.process_tick(market_data)
            result3 = engine.process_tick(market_data)
        
        # Count warnings about model not loaded
        warning_count = sum(
            1 for record in caplog.records 
            if record.levelname == 'WARNING' and 'model not loaded' in record.message.lower()
        )
        
        # Should be 0 warnings
        assert warning_count == 0, f"Expected no warnings, got {warning_count}"
        
        # Flags should still be False since warning was never needed
        assert engine.ml_warning_logged is False
        assert engine.rl_warning_logged is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
