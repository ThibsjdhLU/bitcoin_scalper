"""
Unit tests for the Trading Engine.

Tests the core orchestration logic without requiring actual broker connection.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.bitcoin_scalper.core.engine import TradingEngine, TradingMode, MarketRegime
from src.bitcoin_scalper.core.config import TradingConfig


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


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TradingConfig(
        mode="ml",
        model_type="xgboost",
        symbol="BTCUSD",
        timeframe="M1",
        max_drawdown=0.10,
        max_daily_loss=0.10,
        risk_per_trade=0.02,
        position_sizer="kelly",
        kelly_fraction=0.25,
        drift_enabled=False,  # Disable drift detection for simple tests
    )


class TestTradingEngine:
    """Test suite for TradingEngine."""
    
    def test_engine_initialization(self, mock_mt5_client, test_config, tmp_path):
        """Test that engine initializes correctly."""
        engine = TradingEngine(
            mt5_client=mock_mt5_client,
            mode=TradingMode.ML,
            symbol=test_config.symbol,
            timeframe=test_config.timeframe,
            log_dir=tmp_path,
            risk_params={
                'max_drawdown': test_config.max_drawdown,
                'max_daily_loss': test_config.max_daily_loss,
            },
            drift_detection=False,
        )
        
        assert engine.symbol == test_config.symbol
        assert engine.timeframe == test_config.timeframe
        assert engine.mode == TradingMode.ML
        assert engine.tick_count == 0
        assert not engine.in_safe_mode
    
    def test_process_tick_without_model(self, mock_mt5_client, test_config, tmp_path):
        """Test process_tick when no model is loaded."""
        engine = TradingEngine(
            mt5_client=mock_mt5_client,
            mode=TradingMode.ML,
            symbol=test_config.symbol,
            log_dir=tmp_path,
            drift_detection=False,
        )
        
        # Get mock market data
        market_data = mock_mt5_client.get_ohlcv(test_config.symbol, limit=50)
        
        result = engine.process_tick(market_data)
        
        # Should return with no signal since no model is loaded
        assert result['signal'] is None or result['signal'] == 'hold'
        assert result['tick_number'] == 1
        assert engine.tick_count == 1
    
    def test_execute_order(self, mock_mt5_client, test_config, tmp_path):
        """Test order execution."""
        engine = TradingEngine(
            mt5_client=mock_mt5_client,
            mode=TradingMode.ML,
            symbol=test_config.symbol,
            log_dir=tmp_path,
            drift_detection=False,
        )
        
        result = engine.execute_order(
            signal='buy',
            volume=0.1,
            sl=49000.0,
            tp=51000.0
        )
        
        assert result['success'] is True
        assert len(mock_mt5_client.orders) == 1
        assert mock_mt5_client.orders[0]['action'] == 'buy'
        assert mock_mt5_client.orders[0]['volume'] == 0.1
    
    def test_execute_order_invalid_signal(self, mock_mt5_client, test_config, tmp_path):
        """Test that invalid signals are rejected."""
        engine = TradingEngine(
            mt5_client=mock_mt5_client,
            mode=TradingMode.ML,
            symbol=test_config.symbol,
            log_dir=tmp_path,
            drift_detection=False,
        )
        
        result = engine.execute_order(
            signal='invalid',
            volume=0.1
        )
        
        assert result['success'] is False
        assert 'invalid signal' in result['error'].lower()
    
    def test_get_status(self, mock_mt5_client, test_config, tmp_path):
        """Test status reporting."""
        engine = TradingEngine(
            mt5_client=mock_mt5_client,
            mode=TradingMode.ML,
            symbol=test_config.symbol,
            log_dir=tmp_path,
            drift_detection=True,
        )
        
        status = engine.get_status()
        
        assert status['mode'] == 'ml'
        assert status['symbol'] == test_config.symbol
        assert status['tick_count'] == 0
        assert status['in_safe_mode'] is False
        assert status['drift_detection_enabled'] is True
        assert status['model_loaded'] is False
    
    def test_safe_mode_reset(self, mock_mt5_client, test_config, tmp_path):
        """Test safe mode reset."""
        engine = TradingEngine(
            mt5_client=mock_mt5_client,
            mode=TradingMode.ML,
            symbol=test_config.symbol,
            log_dir=tmp_path,
            drift_detection=True,
            safe_mode_on_drift=True,
        )
        
        # Enter safe mode
        engine.in_safe_mode = True
        assert engine.in_safe_mode is True
        
        # Reset
        engine.reset_safe_mode()
        assert engine.in_safe_mode is False


class TestTradingConfig:
    """Test suite for TradingConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TradingConfig()
        
        assert config.mode == "ml"
        assert config.symbol == "BTCUSD"
        assert config.timeframe == "M1"
        assert config.max_drawdown == 0.05
        assert config.position_sizer == "kelly"
    
    def test_config_from_dict(self):
        """Test loading config from dictionary."""
        data = {
            'trading': {
                'mode': 'rl',
                'model_type': 'ppo',
                'symbol': 'ETHUSD',
            },
            'risk': {
                'max_drawdown': 0.10,
                'position_sizer': 'target_vol',
            }
        }
        
        config = TradingConfig.from_dict(data)
        
        assert config.mode == 'rl'
        assert config.model_type == 'ppo'
        assert config.symbol == 'ETHUSD'
        assert config.max_drawdown == 0.10
        assert config.position_sizer == 'target_vol'
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = TradingConfig(
            mode='ml',
            symbol='BTCUSD',
            max_drawdown=0.08
        )
        
        data = config.to_dict()
        
        assert data['trading']['mode'] == 'ml'
        assert data['trading']['symbol'] == 'BTCUSD'
        assert data['risk']['max_drawdown'] == 0.08
    
    def test_config_yaml_roundtrip(self, tmp_path):
        """Test saving and loading config as YAML."""
        config = TradingConfig(
            mode='rl',
            model_type='dqn',
            symbol='ETHUSD',
            max_drawdown=0.07
        )
        
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(str(yaml_path))
        
        loaded_config = TradingConfig.from_yaml(str(yaml_path))
        
        assert loaded_config.mode == 'rl'
        assert loaded_config.model_type == 'dqn'
        assert loaded_config.symbol == 'ETHUSD'
        assert loaded_config.max_drawdown == 0.07


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
