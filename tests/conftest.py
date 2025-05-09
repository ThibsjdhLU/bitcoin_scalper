"""
Configuration des tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

@pytest.fixture
def sample_trades_df():
    """Fixture pour les données de trades de test."""
    return pd.DataFrame({
        'time': [datetime.now()],
        'type': ['BUY'],
        'price': [50000.0],
        'volume': [0.1],
        'profit': [100.0]
    })

@pytest.fixture
def sample_price_data():
    """Fixture pour les données de prix de test."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    return pd.DataFrame({
        'open': np.random.normal(50000, 1000, 100),
        'high': np.random.normal(51000, 1000, 100),
        'low': np.random.normal(49000, 1000, 100),
        'close': np.random.normal(50500, 1000, 100),
        'volume': np.random.normal(100, 10, 100)
    }, index=dates)

@pytest.fixture
def sample_strategy_params():
    """Fixture pour les paramètres de stratégie de test."""
    return {
        'ma_period': 20,
        'stop_loss': 0.02,
        'take_profit': 0.04
    }

@pytest.fixture
def test_directories():
    """Fixture pour les répertoires de test."""
    test_data_dir = Path("test_data")
    test_logs_dir = Path("test_logs")
    
    # Création des répertoires
    test_data_dir.mkdir(exist_ok=True)
    test_logs_dir.mkdir(exist_ok=True)
    
    yield test_data_dir, test_logs_dir
    
    # Nettoyage
    if test_data_dir.exists():
        for file in test_data_dir.glob("*"):
            file.unlink()
        test_data_dir.rmdir()
        
    if test_logs_dir.exists():
        for file in test_logs_dir.glob("*"):
            file.unlink()
        test_logs_dir.rmdir()

@pytest.fixture
def mock_mt5_connection():
    """Fixture pour la connexion MT5 mockée."""
    class MockMT5:
        def get_positions(self):
            return [{
                'time': datetime.now(),
                'type': 'BUY',
                'price': 50000.0,
                'volume': 0.1,
                'profit': 100.0
            }]
            
        def get_account_info(self):
            return {
                'balance': 10000.0,
                'equity': 10100.0,
                'profit': 100.0
            }
            
        def get_price_history(self, symbol="BTCUSD"):
            dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
            return pd.DataFrame({
                'open': np.random.normal(50000, 1000, 100),
                'high': np.random.normal(51000, 1000, 100),
                'low': np.random.normal(49000, 1000, 100),
                'close': np.random.normal(50500, 1000, 100),
                'volume': np.random.normal(100, 10, 100)
            }, index=dates)
            
        def get_available_symbols(self):
            return ["BTCUSD", "ETHUSD", "XRPUSD"]
            
    return MockMT5() 