"""
Tests unitaires pour le module DataFetcher.
"""
import json
from datetime import datetime, timedelta
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from core.mt5_connector import MT5Connector
from core.data_fetcher import DataFetcher, TimeFrame

@pytest.fixture
def mock_mt5():
    """Mock de MetaTrader5 pour les tests."""
    with patch('core.data_fetcher.mt5') as mock:
        # Constantes MT5
        mock.TIMEFRAME_M1 = 1
        mock.TIMEFRAME_M5 = 5
        mock.TIMEFRAME_M15 = 15
        mock.TIMEFRAME_M30 = 30
        mock.TIMEFRAME_H1 = 16385
        mock.TIMEFRAME_H4 = 16388
        mock.TIMEFRAME_D1 = 16408
        
        # Mock des données historiques
        candle_data = {
            'time': int(datetime.now().timestamp()),
            'open': 50000.0,
            'high': 51000.0,
            'low': 49000.0,
            'close': 50500.0,
            'tick_volume': 100
        }
        mock.copy_rates_range.return_value = [candle_data]
        mock.copy_rates_from_pos.return_value = [candle_data]
        
        # Mock des informations de symbole
        tick_info = MagicMock()
        tick_info.bid = 50000.0
        tick_info.ask = 50001.0
        tick_info.last = 50000.5
        tick_info.volume = 1000
        mock.symbol_info_tick.return_value = tick_info
        
        symbol_info = MagicMock()
        symbol_info.name = "BTCUSD"
        symbol_info.bid = 50000.0
        symbol_info.ask = 50001.0
        symbol_info.point = 0.1
        symbol_info.digits = 2
        symbol_info.spread = 1
        symbol_info.volume_min = 0.01
        symbol_info.volume_max = 1.0
        symbol_info.volume_step = 0.01
        symbol_info.swap_long = 0.0
        symbol_info.swap_short = 0.0
        symbol_info.margin_initial = 1000.0
        symbol_info.margin_maintenance = 500.0
        mock.symbol_info.return_value = symbol_info
        
        yield mock

@pytest.fixture
def mock_connector():
    """Mock du connecteur MT5."""
    connector = MagicMock(spec=MT5Connector)
    connector.connected = True
    return connector

@pytest.fixture
def data_fetcher(mock_connector, tmp_path):
    """Instance de DataFetcher pour les tests."""
    config = {
        "broker": {
            "mt5": {
                "server": "test-server",
                "login": "12345",
                "password": "test-password",
                "symbols": ["BTCUSD", "ETHUSD"]
            }
        }
    }
    config_path = tmp_path / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    return DataFetcher(mock_connector, str(config_path))

def test_init(data_fetcher, mock_connector):
    """Teste l'initialisation du récupérateur de données."""
    assert data_fetcher.connector == mock_connector
    assert hasattr(data_fetcher, 'config')

def test_get_historical_data(data_fetcher, mock_mt5):
    """Teste la récupération des données historiques."""
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()
    
    df = data_fetcher.get_historical_data(
        symbol="BTCUSD",
        timeframe=TimeFrame.M15,
        start_date=start_date,
        end_date=end_date
    )
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    
    mock_mt5.copy_rates_range.assert_called_once_with(
        "BTCUSD",
        TimeFrame.M15.value,
        start_date,
        end_date
    )

def test_get_latest_candle(data_fetcher, mock_mt5):
    """Teste la récupération de la dernière bougie."""
    candle = data_fetcher.get_latest_candle(
        symbol="BTCUSD",
        timeframe=TimeFrame.M15
    )
    
    assert isinstance(candle, dict)
    assert all(key in candle for key in ['time', 'open', 'high', 'low', 'close', 'volume'])
    assert candle['open'] == 50000.0
    assert candle['close'] == 50500.0
    
    mock_mt5.copy_rates_from_pos.assert_called_once_with(
        "BTCUSD",
        TimeFrame.M15.value,
        0,
        1
    )

def test_get_current_price(data_fetcher, mock_mt5):
    """Teste la récupération du prix actuel."""
    price = data_fetcher.get_current_price("BTCUSD")
    
    assert isinstance(price, dict)
    assert all(key in price for key in ['bid', 'ask', 'last', 'volume'])
    assert price['bid'] == 50000.0
    assert price['ask'] == 50001.0
    
    mock_mt5.symbol_info_tick.assert_called_once_with("BTCUSD")

def test_get_symbol_info(data_fetcher, mock_mt5):
    """Teste la récupération des informations du symbole."""
    info = data_fetcher.get_symbol_info("BTCUSD")
    
    assert isinstance(info, dict)
    assert all(key in info for key in [
        'name', 'bid', 'ask', 'point', 'digits', 'spread',
        'volume_min', 'volume_max', 'volume_step',
        'swap_long', 'swap_short',
        'margin_initial', 'margin_maintenance'
    ])
    assert info['name'] == "BTCUSD"
    assert info['volume_min'] == 0.01
    
    mock_mt5.symbol_info.assert_called_once_with("BTCUSD")

def test_error_handling(data_fetcher, mock_mt5):
    """Teste la gestion des erreurs."""
    # Simuler une erreur de connexion
    data_fetcher.connector.connected = False
    
    assert data_fetcher.get_historical_data("BTCUSD", TimeFrame.M15, datetime.now()) is None
    assert data_fetcher.get_latest_candle("BTCUSD", TimeFrame.M15) is None
    assert data_fetcher.get_current_price("BTCUSD") is None
    assert data_fetcher.get_symbol_info("BTCUSD") is None
    
    # Simuler une erreur de données
    data_fetcher.connector.connected = True
    mock_mt5.copy_rates_range.return_value = None
    
    assert data_fetcher.get_historical_data("BTCUSD", TimeFrame.M15, datetime.now()) is None 