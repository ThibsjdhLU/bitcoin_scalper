"""
Configuration globale pour pytest
"""

import os
import sys
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

# Configuration de l'environnement de test
os.environ['DEMO_MODE'] = '1'
os.environ['MT5_LOGIN'] = 'test_login'
os.environ['MT5_PASSWORD'] = 'test_password'
os.environ['MT5_SERVER'] = 'test_server'

# Configuration de Streamlit pour les tests
import streamlit as st
st.set_page_config(
    page_title="Bitcoin Trading Bot",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation des variables de session Streamlit
if 'bot_status' not in st.session_state:
    st.session_state.bot_status = "Inactif"
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 10
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = 'BTCUSDT'
if 'available_symbols' not in st.session_state:
    st.session_state.available_symbols = ['BTCUSDT', 'ETHUSDT']
if 'trading_params' not in st.session_state:
    st.session_state.trading_params = {
        'initial_capital': 1000.0,
        'risk_per_trade': 1.0,
        'strategy': ['EMA Crossover'],
        'take_profit': 2.0,
        'stop_loss': 1.0,
        'trailing_stop': True
    }
if 'account_stats' not in st.session_state:
    st.session_state.account_stats = {
        'balance': 1000.0,
        'profit': 0.0,
        'max_drawdown': 0.0
    }
if 'indicators' not in st.session_state:
    st.session_state.indicators = {
        'show_sma': True,
        'sma_period': 20,
        'show_ema': True,
        'ema_period': 20,
        'show_bollinger': True,
        'bollinger_period': 20,
        'show_rsi': True,
        'rsi_period': 14,
        'show_macd': True,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    }
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_data_fetcher():
    """Fixture pour simuler un DataFetcher"""
    mock = MagicMock()
    mock.get_historical_data.return_value = {
        "timestamp": [1, 2, 3],
        "open": [100, 101, 102],
        "high": [105, 106, 107],
        "low": [95, 96, 97],
        "close": [102, 103, 104],
        "volume": [1000, 1100, 1200]
    }
    return mock

@pytest.fixture
def mock_order_executor():
    """Fixture pour simuler un OrderExecutor"""
    mock = MagicMock()
    mock.execute_order.return_value = {"order_id": "123", "status": "filled"}
    return mock

@pytest.fixture
def mock_risk_manager():
    """Fixture pour simuler un RiskManager"""
    mock = MagicMock()
    mock.calculate_position_size.return_value = 0.1
    mock.check_risk_limits.return_value = True
    return mock

@pytest.fixture
def sample_market_data():
    """Fixture pour fournir des données de marché de test"""
    return {
        "BTCUSD": {
            "timestamp": [1, 2, 3, 4, 5],
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "close": [102, 103, 104, 105, 106],
            "volume": [1000, 1100, 1200, 1300, 1400]
        }
    }
