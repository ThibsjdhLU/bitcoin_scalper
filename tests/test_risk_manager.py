import pytest
from datetime import datetime, timedelta
from risk_manager import RiskManager
from unittest.mock import Mock

@pytest.fixture
def mock_api():
    """Mock de l'API"""
    api = Mock()
    api.get_account_info.return_value = {
        'balance': 10000,
        'equity': 10000,
        'margin': 0
    }
    return api

@pytest.fixture
def risk_manager(mock_api):
    """Instance du gestionnaire de risque"""
    return RiskManager(mock_api)

def test_risk_manager_initialization(risk_manager):
    """Test de l'initialisation du gestionnaire de risque"""
    assert risk_manager.api is not None
    assert isinstance(risk_manager.config, dict)
    assert len(risk_manager.daily_trades) == 0
    assert risk_manager.daily_risk_used == 0.0
    assert risk_manager.peak_balance == 0.0
    assert risk_manager.current_drawdown == 0.0

def test_signal_validation(risk_manager, mock_api):
    """Test de la validation des signaux"""
    signal = {
        'type': 'buy',
        'price': 50000,
        'volume': 0.1,
        'sl': 49000,
        'tp': 51000
    }
    
    # Test avec un compte valide
    assert risk_manager.validate_signal(signal) is True
    
    # Test avec un drawdown trop important
    risk_manager.current_drawdown = 0.15
    assert risk_manager.validate_signal(signal) is False
    risk_manager.current_drawdown = 0.0
    
    # Test avec trop de positions ouvertes
    risk_manager.config['max_open_positions'] = 0
    assert risk_manager.validate_signal(signal) is False
    risk_manager.config['max_open_positions'] = 3
    
    # Test avec risque quotidien dépassé
    risk_manager.daily_risk_used = 0.06
    assert risk_manager.validate_signal(signal) is False
    risk_manager.daily_risk_used = 0.0

def test_peak_balance_update(risk_manager):
    """Test de la mise à jour du peak balance"""
    risk_manager._update_peak_balance(10000)
    assert risk_manager.peak_balance == 10000
    
    risk_manager._update_peak_balance(11000)
    assert risk_manager.peak_balance == 11000
    
    risk_manager._update_peak_balance(9000)
    assert risk_manager.peak_balance == 11000

def test_open_positions_count(risk_manager, mock_api):
    """Test du comptage des positions ouvertes"""
    mock_api.get_open_positions.return_value = [
        {'type': 'buy', 'volume': 0.1},
        {'type': 'sell', 'volume': 0.1}
    ]
    
    count = risk_manager._count_open_positions()
    assert count == 2

def test_daily_risk_update(risk_manager):
    """Test de la mise à jour du risque quotidien"""
    # Ajout d'un trade
    trade = {
        'timestamp': datetime.now(),
        'risk': 0.02
    }
    risk_manager.daily_trades.append(trade)
    
    # Mise à jour du risque
    risk_manager._update_daily_risk()
    assert risk_manager.daily_risk_used == 0.02
    
    # Ajout d'un vieux trade
    old_trade = {
        'timestamp': datetime.now() - timedelta(days=2),
        'risk': 0.03
    }
    risk_manager.daily_trades.append(old_trade)
    
    # Mise à jour du risque
    risk_manager._update_daily_risk()
    assert risk_manager.daily_risk_used == 0.02

def test_risk_reward_calculation(risk_manager):
    """Test du calcul du ratio risque/rendement"""
    signal = {
        'type': 'buy',
        'price': 50000,
        'sl': 49000,
        'tp': 51000
    }
    
    ratio = risk_manager._calculate_risk_reward(signal)
    assert isinstance(ratio, float)
    assert ratio > 0

def test_high_volatility_detection(risk_manager, mock_api):
    """Test de la détection de forte volatilité"""
    mock_api.get_market_data.return_value = {
        'atr': 1000
    }
    
    assert risk_manager._is_high_volatility() is True
    
    mock_api.get_market_data.return_value = {
        'atr': 100
    }
    
    assert risk_manager._is_high_volatility() is False

def test_position_size_calculation(risk_manager):
    """Test du calcul de la taille de position"""
    signal = {
        'type': 'buy',
        'price': 50000,
        'sl': 49000
    }
    balance = 10000
    
    size = risk_manager._calculate_position_size(signal, balance)
    assert isinstance(size, float)
    assert size > 0

def test_trade_risk_calculation(risk_manager):
    """Test du calcul du risque par trade"""
    signal = {
        'type': 'buy',
        'price': 50000,
        'sl': 49000,
        'volume': 0.1
    }
    balance = 10000
    
    risk = risk_manager._calculate_trade_risk(signal, balance)
    assert isinstance(risk, float)
    assert 0 <= risk <= 1

def test_trailing_stop_calculation(risk_manager):
    """Test du calcul du trailing stop"""
    position = {
        'type': 'buy',
        'entry_price': 50000,
        'current_price': 51000,
        'sl': 49000
    }
    
    new_sl = risk_manager.calculate_trailing_stop(position, 51000)
    assert isinstance(new_sl, float)
    assert new_sl > position['sl']

def test_drawdown_check(risk_manager):
    """Test de la vérification du drawdown"""
    risk_manager.current_drawdown = 0.05
    assert risk_manager.should_close_on_drawdown() is False
    
    risk_manager.current_drawdown = 0.15
    assert risk_manager.should_close_on_drawdown() is True 