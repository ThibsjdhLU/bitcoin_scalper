import pytest
import pandas as pd
import numpy as np
from indicators import TechnicalIndicators

@pytest.fixture
def sample_data():
    """Données de test pour les indicateurs"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(45000, 55000, 100),
        'high': np.random.uniform(46000, 56000, 100),
        'low': np.random.uniform(44000, 54000, 100),
        'close': np.random.uniform(45000, 55000, 100),
        'volume': np.random.uniform(1, 10, 100)
    }, index=dates)
    return data

@pytest.fixture
def indicators():
    """Instance des indicateurs techniques"""
    return TechnicalIndicators()

def test_ema_calculation(indicators, sample_data):
    """Test du calcul des EMAs"""
    result = indicators.calculate(sample_data)
    assert 'ema_short' in result
    assert 'ema_long' in result
    assert len(result['ema_short']) == len(sample_data)
    assert len(result['ema_long']) == len(sample_data)

def test_rsi_calculation(indicators, sample_data):
    """Test du calcul du RSI"""
    result = indicators.calculate(sample_data)
    assert 'rsi' in result
    assert len(result['rsi']) == len(sample_data)
    assert all(0 <= rsi <= 100 for rsi in result['rsi'] if not np.isnan(rsi))

def test_macd_calculation(indicators, sample_data):
    """Test du calcul du MACD"""
    result = indicators.calculate(sample_data)
    assert 'macd' in result
    assert 'macd_signal' in result
    assert 'macd_hist' in result
    assert len(result['macd']) == len(sample_data)

def test_stochastic_calculation(indicators, sample_data):
    """Test du calcul du Stochastique"""
    result = indicators.calculate(sample_data)
    assert 'stoch_k' in result
    assert 'stoch_d' in result
    assert len(result['stoch_k']) == len(sample_data)
    assert len(result['stoch_d']) == len(sample_data)
    assert all(0 <= k <= 100 for k in result['stoch_k'] if not np.isnan(k))
    assert all(0 <= d <= 100 for d in result['stoch_d'] if not np.isnan(d))

def test_atr_calculation(indicators, sample_data):
    """Test du calcul de l'ATR"""
    result = indicators.calculate(sample_data)
    assert 'atr' in result
    assert len(result['atr']) == len(sample_data)
    assert all(atr >= 0 for atr in result['atr'] if not np.isnan(atr))

def test_crossing_detection(indicators, sample_data):
    """Test de la détection des croisements"""
    result = indicators.calculate(sample_data)
    assert isinstance(indicators.is_crossing_up('ema_short', 'ema_long'), bool)
    assert isinstance(indicators.is_crossing_down('ema_short', 'ema_long'), bool)

def test_invalid_data(indicators):
    """Test avec des données invalides"""
    with pytest.raises(ValueError):
        indicators.calculate(None)
    with pytest.raises(ValueError):
        indicators.calculate(pd.DataFrame())

def test_update_params(indicators):
    """Test de la mise à jour des paramètres"""
    new_params = {
        'ema_short': 5,
        'ema_long': 15,
        'rsi_period': 10
    }
    indicators.update_params(**new_params)
    assert indicators.ema_short == 5
    assert indicators.ema_long == 15
    assert indicators.rsi_period == 10 