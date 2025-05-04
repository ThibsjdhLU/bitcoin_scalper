"""
Tests unitaires pour le module d'indicateurs techniques.
"""
import numpy as np
import pandas as pd
import pytest

from utils.indicators import (
    calculate_ema,
    calculate_sma,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_stochastic,
    calculate_volume_profile,
    calculate_support_resistance
)

@pytest.fixture
def sample_data():
    """Crée des données de test."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, 100),
        'high': np.random.normal(101, 1, 100),
        'low': np.random.normal(99, 1, 100),
        'close': np.random.normal(100, 1, 100),
        'volume': np.random.randint(1000, 5000, 100)
    }, index=dates)
    
    # Assurer que high > low
    data['high'] = data[['open', 'close']].max(axis=1) + abs(np.random.normal(0, 0.5, 100))
    data['low'] = data[['open', 'close']].min(axis=1) - abs(np.random.normal(0, 0.5, 100))
    
    return data

def test_calculate_ema(sample_data):
    """Teste le calcul de l'EMA."""
    period = 20
    ema = calculate_ema(sample_data['close'], period)
    
    assert len(ema) == len(sample_data)
    assert not ema.isna().all()
    assert ema.iloc[-1] is not None

def test_calculate_sma(sample_data):
    """Teste le calcul du SMA."""
    period = 20
    sma = calculate_sma(sample_data['close'], period)
    
    assert len(sma) == len(sample_data)
    assert sma.iloc[period-1] is not None
    assert sma.iloc[0:period-1].isna().all()

def test_calculate_rsi(sample_data):
    """Teste le calcul du RSI."""
    period = 14
    rsi = calculate_rsi(sample_data['close'], period)
    
    assert len(rsi) == len(sample_data)
    assert not rsi.isna().all()
    assert (rsi >= 0).all() and (rsi <= 100).all()

def test_calculate_macd(sample_data):
    """Teste le calcul du MACD."""
    macd, signal, hist = calculate_macd(sample_data['close'])
    
    assert len(macd) == len(sample_data)
    assert len(signal) == len(sample_data)
    assert len(hist) == len(sample_data)
    assert not macd.isna().all()
    assert not signal.isna().all()
    assert not hist.isna().all()

def test_calculate_bollinger_bands(sample_data):
    """Teste le calcul des Bandes de Bollinger."""
    upper, middle, lower = calculate_bollinger_bands(sample_data['close'])
    
    assert len(upper) == len(sample_data)
    assert len(middle) == len(sample_data)
    assert len(lower) == len(sample_data)
    assert (upper >= middle).all()
    assert (middle >= lower).all()

def test_calculate_atr(sample_data):
    """Teste le calcul de l'ATR."""
    atr = calculate_atr(
        sample_data['high'],
        sample_data['low'],
        sample_data['close']
    )
    
    assert len(atr) == len(sample_data)
    assert not atr.isna().all()
    assert (atr >= 0).all()

def test_calculate_stochastic(sample_data):
    """Teste le calcul de l'oscillateur stochastique."""
    k, d = calculate_stochastic(
        sample_data['high'],
        sample_data['low'],
        sample_data['close']
    )
    
    assert len(k) == len(sample_data)
    assert len(d) == len(sample_data)
    assert (k >= 0).all() and (k <= 100).all()
    assert (d >= 0).all() and (d <= 100).all()

def test_calculate_volume_profile(sample_data):
    """Teste le calcul du profil de volume."""
    bins, volumes = calculate_volume_profile(
        sample_data['close'],
        sample_data['volume']
    )
    
    assert len(bins) == len(volumes)
    assert (volumes >= 0).all()
    assert len(bins) == 10  # Valeur par défaut

def test_calculate_support_resistance(sample_data):
    """Teste l'identification des supports et résistances."""
    supports, resistances = calculate_support_resistance(sample_data['close'])
    
    assert isinstance(supports, list)
    assert isinstance(resistances, list)
    assert all(isinstance(x, float) for x in supports)
    assert all(isinstance(x, float) for x in resistances)

def test_edge_cases():
    """Teste les cas limites."""
    # Données vides
    empty_data = pd.Series([])
    assert len(calculate_ema(empty_data, 20)) == 0
    assert len(calculate_sma(empty_data, 20)) == 0
    
    # Données constantes
    constant_data = pd.Series([100] * 100)
    ema = calculate_ema(constant_data, 20)
    assert (ema == 100).all()
    
    # Période plus grande que les données
    short_data = pd.Series([1, 2, 3])
    sma = calculate_sma(short_data, 5)
    assert sma.isna().all() 