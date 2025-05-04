"""
Tests unitaires pour la stratégie EMA Crossover.
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.data_fetcher import TimeFrame
from core.order_executor import OrderSide
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.base_strategy import SignalType

@pytest.fixture
def mock_data_fetcher():
    """Crée un mock pour DataFetcher."""
    return MagicMock()

@pytest.fixture
def mock_order_executor():
    """Crée un mock pour OrderExecutor."""
    return MagicMock()

@pytest.fixture
def sample_data():
    """Crée des données de test avec un croisement d'EMA."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    
    # Créer une tendance haussière
    trend = np.linspace(0, 10, 100)
    noise = np.random.normal(0, 0.5, 100)
    prices = 100 + trend + noise
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices + abs(np.random.normal(0, 0.5, 100)),
        'low': prices - abs(np.random.normal(0, 0.5, 100)),
        'close': prices,
        'volume': np.random.randint(1000, 5000, 100)
    }, index=dates)
    
    return data

@pytest.fixture
def strategy(mock_data_fetcher, mock_order_executor):
    """Crée une instance de la stratégie."""
    return EMACrossoverStrategy(
        data_fetcher=mock_data_fetcher,
        order_executor=mock_order_executor,
        symbols=['BTCUSD'],
        timeframe=TimeFrame.M5,
        params={
            'fast_period': 5,
            'slow_period': 10,
            'min_crossover_strength': 0.0001
        }
    )

def test_init(strategy):
    """Teste l'initialisation de la stratégie."""
    assert strategy.name == "EMA Crossover"
    assert strategy.params['fast_period'] == 5
    assert strategy.params['slow_period'] == 10
    assert strategy.params['min_crossover_strength'] == 0.0001

def test_validate_params():
    """Teste la validation des paramètres."""
    # Test avec des paramètres valides
    strategy = EMACrossoverStrategy(
        data_fetcher=MagicMock(),
        order_executor=MagicMock(),
        symbols=['BTCUSD'],
        timeframe=TimeFrame.M5,
        params={
            'fast_period': 5,
            'slow_period': 10,
            'min_crossover_strength': 0.0001
        }
    )
    
    # Test avec des paramètres invalides
    with pytest.raises(ValueError):
        EMACrossoverStrategy(
            data_fetcher=MagicMock(),
            order_executor=MagicMock(),
            symbols=['BTCUSD'],
            timeframe=TimeFrame.M5,
            params={
                'fast_period': 10,
                'slow_period': 5,
                'min_crossover_strength': 0.0001
            }
        )
    
    with pytest.raises(ValueError):
        EMACrossoverStrategy(
            data_fetcher=MagicMock(),
            order_executor=MagicMock(),
            symbols=['BTCUSD'],
            timeframe=TimeFrame.M5,
            params={
                'fast_period': 5,
                'slow_period': 10,
                'min_crossover_strength': -0.0001
            }
        )

def test_get_required_data(strategy):
    """Teste le calcul du nombre de bougies requises."""
    assert strategy.get_required_data() == 20  # 2 * slow_period

def test_calculate_emas(strategy, sample_data):
    """Teste le calcul des EMA."""
    fast_ema, slow_ema = strategy._calculate_emas(sample_data)
    
    assert len(fast_ema) == len(sample_data)
    assert len(slow_ema) == len(sample_data)
    assert not fast_ema.isna().all()
    assert not slow_ema.isna().all()

def test_calculate_crossover_strength(strategy):
    """Teste le calcul de la force du croisement."""
    fast_ema = pd.Series([100, 101, 102])
    slow_ema = pd.Series([100, 100.5, 101])
    
    strength = strategy._calculate_crossover_strength(fast_ema, slow_ema)
    assert strength > 0
    assert strength == abs(102 - 101) / 101

def test_should_enter(strategy, sample_data):
    """Teste la génération des signaux d'entrée."""
    # Modifier les données pour créer un croisement
    sample_data.loc[sample_data.index[-2], 'close'] = 100
    sample_data.loc[sample_data.index[-1], 'close'] = 102
    
    should_enter, signal = strategy.should_enter('BTCUSD', sample_data)
    
    if should_enter:
        assert signal is not None
        assert signal.type == SignalType.BUY
        assert signal.symbol == 'BTCUSD'
        assert signal.strength > 0
        assert 'fast_ema' in signal.metadata
        assert 'slow_ema' in signal.metadata
        assert 'crossover_strength' in signal.metadata
    else:
        assert signal is None

def test_should_exit(strategy, sample_data):
    """Teste la génération des signaux de sortie."""
    # Modifier les données pour créer un croisement
    sample_data.loc[sample_data.index[-2], 'close'] = 102
    sample_data.loc[sample_data.index[-1], 'close'] = 100
    
    should_exit, signal = strategy.should_exit(
        'BTCUSD',
        sample_data,
        OrderSide.BUY
    )
    
    if should_exit:
        assert signal is not None
        assert signal.type == SignalType.SELL
        assert signal.symbol == 'BTCUSD'
        assert signal.strength > 0
        assert 'fast_ema' in signal.metadata
        assert 'slow_ema' in signal.metadata
        assert 'crossover_strength' in signal.metadata
    else:
        assert signal is None

def test_generate_signals(strategy, sample_data):
    """Teste la génération des signaux."""
    # Modifier les données pour créer un croisement
    sample_data.loc[sample_data.index[-2], 'close'] = 100
    sample_data.loc[sample_data.index[-1], 'close'] = 102
    
    signals = strategy.generate_signals('BTCUSD', sample_data)
    
    if signals:
        assert len(signals) > 0
        signal = signals[0]
        assert signal.type in [SignalType.BUY, SignalType.SELL]
        assert signal.symbol == 'BTCUSD'
        assert signal.strength > 0
        assert 'fast_ema' in signal.metadata
        assert 'slow_ema' in signal.metadata
        assert 'crossover_strength' in signal.metadata
    else:
        assert len(signals) == 0

def test_analyze(strategy, sample_data):
    """Teste l'analyse complète d'un symbole."""
    strategy.data_fetcher.get_historical_data.return_value = sample_data
    
    result = strategy.analyze('BTCUSD')
    
    assert result is not None
    should_enter, should_exit, signals = result
    assert isinstance(should_enter, bool)
    assert isinstance(should_exit, bool)
    assert isinstance(signals, list) 