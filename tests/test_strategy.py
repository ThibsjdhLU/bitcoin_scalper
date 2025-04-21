import pytest
import pandas as pd
import numpy as np
from strategy import ScalpingStrategy
from indicators import TechnicalIndicators

@pytest.fixture
def indicators():
    """Instance des indicateurs techniques"""
    return TechnicalIndicators()

@pytest.fixture
def strategy(indicators):
    """Instance de la stratégie"""
    return ScalpingStrategy(indicators)

@pytest.fixture
def market_data():
    """Données de marché de test"""
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
def sample_data():
    """Crée des données de test"""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(45000, 55000, 200),
        'high': np.random.uniform(45000, 55000, 200),
        'low': np.random.uniform(45000, 55000, 200),
        'close': np.random.uniform(45000, 55000, 200),
        'volume': np.random.uniform(1, 10, 200)
    }, index=dates)
    return data

@pytest.fixture
def strategy_config():
    """Configuration de test pour la stratégie"""
    return {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'volume_threshold': 1.5,
        'atr_multiplier': 1.5,
        'min_volatility': 0.001,
        'max_consecutive_losses': 3
    }

def test_strategy_initialization(strategy_config):
    """Test l'initialisation de la stratégie"""
    strategy = ScalpingStrategy(strategy_config)
    assert strategy.rsi_oversold == 30
    assert strategy.rsi_overbought == 70
    assert strategy.volume_threshold == 1.5
    assert strategy.consecutive_losses == 0

def test_generate_signal_insufficient_data(strategy_config):
    """Test la génération de signal avec données insuffisantes"""
    strategy = ScalpingStrategy(strategy_config)
    data = pd.DataFrame({'close': [1, 2, 3]})
    signal = strategy.generate_signal(data)
    assert signal is None

def test_generate_signal_valid_data(strategy_config, sample_data):
    """Test la génération de signal avec données valides"""
    strategy = ScalpingStrategy(strategy_config)
    signal = strategy.generate_signal(sample_data)
    assert signal is not None
    if signal:
        assert 'type' in signal
        assert 'price' in signal
        assert 'stop_loss' in signal
        assert 'take_profit' in signal

def test_handle_trade_result(strategy_config):
    """Test la gestion des résultats de trade"""
    strategy = ScalpingStrategy(strategy_config)
    initial_rsi_oversold = strategy.rsi_oversold
    
    # Simuler des pertes consécutives
    for _ in range(3):
        strategy.handle_trade_result({'profit': -100})
    
    assert strategy.consecutive_losses == 0  # Réinitialisé après max_consecutive_losses
    assert strategy.rsi_oversold > initial_rsi_oversold  # Ajusté après les pertes

def test_handle_trade_result_profit(strategy_config):
    """Test la gestion d'un trade profitable"""
    strategy = ScalpingStrategy(strategy_config)
    strategy.consecutive_losses = 2
    strategy.handle_trade_result({'profit': 100})
    assert strategy.consecutive_losses == 0

def test_evaluate_indicators(strategy_config, sample_data):
    """Test l'évaluation des indicateurs"""
    strategy = ScalpingStrategy(strategy_config)
    indicators = strategy.indicators.calculate(sample_data)
    
    ema_signal = strategy._evaluate_ema(indicators)
    rsi_signal = strategy._evaluate_rsi(indicators)
    macd_signal = strategy._evaluate_macd(indicators)
    stoch_signal = strategy._evaluate_stoch(indicators)
    
    assert isinstance(ema_signal, int)
    assert isinstance(rsi_signal, int)
    assert isinstance(macd_signal, int)
    assert isinstance(stoch_signal, int)
    assert all(signal in [-1, 0, 1] for signal in [ema_signal, rsi_signal, macd_signal, stoch_signal])

def test_generate_signals(strategy_config, sample_data):
    """Test la génération des signaux d'achat et de vente"""
    strategy = ScalpingStrategy(strategy_config)
    indicators = strategy.indicators.calculate(sample_data)
    current_price = sample_data['close'].iloc[-1]
    atr = indicators['atr'].iloc[-1]
    
    buy_signal = strategy._generate_buy_signal(current_price, atr)
    sell_signal = strategy._generate_sell_signal(current_price, atr)
    
    assert buy_signal['type'] == 'BUY'
    assert sell_signal['type'] == 'SELL'
    assert buy_signal['stop_loss'] < buy_signal['price']
    assert sell_signal['stop_loss'] > sell_signal['price']
    assert buy_signal['take_profit'] > buy_signal['price']
    assert sell_signal['take_profit'] < sell_signal['price']

def test_strategy_initialization(strategy):
    """Test de l'initialisation de la stratégie"""
    assert strategy.indicators is not None
    assert isinstance(strategy.params, dict)
    assert len(strategy.signal_history) == 0
    assert strategy.consecutive_losses == 0

def test_ema_crossover_evaluation(strategy, market_data):
    """Test de l'évaluation du croisement EMA"""
    latest = strategy.indicators.calculate(market_data)
    signal = strategy.evaluate_ema_crossover(latest)
    assert signal in [-1, 0, 1]

def test_rsi_evaluation(strategy, market_data):
    """Test de l'évaluation du RSI"""
    latest = strategy.indicators.calculate(market_data)
    signal = strategy.evaluate_rsi(latest)
    assert signal in [-1, 0, 1]

def test_macd_evaluation(strategy, market_data):
    """Test de l'évaluation du MACD"""
    latest = strategy.indicators.calculate(market_data)
    signal = strategy.evaluate_macd(latest)
    assert signal in [-1, 0, 1]

def test_stochastic_evaluation(strategy, market_data):
    """Test de l'évaluation du Stochastique"""
    latest = strategy.indicators.calculate(market_data)
    signal = strategy.evaluate_stochastic(latest)
    assert signal in [-1, 0, 1]

def test_signal_generation(strategy, market_data):
    """Test de la génération de signaux"""
    signal = strategy.generate_signal(market_data)
    assert isinstance(signal, dict)
    assert 'type' in signal
    assert 'price' in signal
    assert 'sl' in signal
    assert 'tp' in signal

def test_position_size_calculation(strategy):
    """Test du calcul de la taille de position"""
    signal_type = 'buy'
    price = 50000
    account_balance = 10000
    size = strategy.calculate_position_size(signal_type, price, account_balance)
    assert isinstance(size, float)
    assert size > 0

def test_stop_loss_calculation(strategy):
    """Test du calcul du stop loss"""
    signal_type = 'buy'
    price = 50000
    sl = strategy.calculate_stop_loss(signal_type, price)
    assert isinstance(sl, float)
    assert sl > 0

def test_take_profit_calculation(strategy):
    """Test du calcul du take profit"""
    signal_type = 'buy'
    price = 50000
    sl = 49000
    tp = strategy.calculate_take_profit(signal_type, price, sl)
    assert isinstance(tp, float)
    assert tp > price

def test_strategy_adjustment(strategy):
    """Test de l'ajustement de la stratégie"""
    initial_params = strategy.params.copy()
    strategy.consecutive_losses = 3
    strategy.adjust_strategy_on_loss()
    assert strategy.params != initial_params

def test_signal_logging(strategy):
    """Test de l'enregistrement des signaux"""
    signal = {
        'type': 'buy',
        'price': 50000,
        'sl': 49000,
        'tp': 51000
    }
    strategy.log_signal(signal)
    assert len(strategy.signal_history) == 1
    assert strategy.signal_history[0] == signal

def test_trade_result_handling(strategy):
    """Test du traitement des résultats de trade"""
    result = {
        'profit': -100,
        'type': 'buy'
    }
    initial_losses = strategy.consecutive_losses
    strategy.on_trade_result(result)
    if result['profit'] < 0:
        assert strategy.consecutive_losses == initial_losses + 1
    else:
        assert strategy.consecutive_losses == 0 