import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtest import Backtest, Trade
from strategy import Signal, SignalType

def generate_sample_data(n_periods=100):
    """Génère des données de test"""
    dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='1min')
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, n_periods),
        'high': np.random.normal(101, 1, n_periods),
        'low': np.random.normal(99, 1, n_periods),
        'close': np.random.normal(100, 1, n_periods),
        'volume': np.random.normal(1000, 100, n_periods)
    }, index=dates)
    return data

@pytest.fixture
def config():
    """Configuration de test"""
    return {
        'initial_balance': 10000,
        'commission_rate': 0.001,
        'slippage': 0.0001,
        'strategy': {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'min_volume': 100,
            'min_volatility': 0.001
        }
    }

@pytest.fixture
def backtest(config):
    """Instance de Backtest pour les tests"""
    return Backtest(config)

def test_backtest_initialization(backtest, config):
    """Test l'initialisation du backtest"""
    assert backtest.initial_balance == config['initial_balance']
    assert backtest.commission_rate == config['commission_rate']
    assert backtest.slippage == config['slippage']
    assert backtest.balance == config['initial_balance']
    assert len(backtest.positions) == 0
    assert len(backtest.trades_history) == 0
    assert len(backtest.equity_curve) == 1
    assert backtest.equity_curve[0] == config['initial_balance']

def test_backtest_run(backtest):
    """Test l'exécution du backtest"""
    # Génération des données de test
    data = {
        'BTC/USDT': generate_sample_data(100)
    }
    
    # Exécution du backtest
    results = backtest.run(data)
    
    # Vérification des résultats
    assert 'trades' in results
    assert 'equity_curve' in results
    assert 'metrics' in results
    assert isinstance(results['trades'], list)
    assert isinstance(results['equity_curve'], list)
    assert isinstance(results['metrics'], dict)

def test_position_management(backtest):
    """Test la gestion des positions"""
    # Création d'une position de test
    position = Trade(
        symbol='BTC/USDT',
        entry_time=datetime.now(),
        exit_time=None,
        entry_price=100.0,
        exit_price=None,
        side='long',
        size=1.0,
        stop_loss=95.0,
        take_profit=105.0,
        pnl=None,
        fees=0.1,
        metadata={'signal_strength': 0.8}
    )
    
    # Ajout de la position
    position_id = f"{position.symbol}_{position.entry_time.timestamp()}"
    backtest.positions[position_id] = position
    
    # Vérification de la position
    assert len(backtest.positions) == 1
    assert backtest.positions[position_id] == position
    
    # Fermeture de la position
    backtest._close_position(position, 105.0, datetime.now(), 'take_profit')
    
    # Vérification de la fermeture
    assert position.exit_time is not None
    assert position.exit_price == 105.0
    assert position.pnl is not None
    assert len(backtest.trades_history) == 1
    assert backtest.trades_history[0] == position

def test_metrics_calculation(backtest):
    """Test le calcul des métriques"""
    # Création de trades de test
    trades = [
        Trade(
            symbol='BTC/USDT',
            entry_time=datetime.now() - timedelta(minutes=10),
            exit_time=datetime.now() - timedelta(minutes=5),
            entry_price=100.0,
            exit_price=105.0,
            side='long',
            size=1.0,
            stop_loss=95.0,
            take_profit=105.0,
            pnl=5.0,
            fees=0.1,
            metadata={'signal_strength': 0.8}
        ),
        Trade(
            symbol='BTC/USDT',
            entry_time=datetime.now() - timedelta(minutes=20),
            exit_time=datetime.now() - timedelta(minutes=15),
            entry_price=100.0,
            exit_price=95.0,
            side='long',
            size=1.0,
            stop_loss=95.0,
            take_profit=105.0,
            pnl=-5.0,
            fees=0.1,
            metadata={'signal_strength': 0.8}
        )
    ]
    
    # Ajout des trades
    backtest.trades_history.extend(trades)
    backtest.equity_curve = [10000, 10005, 10000]
    
    # Calcul des métriques
    metrics = backtest._calculate_metrics()
    
    # Vérification des métriques
    assert metrics['total_trades'] == 2
    assert metrics['winning_trades'] == 1
    assert metrics['losing_trades'] == 1
    assert metrics['win_rate'] == 0.5
    assert metrics['total_pnl'] == 0.0
    assert metrics['total_fees'] == 0.2
    assert metrics['max_drawdown'] == -0.0005
    assert metrics['sharpe_ratio'] == 0.0
    assert metrics['final_balance'] == 10000
    assert metrics['return'] == 0.0

def test_signal_handling(backtest):
    """Test le traitement des signaux"""
    # Création d'un signal de test
    signal = Signal(
        symbol='BTC/USDT',
        type=SignalType.BUY,
        strength=0.8,
        price=100.0,
        timestamp=datetime.now()
    )
    
    # Traitement du signal
    backtest._handle_signal(signal, 100.0, datetime.now())
    
    # Vérification de la position ouverte
    assert len(backtest.positions) == 1
    position = list(backtest.positions.values())[0]
    assert position.symbol == signal.symbol
    assert position.side == 'long'
    assert position.entry_price == 100.0
    assert position.size > 0
    assert position.stop_loss < 100.0
    assert position.take_profit > 100.0 