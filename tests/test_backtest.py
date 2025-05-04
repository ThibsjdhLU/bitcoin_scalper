"""
Tests unitaires pour le moteur de backtest.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from backtest.backtest_engine import BacktestEngine
from strategies.base_strategy import BaseStrategy, Signal, SignalType

class MockStrategy(BaseStrategy):
    """Stratégie mock pour les tests."""
    
    def __init__(self, name: str = "Mock Strategy"):
        super().__init__(
            name=name,
            description="Stratégie mock pour les tests",
            data_fetcher=None,
            order_executor=None,
            params={},
            symbols=[],
            timeframe=None
        )
        
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> list:
        """Génère des signaux mock."""
        signals = []
        
        # Générer des signaux alternés
        for i in range(len(data)):
            if i % 10 == 0:  # Signal tous les 10 points
                signal_type = SignalType.BUY if i % 20 == 0 else SignalType.SELL
                signal = Signal(
                    type=signal_type,
                    symbol=symbol,
                    timestamp=data.index[i],
                    price=data['close'].iloc[i],
                    strength=0.8,
                    metadata={}
                )
                signals.append(signal)
                
        return signals

class TestBacktestEngine(unittest.TestCase):
    """Tests pour le moteur de backtest."""
    
    def setUp(self):
        """Initialise les données de test."""
        # Créer des données de test
        dates = pd.date_range(
            start='2023-01-01',
            end='2023-01-10',
            freq='1H'
        )
        
        self.data = {
            'BTCUSD': pd.DataFrame({
                'open': np.random.randn(len(dates)).cumsum() + 10000,
                'high': np.random.randn(len(dates)).cumsum() + 10002,
                'low': np.random.randn(len(dates)).cumsum() + 9998,
                'close': np.random.randn(len(dates)).cumsum() + 10000,
                'volume': np.random.randint(1000, 5000, len(dates))
            }, index=dates)
        }
        
        # Créer une stratégie mock
        self.strategy = MockStrategy()
        
        # Créer le moteur de backtest
        self.engine = BacktestEngine(
            data=self.data,
            strategies=[self.strategy],
            initial_capital=10000.0,
            commission=0.001,
            slippage=0.0005
        )
        
    def test_validate_data(self):
        """Teste la validation des données."""
        # Données valides
        self.engine._validate_data()
        
        # Données invalides (colonne manquante)
        invalid_data = self.data.copy()
        invalid_data['BTCUSD'] = invalid_data['BTCUSD'].drop('close', axis=1)
        
        with self.assertRaises(ValueError):
            engine = BacktestEngine(
                data=invalid_data,
                strategies=[self.strategy]
            )
            
    def test_calculate_trade_pnl(self):
        """Teste le calcul du P&L d'un trade."""
        # Trade long gagnant
        pnl_long = self.engine._calculate_trade_pnl(
            entry_price=10000,
            exit_price=10100,
            position_size=1.0,
            side='long'
        )
        self.assertGreater(pnl_long, 0)
        
        # Trade short gagnant
        pnl_short = self.engine._calculate_trade_pnl(
            entry_price=10100,
            exit_price=10000,
            position_size=1.0,
            side='short'
        )
        self.assertGreater(pnl_short, 0)
        
    def test_calculate_position_size(self):
        """Teste le calcul de la taille de position."""
        size = self.engine._calculate_position_size(
            capital=10000,
            price=10000,
            risk_per_trade=0.02
        )
        self.assertGreater(size, 0)
        
    def test_run(self):
        """Teste l'exécution du backtest."""
        results = self.engine.run()
        
        # Vérifier les résultats
        self.assertIn('trades', results)
        self.assertIn('equity_curve', results)
        self.assertIn('metrics', results)
        
        # Vérifier les métriques
        metrics = results['metrics']
        self.assertIn('total_trades', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('total_pnl', metrics)
        self.assertIn('sharpe_ratio', metrics)
        
    def test_plot_results(self):
        """Teste l'affichage des résultats."""
        # Exécuter le backtest
        self.engine.run()
        
        # Tester l'affichage (ne devrait pas lever d'exception)
        self.engine.plot_results()
        
    def test_save_results(self):
        """Teste la sauvegarde des résultats."""
        # Exécuter le backtest
        self.engine.run()
        
        # Sauvegarder les résultats
        self.engine.save_results('test_results')
        
        # Vérifier que les fichiers ont été créés
        import os
        self.assertTrue(os.path.exists('test_results_trades.csv'))
        self.assertTrue(os.path.exists('test_results_equity.csv'))
        self.assertTrue(os.path.exists('test_results_metrics.csv'))
        
        # Nettoyer
        os.remove('test_results_trades.csv')
        os.remove('test_results_equity.csv')
        os.remove('test_results_metrics.csv')
        
if __name__ == '__main__':
    unittest.main() 