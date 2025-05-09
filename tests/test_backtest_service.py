"""
Tests unitaires pour le service de backtesting.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.bitcoin_scalper.services.backtest_service import BacktestService

class TestBacktestService(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test."""
        self.service = BacktestService()
        
    def test_load_historical_data(self):
        """Test du chargement des données historiques."""
        # Création de données de test
        data = {
            'timestamp': [datetime.now()],
            'open': [50000.0],
            'high': [51000.0],
            'low': [49000.0],
            'close': [50500.0],
            'volume': [100.0]
        }
        df = pd.DataFrame(data)
        df.to_csv('test_data.csv', index=False)
        
        # Test
        result = self.service.load_historical_data('test_data.csv')
        
        # Vérifications
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertTrue(isinstance(result.index, pd.DatetimeIndex))
        
    def test_run_backtest(self):
        """Test de l'exécution d'un backtest."""
        # Création de données de test
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        data = pd.DataFrame({
            'open': np.random.normal(50000, 1000, 100),
            'high': np.random.normal(51000, 1000, 100),
            'low': np.random.normal(49000, 1000, 100),
            'close': np.random.normal(50500, 1000, 100),
            'volume': np.random.normal(100, 10, 100)
        }, index=dates)
        
        # Paramètres de stratégie
        strategy_params = {
            'ma_period': 20,
            'stop_loss': 0.02,
            'take_profit': 0.04
        }
        
        # Test
        results = self.service.run_backtest(data, strategy_params)
        
        # Vérifications
        self.assertIsNotNone(results)
        self.assertIn('trades', results)
        self.assertIn('equity_curve', results)
        self.assertIn('metrics', results)
        
    def test_calculate_metrics(self):
        """Test du calcul des métriques."""
        # Création de données de test
        trades = [
            {
                'entry_time': datetime.now(),
                'exit_time': datetime.now() + timedelta(hours=1),
                'entry_price': 50000.0,
                'exit_price': 51000.0,
                'profit': 0.02,
                'capital': 10200.0
            },
            {
                'entry_time': datetime.now() + timedelta(hours=2),
                'exit_time': datetime.now() + timedelta(hours=3),
                'entry_price': 51000.0,
                'exit_price': 50000.0,
                'profit': -0.02,
                'capital': 10000.0
            }
        ]
        
        # Test
        metrics = self.service._calculate_metrics(trades, 10000.0)
        
        # Vérifications
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics['total_trades'], 2)
        self.assertEqual(metrics['win_rate'], 0.5)
        self.assertGreater(metrics['max_drawdown'], 0)
        
    def test_should_enter_long(self):
        """Test de la logique d'entrée en position."""
        # Création de données de test
        candle = pd.Series({
            'open': 50000.0,
            'high': 51000.0,
            'low': 49000.0,
            'close': 50500.0,
            'volume': 100.0
        })
        
        # Paramètres de stratégie
        params = {'ma_period': 20}
        
        # Test
        result = self.service._should_enter_long(candle, params)
        
        # Vérification
        self.assertIsInstance(result, bool)
        
    def test_should_exit_long(self):
        """Test de la logique de sortie de position."""
        # Création de données de test
        candle = pd.Series({
            'open': 50000.0,
            'high': 51000.0,
            'low': 49000.0,
            'close': 49500.0,
            'volume': 100.0
        })
        
        # Paramètres de stratégie
        params = {'ma_period': 20}
        
        # Test
        result = self.service._should_exit_long(candle, params)
        
        # Vérification
        self.assertIsInstance(result, bool)
        
    def test_get_results(self):
        """Test de la récupération des résultats."""
        # Configuration des résultats
        self.service.results = {
            'trades': [],
            'equity_curve': [],
            'metrics': {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        }
        
        # Test
        results = self.service.get_results()
        
        # Vérification
        self.assertIsNotNone(results)
        self.assertEqual(results, self.service.results)

if __name__ == '__main__':
    unittest.main() 