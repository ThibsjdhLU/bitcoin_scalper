import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..core.backtesting.backtester import Backtester
from ..core.strategies.bitcoin_scalper import BitcoinScalper

class TestBacktester(unittest.TestCase):
    """
    Tests pour le module de backtesting
    """
    
    def setUp(self):
        """
        Initialisation des tests
        """
        # Configuration de test
        self.config = {
            'risk_per_trade': 1.0,
            'max_positions': 3,
            'atr_period': 14,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
        
        # Création de données de test
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H')
        self.test_data = pd.DataFrame({
            'open': np.random.normal(30000, 1000, len(dates)),
            'high': np.random.normal(30100, 1000, len(dates)),
            'low': np.random.normal(29900, 1000, len(dates)),
            'close': np.random.normal(30000, 1000, len(dates)),
            'volume': np.random.normal(100, 10, len(dates))
        }, index=dates)
        
        # Initialisation des objets
        self.backtester = Backtester(self.config)
        self.strategy = BitcoinScalper(self.config)
        
    def test_run_backtest(self):
        """
        Test de l'exécution du backtest
        """
        # Exécution du backtest
        results = self.backtester.run_backtest(
            self.test_data,
            self.strategy,
            initial_capital=10000.0
        )
        
        # Vérifications
        self.assertIsInstance(results, dict)
        self.assertIn('total_trades', results)
        self.assertIn('win_rate', results)
        self.assertIn('total_profit', results)
        self.assertIn('sharpe_ratio', results)
        
    def test_calculate_metrics(self):
        """
        Test du calcul des métriques
        """
        # Création de trades de test
        self.backtester.trades = [
            {
                'entry_time': datetime.now(),
                'exit_time': datetime.now() + timedelta(hours=1),
                'entry_price': 30000,
                'exit_price': 30100,
                'direction': 'long',
                'size': 1.0,
                'profit': 100,
                'exit_reason': 'take_profit'
            },
            {
                'entry_time': datetime.now() + timedelta(hours=2),
                'exit_time': datetime.now() + timedelta(hours=3),
                'entry_price': 30100,
                'exit_price': 30000,
                'direction': 'short',
                'size': 1.0,
                'profit': 100,
                'exit_reason': 'take_profit'
            }
        ]
        
        # Calcul des métriques
        metrics = self.backtester._calculate_metrics()
        
        # Vérifications
        self.assertEqual(metrics['total_trades'], 2)
        self.assertEqual(metrics['winning_trades'], 2)
        self.assertEqual(metrics['win_rate'], 1.0)
        self.assertEqual(metrics['total_profit'], 200)
        self.assertEqual(metrics['average_profit'], 100)
        
    def test_plot_results(self):
        """
        Test de l'affichage des résultats
        """
        # Création de données de test pour l'affichage
        self.backtester.equity_curve = pd.Series(
            [10000, 10100, 10200, 10150, 10300],
            index=pd.date_range(start='2023-01-01', periods=5, freq='1H')
        )
        self.backtester.trades = [
            {
                'entry_time': datetime.now(),
                'exit_time': datetime.now() + timedelta(hours=1),
                'entry_price': 30000,
                'exit_price': 30100,
                'direction': 'long',
                'size': 1.0,
                'profit': 100,
                'exit_reason': 'take_profit'
            }
        ]
        
        # Test de l'affichage (ne devrait pas lever d'exception)
        try:
            self.backtester.plot_results()
        except Exception as e:
            self.fail(f"L'affichage des résultats a échoué: {str(e)}")
            
if __name__ == '__main__':
    unittest.main() 