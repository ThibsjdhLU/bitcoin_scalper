"""
Tests unitaires pour le gestionnaire de risques.
"""
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from core.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    """Tests pour le gestionnaire de risques."""
    
    def setUp(self):
        """Initialisation des tests."""
        self.config = {
            'risk': {
                'max_position_size': 1.0,
                'max_daily_trades': 5,
                'max_daily_loss': 1000,
                'max_drawdown': 0.1,
                'risk_per_trade': 0.02
            }
        }
        self.risk_manager = RiskManager(self.config)
        self.mock_mt5 = MagicMock()
        self.mock_mt5.symbol_info_tick.return_value = MagicMock(
            ask=50000.0,
            bid=49900.0
        )
        
    def test_initialization(self):
        """Teste l'initialisation du gestionnaire de risques."""
        self.assertEqual(self.risk_manager.max_position_size, 1.0)
        self.assertEqual(self.risk_manager.max_daily_trades, 5)
        self.assertEqual(self.risk_manager.max_daily_loss, 1000)
        self.assertEqual(self.risk_manager.max_drawdown, 0.1)
        self.assertEqual(self.risk_manager.risk_per_trade, 0.02)
        
    def test_can_open_position(self):
        """Teste la vérification d'ouverture de position."""
        # Test avec des limites respectées
        self.assertTrue(self.risk_manager.can_open_position(
            symbol="BTCUSD",
            volume=0.1,
            mt5_connector=self.mock_mt5
        ))
        
        # Test avec volume trop grand
        self.assertFalse(self.risk_manager.can_open_position(
            symbol="BTCUSD",
            volume=2.0,
            mt5_connector=self.mock_mt5
        ))
        
    def test_position_size_calculation(self):
        """Teste le calcul de la taille de position."""
        account_balance = 10000
        stop_loss_pips = 100
        
        position_size = self.risk_manager.calculate_position_size(
            account_balance=account_balance,
            stop_loss_pips=stop_loss_pips,
            symbol="BTCUSD",
            mt5_connector=self.mock_mt5
        )
        
        self.assertLessEqual(position_size, self.config['risk']['max_position_size'])
        self.assertGreater(position_size, 0)
        
    def test_daily_limits(self):
        """Teste les limites quotidiennes."""
        # Simuler des trades
        for _ in range(5):
            self.risk_manager.update_trade(profit=100)
            
        # Vérifier que le 6ème trade est rejeté
        self.assertFalse(self.risk_manager.can_open_position(
            symbol="BTCUSD",
            volume=0.1,
            mt5_connector=self.mock_mt5
        ))
        
    def test_drawdown_check(self):
        """Teste la vérification du drawdown."""
        # Simuler des pertes importantes
        self.risk_manager.update_trade(profit=-2000)  # -20% du capital initial
        self.risk_manager.update_trade(profit=-1000)  # -10% supplémentaire
        
        # Vérifier que le drawdown est détecté
        self.assertFalse(self.risk_manager.can_open_position(
            symbol="BTCUSD",
            volume=0.1,
            mt5_connector=self.mock_mt5
        ))
        
    def test_risk_metrics(self):
        """Teste le calcul des métriques de risque."""
        # Simuler des trades
        self.risk_manager.update_trade(profit=100)
        self.risk_manager.update_trade(profit=-50)
        self.risk_manager.update_trade(profit=200)
        
        metrics = self.risk_manager.get_risk_metrics()
        
        self.assertIn('win_rate', metrics)
        self.assertIn('profit_factor', metrics)
        self.assertIn('average_win', metrics)
        self.assertIn('average_loss', metrics)
        
    def test_strategy_limits(self):
        """Teste les limites par stratégie."""
        strategy = "EMA_CROSSOVER"
        
        # Simuler des trades pour une stratégie
        for _ in range(3):
            self.risk_manager.update_trade(profit=100, strategy=strategy)
            
        # Vérifier les limites
        self.assertTrue(self.risk_manager.can_open_position(
            symbol="BTCUSD",
            volume=0.1,
            mt5_connector=self.mock_mt5,
            strategy=strategy
        ))
        
    def test_trade_update(self):
        """Teste la mise à jour des trades."""
        # Simuler un trade
        self.risk_manager.update_trade(profit=100)
        
        # Vérifier les métriques
        metrics = self.risk_manager.get_risk_metrics()
        self.assertEqual(metrics['total_trades'], 1)
        self.assertEqual(metrics['total_profit'], 100)
        
if __name__ == '__main__':
    unittest.main() 