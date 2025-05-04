"""
Tests unitaires pour le gestionnaire de risques.
"""
import unittest
import json
from datetime import datetime
from core.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    """Tests pour le gestionnaire de risques."""
    
    def setUp(self):
        """Initialise le gestionnaire de risques."""
        with open('config/risk_config.json', 'r') as f:
            self.config = json.load(f)
            
        self.risk_manager = RiskManager(self.config)
        
    def test_initialization(self):
        """Teste l'initialisation du gestionnaire."""
        self.assertEqual(self.risk_manager.initial_capital, self.config['general']['initial_capital'])
        self.assertEqual(self.risk_manager.current_capital, self.config['general']['initial_capital'])
        self.assertEqual(self.risk_manager.peak_capital, self.config['general']['initial_capital'])
        
    def test_drawdown_check(self):
        """Teste la vérification du drawdown."""
        # Drawdown acceptable
        self.risk_manager.current_capital = self.risk_manager.initial_capital * 0.9
        self.assertTrue(self.risk_manager.check_drawdown())
        
        # Drawdown trop élevé
        self.risk_manager.current_capital = self.risk_manager.initial_capital * 0.8
        self.assertFalse(self.risk_manager.check_drawdown())
        
    def test_daily_limits(self):
        """Teste les limites journalières."""
        # Limites respectées
        self.risk_manager.daily_stats['pnl'] = self.risk_manager.initial_capital * 0.02
        self.risk_manager.daily_stats['trades'] = 5
        self.assertTrue(self.risk_manager.check_daily_limits())
        
        # Perte journalière trop élevée
        self.risk_manager.daily_stats['pnl'] = -self.risk_manager.initial_capital * 0.06
        self.assertFalse(self.risk_manager.check_daily_limits())
        
        # Profit journalier trop élevé
        self.risk_manager.daily_stats['pnl'] = self.risk_manager.initial_capital * 0.11
        self.assertFalse(self.risk_manager.check_daily_limits())
        
        # Nombre de trades trop élevé
        self.risk_manager.daily_stats['pnl'] = 0
        self.risk_manager.daily_stats['trades'] = 21
        self.assertFalse(self.risk_manager.check_daily_limits())
        
    def test_strategy_limits(self):
        """Teste les limites par stratégie."""
        strategy = "ema_crossover"
        
        # Limites respectées
        self.risk_manager.strategy_stats[strategy]['trades'] = 5
        self.risk_manager.strategy_stats[strategy]['pnl'] = self.risk_manager.initial_capital * 0.02
        self.assertTrue(self.risk_manager.check_strategy_limits(strategy))
        
        # Nombre de trades trop élevé
        self.risk_manager.strategy_stats[strategy]['trades'] = 11
        self.assertFalse(self.risk_manager.check_strategy_limits(strategy))
        
        # Perte journalière trop élevée
        self.risk_manager.strategy_stats[strategy]['trades'] = 5
        self.risk_manager.strategy_stats[strategy]['pnl'] = -self.risk_manager.initial_capital * 0.04
        self.assertFalse(self.risk_manager.check_strategy_limits(strategy))
        
        # Profit journalier trop élevé
        self.risk_manager.strategy_stats[strategy]['pnl'] = self.risk_manager.initial_capital * 0.07
        self.assertFalse(self.risk_manager.check_strategy_limits(strategy))
        
    def test_position_size_calculation(self):
        """Teste le calcul de la taille de position."""
        strategy = "ema_crossover"
        symbol = "BTCUSD"
        price = 50000
        stop_loss = 49000
        
        # Calcul normal
        size = self.risk_manager.calculate_position_size(strategy, symbol, price, stop_loss)
        self.assertGreater(size, 0)
        
        # Stop loss égal au prix
        size = self.risk_manager.calculate_position_size(strategy, symbol, price, price)
        self.assertEqual(size, 0)
        
        # Stop loss plus haut que le prix (short)
        size = self.risk_manager.calculate_position_size(strategy, symbol, price, price + 1000)
        self.assertGreater(size, 0)
        
    def test_can_open_position(self):
        """Teste la vérification d'ouverture de position."""
        strategy = "ema_crossover"
        symbol = "BTCUSD"
        price = 50000
        stop_loss = 49000
        
        # Position autorisée
        self.assertTrue(
            self.risk_manager.can_open_position(
                strategy=strategy,
                symbol=symbol,
                side="long",
                price=price,
                stop_loss=stop_loss
            )
        )
        
        # Drawdown trop élevé
        self.risk_manager.current_capital = self.risk_manager.initial_capital * 0.8
        self.assertFalse(
            self.risk_manager.can_open_position(
                strategy=strategy,
                symbol=symbol,
                side="long",
                price=price,
                stop_loss=stop_loss
            )
        )
        
    def test_trade_update(self):
        """Teste la mise à jour après un trade."""
        strategy = "ema_crossover"
        symbol = "BTCUSD"
        pnl = 100
        
        # Mise à jour normale
        initial_capital = self.risk_manager.current_capital
        self.risk_manager.on_trade(strategy, symbol, pnl)
        
        self.assertEqual(self.risk_manager.current_capital, initial_capital + pnl)
        self.assertEqual(self.risk_manager.daily_stats['trades'], 1)
        self.assertEqual(self.risk_manager.daily_stats['pnl'], pnl)
        self.assertEqual(self.risk_manager.strategy_stats[strategy]['trades'], 1)
        self.assertEqual(self.risk_manager.strategy_stats[strategy]['pnl'], pnl)
        
    def test_risk_metrics(self):
        """Teste la récupération des métriques de risque."""
        # Ajouter quelques trades
        self.risk_manager.on_trade("ema_crossover", "BTCUSD", 100)
        self.risk_manager.on_trade("rsi", "ETHUSD", -50)
        
        metrics = self.risk_manager.get_risk_metrics()
        
        self.assertIn('current_capital', metrics)
        self.assertIn('peak_capital', metrics)
        self.assertIn('current_drawdown', metrics)
        self.assertIn('daily_pnl', metrics)
        self.assertIn('daily_trades', metrics)
        self.assertIn('open_positions', metrics)
        self.assertIn('strategy_stats', metrics)
        
if __name__ == '__main__':
    unittest.main() 