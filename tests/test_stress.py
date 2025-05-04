"""
Tests de stress pour le bot de trading.
"""
import unittest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from core.mt5_connector import MT5Connector
from core.order_executor import OrderExecutor
from core.risk_manager import RiskManager
from backtest.backtest_engine import BacktestEngine
from strategies.base_strategy import BaseStrategy

class TestStress(unittest.TestCase):
    """Tests de stress du système."""
    
    def setUp(self):
        """Initialise l'environnement de test."""
        # Charger les configurations
        with open('config/risk_config.json', 'r') as f:
            self.risk_config = json.load(f)
        with open('config/mt5_config.json', 'r') as f:
            self.mt5_config = json.load(f)
            
        # Initialiser les composants
        self.mt5_connector = MT5Connector(**self.mt5_config)
        self.order_executor = OrderExecutor(self.mt5_connector)
        self.risk_manager = RiskManager(self.risk_config)
        
    def test_connection_loss(self):
        """
        Teste la résilience à la perte de connexion.
        
        Scénario :
        1. Démarrer normalement
        2. Simuler une perte de connexion
        3. Vérifier la reconnexion
        4. Vérifier la reprise des opérations
        """
        # Patch de la connexion MT5
        with patch('MetaTrader5.initialize') as mock_init:
            # Simuler une séquence de connexion/déconnexion
            mock_init.side_effect = [True, False, False, True]
            
            # Première connexion
            self.assertTrue(self.mt5_connector.ensure_connection())
            
            # Simuler une perte de connexion
            mock_init.reset_mock()
            
            # Tenter une opération
            symbol_info = self.mt5_connector.get_symbol_info("BTCUSD")
            self.assertIsNotNone(symbol_info)
            
            # Vérifier les tentatives de reconnexion
            self.assertEqual(mock_init.call_count, 2)
            
    def test_high_spread(self):
        """
        Teste le comportement avec des spreads élevés.
        
        Scénario :
        1. Configurer un spread normal
        2. Placer des ordres
        3. Augmenter brutalement le spread
        4. Vérifier le rejet des ordres
        """
        symbol = "BTCUSD"
        
        # Mock du spread
        def mock_symbol_info(symbol):
            return Mock(
                spread=mock_symbol_info.current_spread,
                bid=50000,
                ask=50000 + mock_symbol_info.current_spread
            )
            
        mock_symbol_info.current_spread = 10  # Spread normal
        
        with patch('MetaTrader5.symbol_info', side_effect=mock_symbol_info):
            # Ordre avec spread normal
            order1 = self.order_executor.place_order(
                symbol=symbol,
                order_type="BUY",
                volume=0.1,
                price=50000
            )
            self.assertIsNotNone(order1)
            
            # Augmenter le spread
            mock_symbol_info.current_spread = 1000  # Spread extrême
            
            # Ordre avec spread élevé
            order2 = self.order_executor.place_order(
                symbol=symbol,
                order_type="BUY",
                volume=0.1,
                price=50000
            )
            self.assertIsNone(order2)  # Doit être rejeté
            
    def test_massive_backtest(self):
        """
        Teste la performance avec un grand nombre de backtests.
        
        Scénario :
        1. Générer un grand jeu de données
        2. Exécuter 1000 backtests
        3. Vérifier la stabilité et les fuites mémoire
        """
        # Générer des données de test
        dates = pd.date_range(
            start='2020-01-01',
            end='2023-12-31',
            freq='1H'
        )
        
        data = {
            'BTCUSD': pd.DataFrame({
                'open': np.random.randn(len(dates)).cumsum() + 10000,
                'high': np.random.randn(len(dates)).cumsum() + 10002,
                'low': np.random.randn(len(dates)).cumsum() + 9998,
                'close': np.random.randn(len(dates)).cumsum() + 10000,
                'volume': np.random.randint(1000, 5000, len(dates))
            }, index=dates)
        }
        
        # Créer une stratégie simple
        strategy = BaseStrategy(
            name="Test Strategy",
            description="Strategy for stress testing",
            data_fetcher=None,
            order_executor=None,
            params={},
            symbols=[],
            timeframe=None
        )
        
        # Exécuter les backtests
        start_time = time.time()
        memory_usage = []
        
        for i in range(1000):
            engine = BacktestEngine(
                data=data,
                strategies=[strategy],
                initial_capital=10000.0
            )
            results = engine.run()
            
            # Vérifier les résultats
            self.assertIn('trades', results)
            self.assertIn('equity_curve', results)
            self.assertIn('metrics', results)
            
            # Mesurer l'utilisation mémoire
            import psutil
            process = psutil.Process()
            memory_usage.append(process.memory_info().rss)
            
            if i % 100 == 0:
                logger.info(f"Backtest {i}/1000 complété")
                
        end_time = time.time()
        
        # Analyser les performances
        duration = end_time - start_time
        avg_memory = sum(memory_usage) / len(memory_usage)
        max_memory = max(memory_usage)
        
        logger.info(f"Durée totale: {duration:.2f}s")
        logger.info(f"Mémoire moyenne: {avg_memory/1024/1024:.2f}MB")
        logger.info(f"Mémoire max: {max_memory/1024/1024:.2f}MB")
        
    def test_sudden_drawdown(self):
        """
        Teste la réaction à un drawdown brutal.
        
        Scénario :
        1. Ouvrir plusieurs positions
        2. Simuler une chute brutale des prix
        3. Vérifier la gestion du risque
        """
        # Configuration initiale
        self.risk_manager.current_capital = 10000
        symbol = "BTCUSD"
        
        # Ouvrir des positions
        positions = []
        for i in range(3):
            order = self.order_executor.place_order(
                symbol=symbol,
                order_type="BUY",
                volume=0.1,
                price=50000,
                stop_loss=49000
            )
            if order:
                positions.append(order)
                
        self.assertEqual(len(positions), 3)
        
        # Simuler une chute brutale
        def mock_position_get():
            return [
                Mock(
                    symbol=symbol,
                    type=pos.type,
                    volume=pos.volume,
                    price_open=pos.price,
                    price_current=45000,  # -10% de chute
                    sl=pos.stop_loss,
                    tp=pos.take_profit
                )
                for pos in positions
            ]
            
        with patch('MetaTrader5.positions_get', side_effect=mock_position_get):
            # Vérifier les limites
            self.assertFalse(self.risk_manager.check_drawdown())
            
            # Vérifier la fermeture des positions
            for pos in positions:
                status = self.order_executor.check_order_status(pos.order_id)
                self.assertIsNotNone(status)
                self.assertEqual(status.status, 'FILLED')
                
if __name__ == '__main__':
    unittest.main() 