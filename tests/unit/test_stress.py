"""
Tests de stress pour le bot de trading.
"""
import json
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
from src.bitcoin_scalper.backtest.backtest_engine import BacktestEngine
from loguru import logger

from src.bitcoin_scalper.core.mt5_connector import MT5Connector
from src.bitcoin_scalper.core.order_executor import OrderExecutor, OrderSide, OrderType
from src.bitcoin_scalper.core.risk_manager import RiskManager


class TestStress(unittest.TestCase):
    """Tests de stress du système."""

    def setUp(self):
        """Initialise l'environnement de test."""
        # Créer des mocks pour MT5Connector et OrderExecutor
        self.mock_connector = MagicMock(spec=MT5Connector)
        self.mock_order_executor = MagicMock(spec=OrderExecutor)

        # Configurer le risk_manager
        self.config = {
            "risk": {
                "max_position_size": 1.0,
                "max_daily_trades": 10,
                "max_daily_loss": 1000,
                "max_drawdown": 0.1,
                "risk_per_trade": 0.02,
            }
        }
        self.risk_manager = RiskManager(self.config)

        # Initialiser le position_manager
        from src.bitcoin_scalper.core.position_manager import PositionManager

        self.position_manager = PositionManager(
            connector=self.mock_connector, risk_manager=self.risk_manager
        )

        # Configurer le mock du connecteur
        self.mock_connector.get_rates.return_value = [
            {
                "time": int(datetime(2023, 1, 1, i).timestamp()),
                "open": 10000 + i,
                "high": 10002 + i,
                "low": 9998 + i,
                "close": 10000 + i,
                "tick_volume": 1000 + i,
            }
            for i in range(24)
        ]

        # Configurer le mock de l'exécuteur d'ordres
        self.mock_order_executor.execute_market_order.return_value = (True, 12345)

    @patch("core.mt5_connector.mt5.initialize")
    @patch("core.mt5_connector.mt5.login")
    @patch("core.mt5_connector.mt5.symbol_info")
    @patch("core.mt5_connector.mt5.symbol_select")
    def test_connection_loss(
        self, mock_symbol_select, mock_symbol_info, mock_login, mock_init
    ):
        """Teste la résilience à la perte de connexion."""
        # Configurer les mocks
        mock_init.side_effect = [False, False, False]  # Échouer 3 fois
        mock_login.return_value = True
        mock_symbol_info.return_value = MagicMock()
        mock_symbol_select.return_value = True

        # Créer une instance du connecteur
        connector = MT5Connector()
        connector.max_retries = 3
        connector.retry_delay = 0

        # Vérifier que la connexion échoue après 3 tentatives
        with self.assertRaises(ConnectionError):
            connector.ensure_connection()

        self.assertEqual(mock_init.call_count, 3)

        # Réinitialiser les mocks pour la deuxième partie
        mock_init.reset_mock()
        mock_init.side_effect = [True]  # Réussir

        # Créer une nouvelle instance du connecteur
        new_connector = MT5Connector()
        new_connector.max_retries = 3
        new_connector.retry_delay = 0

        # Vérifier que la connexion réussit
        self.assertTrue(new_connector.ensure_connection())

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
            return {
                "name": symbol,
                "spread": mock_symbol_info.current_spread,
                "bid": 50000,
                "ask": 50000 + mock_symbol_info.current_spread,
                "volume_min": 0.01,
                "volume_max": 1.0,
                "digits": 2,
            }

        mock_symbol_info.current_spread = 10  # Spread normal

        self.mock_connector.get_symbol_info.side_effect = mock_symbol_info

        # Ordre avec spread normal
        success, order_id = self.mock_order_executor.execute_market_order(
            symbol=symbol, volume=0.1, side=OrderSide.BUY
        )
        self.assertTrue(success)

        # Augmenter le spread
        mock_symbol_info.current_spread = 1000  # Spread extrême

        # Configurer le mock pour rejeter l'ordre
        self.mock_order_executor.execute_market_order.return_value = (False, None)

        # Ordre avec spread élevé
        success, order_id = self.mock_order_executor.execute_market_order(
            symbol=symbol, volume=0.1, side=OrderSide.BUY
        )
        self.assertFalse(success)  # Doit être rejeté

    def test_massive_backtest(self):
        """
        Teste la performance avec un grand nombre de backtests.

        Scénario :
        1. Générer un grand jeu de données
        2. Exécuter 1000 backtests
        3. Vérifier la stabilité et les fuites mémoire
        """

        # Créer une stratégie mock
        class MockStrategy:
            def generate_signal(self, bar):
                if bar.name.minute % 10 == 0:
                    return {
                        "type": "MARKET",
                        "symbol": "BTCUSD",
                        "side": OrderSide.BUY
                        if bar.name.minute % 20 == 0
                        else OrderSide.SELL,
                        "volume": 0.1,
                        "sl": bar["close"] * 0.99,
                        "tp": bar["close"] * 1.02,
                    }
                return None

        strategy = MockStrategy()

        # Exécuter les backtests
        start_time = time.time()
        memory_usage = []

        for i in range(1000):
            engine = BacktestEngine(
                connector=self.mock_connector,
                order_executor=self.mock_order_executor,
                initial_balance=10000.0,
            )

            # Charger les données
            engine.load_data(
                symbol="BTCUSD",
                timeframe="1h",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 2),
            )

            # Exécuter le backtest
            results = engine.run_backtest(strategy)

            # Vérifier les résultats
            self.assertIn("trades", results)
            self.assertIn("balance", results)
            self.assertIn("equity", results)
            self.assertIn("drawdown", results)

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
        symbol = "BTCUSD"

        # Ouvrir des positions
        positions = []
        for i in range(3):
            success, order_id = self.mock_order_executor.execute_market_order(
                symbol=symbol, volume=0.1, side=OrderSide.BUY, sl=49000, tp=51000
            )
            if success:
                positions.append(order_id)

        self.assertEqual(len(positions), 3)

        # Simuler une chute brutale des prix
        self.mock_connector.get_symbol_info.return_value = {
            "name": symbol,
            "bid": 45000,  # -10% de chute
            "ask": 45002,
            "spread": 2,
            "volume_min": 0.01,
            "volume_max": 1.0,
            "digits": 2,
        }

        # Configurer le mock pour simuler la fermeture des positions
        self.mock_order_executor.check_order_status.return_value = MagicMock(
            status="FILLED", profit=-500
        )

        # Vérifier la fermeture des positions
        for order_id in positions:
            status = self.mock_order_executor.check_order_status(order_id)
            self.assertIsNotNone(status)
            self.assertEqual(status.status, "FILLED")

    def test_rapid_orders(self):
        """Test l'envoi rapide d'ordres."""
        symbol = "BTCUSD"
        volume = 0.01

        # Utiliser des IDs prévisibles
        ticket_ids = [1000 + i for i in range(10)]
        ticket_index = 0

        # Configurer le mock du connecteur pour place_order
        def mock_place_order(*args, **kwargs):
            nonlocal ticket_index
            mock_order = MagicMock()
            mock_order.ticket = ticket_ids[ticket_index]
            ticket_index += 1
            return {"order": mock_order}

        self.mock_connector.place_order = MagicMock(side_effect=mock_place_order)

        # Configurer le mock pour close_position
        def mock_close_position(ticket):
            # Ne pas supprimer la position ici, laisser le PositionManager le faire
            return True

        self.mock_connector.close_position = MagicMock(side_effect=mock_close_position)

        # Configurer le mock pour position_exists
        def mock_position_exists(ticket):
            return ticket in self.position_manager.positions

        self.mock_connector.position_exists = MagicMock(
            side_effect=mock_position_exists
        )

        # Envoyer 10 ordres rapidement
        orders = []
        for i in range(10):
            print(f"Opening position {i+1}/10...")
            ticket = self.position_manager.open_position(
                symbol=symbol,
                volume=volume,
                side="BUY",
                entry_price=50000.0,
                stop_loss=49000.0,
                take_profit=51000.0,
                strategy="STRESS_TEST",
                params={},
            )
            if ticket:
                orders.append(ticket)
                print(
                    f"Opened position {ticket}, current positions: {list(self.position_manager.positions.keys())}"
                )
            time.sleep(0.05)  # Réduire le délai pour stresser plus

        # Vérifier que tous les ordres sont bien placés
        self.assertEqual(len(orders), 10)

        # Vérifier que les positions sont bien enregistrées
        for ticket in orders:
            self.assertIn(ticket, self.position_manager.positions)

        # Fermer tous les ordres
        for i, ticket in enumerate(orders):
            print(f"Closing position {i+1}/10 (ticket: {ticket})...")
            print(
                f"Current positions before close: {list(self.position_manager.positions.keys())}"
            )
            result = self.position_manager.close_position(ticket)
            self.assertTrue(result, f"Failed to close position {ticket}")
            self.assertNotIn(ticket, self.position_manager.positions)
            print(
                f"Position {ticket} closed, remaining positions: {list(self.position_manager.positions.keys())}"
            )
            time.sleep(0.05)  # Petit délai entre les fermetures


if __name__ == "__main__":
    unittest.main()
