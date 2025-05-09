"""
Test module for backtest engine functionality.
"""
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
from src.bitcoin_scalper.backtest.backtest_engine import BacktestEngine

from src.bitcoin_scalper.core.mt5_connector import MT5Connector
from src.bitcoin_scalper.core.order_executor import OrderExecutor, OrderSide, OrderType


class MockStrategy:
    """Stratégie mock pour les tests."""

    def __init__(self):
        self.name = "Mock Strategy"

    def generate_signal(self, bar: pd.Series) -> dict:
        """Génère un signal mock."""
        # Générer des signaux alternés
        if bar.name.minute % 10 == 0:  # Signal tous les 10 minutes
            side = OrderSide.BUY if bar.name.minute % 20 == 0 else OrderSide.SELL
            return {
                "type": "MARKET",
                "symbol": "BTCUSD",
                "side": side,
                "volume": 0.1,
                "sl": bar["close"] * 0.99
                if side == OrderSide.BUY
                else bar["close"] * 1.01,
                "tp": bar["close"] * 1.02
                if side == OrderSide.BUY
                else bar["close"] * 0.98,
            }
        return None


class TestBacktestEngine(unittest.TestCase):
    """Tests pour le moteur de backtest."""

    def setUp(self):
        """Initialise les données de test."""
        # Créer des mocks pour MT5Connector et OrderExecutor
        self.mock_connector = MagicMock(spec=MT5Connector)
        self.mock_order_executor = MagicMock(spec=OrderExecutor)

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

        # Créer une stratégie mock
        self.strategy = MockStrategy()

        # Créer le moteur de backtest
        self.engine = BacktestEngine(
            connector=self.mock_connector,
            order_executor=self.mock_order_executor,
            initial_balance=10000.0,
        )

    def test_load_data(self):
        """Teste le chargement des données."""
        success = self.engine.load_data(
            symbol="BTCUSD",
            timeframe="1h",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 2),
        )

        self.assertTrue(success)
        self.assertFalse(self.engine.data.empty)
        self.assertEqual(len(self.engine.data), 24)

    def test_run_backtest(self):
        """Teste l'exécution du backtest."""
        # Charger les données
        self.engine.load_data(
            symbol="BTCUSD",
            timeframe="1h",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 2),
        )

        # Exécuter le backtest
        results = self.engine.run_backtest(self.strategy)

        # Vérifier les résultats
        self.assertIn("trades", results)
        self.assertIn("balance", results)
        self.assertIn("equity", results)
        self.assertIn("drawdown", results)
        self.assertIn("win_rate", results)
        self.assertIn("profit_factor", results)

    def test_execute_signal(self):
        """Teste l'exécution d'un signal."""
        # Créer un signal mock
        signal = {
            "type": "MARKET",
            "symbol": "BTCUSD",
            "side": OrderSide.BUY,
            "volume": 0.1,
            "sl": 9900,
            "tp": 10200,
        }

        # Créer une barre mock
        bar = pd.Series(
            {
                "open": 10000,
                "high": 10002,
                "low": 9998,
                "close": 10000,
                "tick_volume": 1000,
            },
            name=datetime(2023, 1, 1),
        )

        # Exécuter le signal
        success, order_id = self.engine._execute_signal(signal, bar)

        self.assertTrue(success)
        self.assertEqual(order_id, 12345)

    def test_update_results(self):
        """Teste la mise à jour des résultats."""
        # Créer des résultats initiaux
        results = {"trades": [], "balance": [10000], "equity": [10000], "drawdown": [0]}

        # Créer un signal mock
        signal = {
            "type": "MARKET",
            "symbol": "BTCUSD",
            "side": OrderSide.BUY,
            "volume": 0.1,
            "sl": 9900,
            "tp": 10200,
        }

        # Créer une barre mock
        bar = pd.Series(
            {
                "open": 10000,
                "high": 10002,
                "low": 9998,
                "close": 10000,
                "tick_volume": 1000,
            },
            name=datetime(2023, 1, 1),
        )

        # Mettre à jour les résultats
        self.engine._update_results(results, signal, bar)

        self.assertEqual(len(results["trades"]), 1)
        self.assertEqual(len(results["balance"]), 2)
        self.assertEqual(len(results["equity"]), 2)
        self.assertEqual(len(results["drawdown"]), 2)

    def test_calculate_metrics(self):
        """Teste le calcul des métriques."""
        # Créer des résultats avec des trades
        results = {
            "trades": [
                {"pnl": 100},  # Trade gagnant
                {"pnl": -50},  # Trade perdant
                {"pnl": 200},  # Trade gagnant
                {"pnl": -30},  # Trade perdant
            ],
            "balance": [10000, 10100, 10050, 10250, 10220],
            "equity": [10000, 10100, 10050, 10250, 10220],
            "drawdown": [0, 0, 0.5, 0, 0],
        }

        # Calculer les métriques
        self.engine._calculate_metrics(results)

        self.assertEqual(results["win_rate"], 50.0)  # 2 trades gagnants sur 4
        self.assertEqual(results["total_trades"], 4)
        self.assertEqual(results["winning_trades"], 2)
        self.assertEqual(results["losing_trades"], 2)


if __name__ == "__main__":
    unittest.main()
