"""
Tests unitaires pour la stratégie MACD.
"""
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.bitcoin_scalper.core.data_fetcher import DataFetcher, TimeFrame
from src.bitcoin_scalper.core.order_executor import OrderExecutor
from src.bitcoin_scalper.strategies.macd_strategy import MACDStrategy


class TestMACDStrategy(unittest.TestCase):
    """Tests pour la stratégie MACD."""

    def setUp(self):
        """Initialise les données de test."""
        # Création d'un DataFrame de test avec une tendance et des retournements
        self.data = pd.DataFrame(
            {
                "open": [
                    100,
                    102,
                    104,
                    103,
                    101,
                    99,
                    97,
                    96,
                    98,
                    100,
                    101,
                    103,
                    105,
                    104,
                    102,
                    100,
                    98,
                    97,
                    99,
                    101,
                ],
                "high": [
                    102,
                    104,
                    105,
                    104,
                    102,
                    100,
                    98,
                    97,
                    99,
                    101,
                    102,
                    104,
                    106,
                    105,
                    103,
                    101,
                    99,
                    98,
                    100,
                    102,
                ],
                "low": [
                    99,
                    101,
                    103,
                    102,
                    100,
                    98,
                    96,
                    95,
                    97,
                    99,
                    100,
                    102,
                    104,
                    103,
                    101,
                    99,
                    97,
                    96,
                    98,
                    100,
                ],
                "close": [
                    101,
                    103,
                    104,
                    102,
                    100,
                    98,
                    96,
                    97,
                    99,
                    100,
                    102,
                    104,
                    105,
                    103,
                    101,
                    99,
                    97,
                    98,
                    100,
                    101,
                ],
                "volume": [1000] * 20,
            }
        )

        # Créer des mocks pour les dépendances
        self.data_fetcher = Mock(spec=DataFetcher)
        self.order_executor = Mock(spec=OrderExecutor)

        # Initialiser la stratégie avec les paramètres de test
        self.strategy = MACDStrategy(
            data_fetcher=self.data_fetcher,
            order_executor=self.order_executor,
            symbols=["BTC/USD"],
            timeframe=TimeFrame.M5,
            params={
                "fast_period": 5,
                "slow_period": 10,
                "signal_period": 3,
                "trend_ema_period": 10,
                "min_histogram_change": 0.0001,
                "divergence_lookback": 5,
            },
        )

    def test_validate_params(self):
        """Teste la validation des paramètres."""
        # Test avec des paramètres invalides
        with self.assertRaises(ValueError):
            MACDStrategy(
                data_fetcher=self.data_fetcher,
                order_executor=self.order_executor,
                symbols=["BTC/USD"],
                timeframe=TimeFrame.M5,
                params={
                    "fast_period": 12,
                    "slow_period": 10,  # fast_period > slow_period
                },
            )

        with self.assertRaises(ValueError):
            MACDStrategy(
                data_fetcher=self.data_fetcher,
                order_executor=self.order_executor,
                symbols=["BTC/USD"],
                timeframe=TimeFrame.M5,
                params={
                    "slow_period": 20,
                    "signal_period": 25,  # signal_period > slow_period
                },
            )

        with self.assertRaises(ValueError):
            MACDStrategy(
                data_fetcher=self.data_fetcher,
                order_executor=self.order_executor,
                symbols=["BTC/USD"],
                timeframe=TimeFrame.M5,
                params={"min_histogram_change": 0},  # doit être > 0
            )

        with self.assertRaises(ValueError):
            MACDStrategy(
                data_fetcher=self.data_fetcher,
                order_executor=self.order_executor,
                symbols=["BTC/USD"],
                timeframe=TimeFrame.M5,
                params={"divergence_lookback": 3},  # doit être >= 5
            )

    def test_detect_divergence(self):
        """Teste la détection des divergences."""
        # Créer des données de test pour les divergences
        price = pd.Series([100, 98, 97, 96, 98])  # Prix baisse puis remonte
        macd = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])  # MACD monte constamment

        bull_div, bear_div = self.strategy._detect_divergence(price, macd, lookback=5)

        # Il devrait y avoir une divergence haussière
        self.assertTrue(bull_div)
        self.assertFalse(bear_div)

    def test_calculate_signals(self):
        """Teste le calcul des signaux d'achat et de vente."""
        buy_signals, sell_signals = self.strategy.calculate_signals(self.data)

        # Vérifie que les signaux sont des Series pandas
        self.assertIsInstance(buy_signals, pd.Series)
        self.assertIsInstance(sell_signals, pd.Series)

        # Vérifie la longueur des signaux
        self.assertEqual(len(buy_signals), len(self.data))
        self.assertEqual(len(sell_signals), len(self.data))

        # Vérifie qu'il n'y a pas de signaux contradictoires
        self.assertTrue(not any(buy_signals & sell_signals))

    def test_generate_trade_metadata(self):
        """Teste la génération des métadonnées de trade."""
        # Test pour un signal d'achat
        buy_metadata = self.strategy.generate_trade_metadata(
            self.data, index=7, signal_type="buy"  # Point bas
        )

        # Vérifie les clés requises
        required_keys = [
            "signal_type",
            "price",
            "macd",
            "signal",
            "histogram",
            "signal_strength",
            "trend_ema",
        ]
        for key in required_keys:
            self.assertIn(key, buy_metadata)

        # Vérifie le type de signal
        self.assertEqual(buy_metadata["signal_type"], "buy")

        # Test pour un signal de vente
        sell_metadata = self.strategy.generate_trade_metadata(
            self.data, index=12, signal_type="sell"  # Point haut
        )

        # Vérifie le type de signal
        self.assertEqual(sell_metadata["signal_type"], "sell")

    def test_calculate_stop_loss(self):
        """Teste le calcul du stop loss."""
        # Test pour un signal d'achat
        buy_stop = self.strategy.calculate_stop_loss(
            self.data, index=7, signal_type="buy"  # Point bas
        )

        # Le stop loss doit être inférieur au prix actuel
        self.assertLess(buy_stop, self.data["close"].iloc[7])

        # Test pour un signal de vente
        sell_stop = self.strategy.calculate_stop_loss(
            self.data, index=12, signal_type="sell"  # Point haut
        )

        # Le stop loss doit être supérieur au prix actuel
        self.assertGreater(sell_stop, self.data["close"].iloc[12])

    def test_calculate_take_profit(self):
        """Teste le calcul du take profit."""
        # Test pour un signal d'achat
        buy_tp = self.strategy.calculate_take_profit(
            self.data, index=7, signal_type="buy"  # Point bas
        )

        # Le take profit doit être supérieur au prix actuel
        self.assertGreater(buy_tp, self.data["close"].iloc[7])

        # Test pour un signal de vente
        sell_tp = self.strategy.calculate_take_profit(
            self.data, index=12, signal_type="sell"  # Point haut
        )

        # Le take profit doit être inférieur au prix actuel
        self.assertLess(sell_tp, self.data["close"].iloc[12])


if __name__ == "__main__":
    unittest.main()
