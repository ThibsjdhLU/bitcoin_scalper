"""
Tests unitaires pour la stratégie Bollinger Bands Reversal.
"""
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.bitcoin_scalper.core.data_fetcher import DataFetcher, TimeFrame
from src.bitcoin_scalper.core.order_executor import OrderExecutor
from src.bitcoin_scalper.strategies.bollinger_bands_reversal import BollingerBandsReversalStrategy


class TestBollingerBandsReversalStrategy(unittest.TestCase):
    """Tests pour la stratégie Bollinger Bands Reversal."""

    def setUp(self):
        """Initialise les données de test."""
        # Création d'un DataFrame de test avec une tendance et des retournements
        self.data = pd.DataFrame(
            {
                "open": [100, 102, 104, 101, 98, 96, 94, 93, 95, 98],
                "high": [103, 104, 105, 102, 99, 97, 95, 94, 97, 100],
                "low": [99, 101, 103, 97, 96, 94, 92, 91, 94, 97],
                "close": [102, 104, 101, 98, 96, 94, 93, 95, 98, 99],
                "volume": [1000] * 10,
            }
        )

        # Créer des mocks pour les dépendances
        self.data_fetcher = Mock(spec=DataFetcher)
        self.order_executor = Mock(spec=OrderExecutor)

        # Initialiser la stratégie avec les paramètres de test
        self.strategy = BollingerBandsReversalStrategy(
            data_fetcher=self.data_fetcher,
            order_executor=self.order_executor,
            symbols=["BTC/USD"],
            timeframe=TimeFrame.M5,
            params={
                "bb_period": 5,
                "bb_std": 2.0,
                "rsi_period": 5,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "min_reversal_pct": 0.5,
            },
        )

    def test_validate_params(self):
        """Teste la validation des paramètres."""
        # Test avec des paramètres invalides
        with self.assertRaises(ValueError):
            BollingerBandsReversalStrategy(
                data_fetcher=self.data_fetcher,
                order_executor=self.order_executor,
                symbols=["BTC/USD"],
                timeframe=TimeFrame.M5,
                params={"bb_period": 1},  # doit être >= 2
            )

        with self.assertRaises(ValueError):
            BollingerBandsReversalStrategy(
                data_fetcher=self.data_fetcher,
                order_executor=self.order_executor,
                symbols=["BTC/USD"],
                timeframe=TimeFrame.M5,
                params={"bb_std": 0},  # doit être > 0
            )

        with self.assertRaises(ValueError):
            BollingerBandsReversalStrategy(
                data_fetcher=self.data_fetcher,
                order_executor=self.order_executor,
                symbols=["BTC/USD"],
                timeframe=TimeFrame.M5,
                params={
                    "rsi_oversold": 40,
                    "rsi_overbought": 30,  # doit être > oversold
                },
            )

        with self.assertRaises(ValueError):
            BollingerBandsReversalStrategy(
                data_fetcher=self.data_fetcher,
                order_executor=self.order_executor,
                symbols=["BTC/USD"],
                timeframe=TimeFrame.M5,
                params={"min_reversal_pct": 0},  # doit être > 0
            )

    def test_calculate_signals(self):
        """Teste le calcul des signaux d'achat et de vente."""
        buy_signals, sell_signals = self.strategy.calculate_signals(self.data)

        # Vérifie que les signaux sont des Series pandas
        self.assertIsInstance(buy_signals, pd.Series)
        self.assertIsInstance(sell_signals, pd.Series)

        # Vérifie la longueur des signaux
        self.assertEqual(len(buy_signals), len(self.data))
        self.assertEqual(len(sell_signals), len(self.data))

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
            "rsi",
            "band_distance",
            "signal_strength",
            "bb_upper",
            "bb_middle",
            "bb_lower",
        ]
        for key in required_keys:
            self.assertIn(key, buy_metadata)

        # Vérifie le type de signal
        self.assertEqual(buy_metadata["signal_type"], "buy")

        # Test pour un signal de vente
        sell_metadata = self.strategy.generate_trade_metadata(
            self.data, index=2, signal_type="sell"  # Point haut
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
            self.data, index=2, signal_type="sell"  # Point haut
        )

        # Le stop loss doit être supérieur au prix actuel
        self.assertGreater(sell_stop, self.data["close"].iloc[2])

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
            self.data, index=2, signal_type="sell"  # Point haut
        )

        # Le take profit doit être inférieur au prix actuel
        self.assertLess(sell_tp, self.data["close"].iloc[2])


if __name__ == "__main__":
    unittest.main()
