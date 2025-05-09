"""
Tests unitaires pour le bot de trading
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.analysis.indicators import TechnicalIndicators
from src.connectors.metatrader import Exchange
from src.core.bot import TradingBot


class TestTradingBot(unittest.TestCase):
    """Tests pour la classe TradingBot"""

    def setUp(self):
        """Configuration initiale pour les tests"""
        self.config_patcher = patch("src.core.config.Config")
        self.mock_config = self.config_patcher.start()

        self.exchange_patcher = patch("src.connectors.metatrader.Exchange")
        self.mock_exchange = self.exchange_patcher.start()

        self.indicators_patcher = patch("src.analysis.indicators.TechnicalIndicators")
        self.mock_indicators = self.indicators_patcher.start()

        self.bot = TradingBot()

    def tearDown(self):
        """Nettoyage après les tests"""
        self.config_patcher.stop()
        self.exchange_patcher.stop()
        self.indicators_patcher.stop()

    def test_bot_initialization(self):
        """Test l'initialisation du bot"""
        self.assertIsNotNone(self.bot)
        self.assertFalse(self.bot.running)
        self.assertEqual(self.bot.daily_trades, 0)
        self.assertEqual(self.bot.daily_pnl, 0.0)

    def test_start_stop(self):
        """Test le démarrage et l'arrêt du bot"""
        self.bot.start()
        self.assertTrue(self.bot.running)

        self.bot.stop()
        self.assertFalse(self.bot.running)


if __name__ == "__main__":
    unittest.main()
