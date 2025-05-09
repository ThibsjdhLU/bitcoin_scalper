import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from src.bitcoin_scalper.ui.app import Dashboard


class TestDashboard(unittest.TestCase):
    """Tests d'intégration pour le dashboard."""

    def setUp(self):
        """Prépare l'environnement de test."""
        # Créer un répertoire temporaire pour les données de test
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.test_dir) / "data"
        self.logs_dir = Path(self.test_dir) / "logs"
        os.makedirs(self.data_dir)
        os.makedirs(self.logs_dir)

        # Créer des données de test
        self._create_test_data()

        # Initialiser le dashboard avec le répertoire de test
        self.dashboard = Dashboard()
        self.dashboard.data_dir = self.data_dir
        self.dashboard.logs_dir = self.logs_dir

    def _create_test_data(self):
        """Crée des données de test."""
        # Créer des données de trading
        trades_data = {
            "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="h"),
            "symbol": ["BTCUSD"] * 100,
            "strategy": ["EMA"] * 50 + ["RSI"] * 50,
            "side": ["BUY", "SELL"] * 50,
            "volume": np.random.uniform(0.01, 1.0, 100),
            "price": np.random.uniform(40000, 50000, 100),
            "profit": np.random.uniform(-100, 100, 100),
        }
        trades_df = pd.DataFrame(trades_data)
        trades_df.to_csv(self.data_dir / "trades.csv", index=False)

        # Créer des données de performance
        performance_data = {
            "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="h"),
            "balance": np.cumsum(np.random.uniform(-100, 100, 100)) + 10000,
            "equity": np.cumsum(np.random.uniform(-100, 100, 100)) + 10000,
        }
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_csv(self.data_dir / "performance.csv", index=False)

        # Créer des logs d'erreurs
        error_logs = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "error": "Connection error",
                "details": "Failed to connect to MT5",
            },
            {
                "timestamp": "2024-01-01T11:00:00",
                "error": "Order rejected",
                "details": "Invalid price",
            },
        ]
        for i, log in enumerate(error_logs):
            with open(self.logs_dir / f"error_{i}.json", "w") as f:
                json.dump(log, f)

    def test_data_loading(self):
        """Test le chargement des données."""
        self.dashboard.load_data()

        # Vérifier les données de trading
        self.assertIsInstance(self.dashboard.trades_df, pd.DataFrame)
        self.assertEqual(len(self.dashboard.trades_df), 100)
        self.assertIn("strategy", self.dashboard.trades_df.columns)

        # Vérifier les données de performance
        self.assertIsInstance(self.dashboard.performance_df, pd.DataFrame)
        self.assertEqual(len(self.dashboard.performance_df), 100)
        self.assertIn("equity", self.dashboard.performance_df.columns)

        # Vérifier les logs d'erreurs
        self.assertIsInstance(self.dashboard.errors_df, pd.DataFrame)
        self.assertEqual(len(self.dashboard.errors_df), 2)

    def test_metrics_calculation(self):
        """Test le calcul des métriques."""
        self.dashboard.load_data()
        self.dashboard.calculate_metrics()

        # Vérifier le win rate
        self.assertGreaterEqual(self.dashboard.win_rate, 0)
        self.assertLessEqual(self.dashboard.win_rate, 1)

        # Vérifier les win rates par stratégie
        self.assertIn("EMA", self.dashboard.strategy_win_rates)
        self.assertIn("RSI", self.dashboard.strategy_win_rates)

        # Vérifier le profit factor
        self.assertGreaterEqual(self.dashboard.profit_factor, 0)

        # Vérifier le max drawdown
        self.assertLessEqual(self.dashboard.max_drawdown, 0)

    def test_large_dataset(self):
        """Test avec un grand volume de données."""
        # Créer un grand dataset
        trades_data = {
            "timestamp": pd.date_range(start="2024-01-01", periods=10000, freq="h"),
            "symbol": ["BTCUSD"] * 10000,
            "strategy": ["EMA"] * 5000 + ["RSI"] * 5000,
            "side": ["BUY", "SELL"] * 5000,
            "volume": np.random.uniform(0.01, 1.0, 10000),
            "price": np.random.uniform(40000, 50000, 10000),
            "profit": np.random.uniform(-100, 100, 10000),
        }
        trades_df = pd.DataFrame(trades_data)
        trades_df.to_csv(self.data_dir / "trades.csv", index=False)

        # Mesurer le temps de chargement
        import time

        start_time = time.time()
        self.dashboard.load_data()
        load_time = time.time() - start_time

        # Vérifier que le chargement est rapide
        self.assertLess(load_time, 5.0)  # Moins de 5 secondes

    def test_data_refresh(self):
        """Test le rafraîchissement des données."""
        # Charger les données initiales
        self.dashboard.load_data()
        initial_trades = len(self.dashboard.trades_df)

        # Ajouter de nouvelles données
        new_trade = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "symbol": ["BTCUSD"],
                "strategy": ["EMA"],
                "side": ["BUY"],
                "volume": [0.1],
                "price": [45000],
                "profit": [50],
            }
        )
        new_trade.to_csv(
            self.data_dir / "trades.csv", mode="a", header=False, index=False
        )

        # Rafraîchir les données
        self.dashboard.load_data()

        # Vérifier que les nouvelles données sont chargées
        self.assertEqual(len(self.dashboard.trades_df), initial_trades + 1)

    def tearDown(self):
        """Nettoie l'environnement de test."""
        shutil.rmtree(self.test_dir)


if __name__ == "__main__":
    unittest.main()
