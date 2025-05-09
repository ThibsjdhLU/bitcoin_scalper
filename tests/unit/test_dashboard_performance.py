import json
import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from src.bitcoin_scalper.dashboard.app import Dashboard


class TestDashboardPerformance(unittest.TestCase):
    """Tests de performance pour le dashboard."""

    def setUp(self):
        """Prépare l'environnement de test."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.test_dir) / "data"
        self.logs_dir = Path(self.test_dir) / "logs"
        os.makedirs(self.data_dir)
        os.makedirs(self.logs_dir)

        self.dashboard = Dashboard()
        self.dashboard.data_dir = self.data_dir
        self.dashboard.logs_dir = self.logs_dir

    def _create_large_dataset(self, num_trades=100000, num_errors=1000):
        """Crée un grand jeu de données pour les tests de performance."""
        # Données de trading
        trades_data = {
            "timestamp": pd.date_range(
                start="2024-01-01", periods=num_trades, freq="min"
            ),
            "symbol": ["BTCUSD"] * num_trades,
            "strategy": np.random.choice(["EMA", "RSI", "BB", "MACD"], num_trades),
            "side": np.random.choice(["BUY", "SELL"], num_trades),
            "volume": np.random.uniform(0.01, 1.0, num_trades),
            "price": np.random.uniform(40000, 50000, num_trades),
            "profit": np.random.uniform(-100, 100, num_trades),
        }
        trades_df = pd.DataFrame(trades_data)
        trades_df.to_csv(self.data_dir / "trades.csv", index=False)

        # Données de performance
        performance_data = {
            "timestamp": pd.date_range(
                start="2024-01-01", periods=num_trades, freq="min"
            ),
            "balance": np.cumsum(np.random.uniform(-100, 100, num_trades)) + 10000,
            "equity": np.cumsum(np.random.uniform(-100, 100, num_trades)) + 10000,
        }
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_csv(self.data_dir / "performance.csv", index=False)

        # Logs d'erreurs
        for i in range(num_errors):
            error_log = {
                "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                "error": f"Error type {i % 5}",
                "details": f"Error details {i}",
            }
            with open(self.logs_dir / f"error_{i}.json", "w") as f:
                json.dump(error_log, f)

    def test_memory_usage(self):
        """Test l'utilisation de la mémoire."""
        self._create_large_dataset()

        # Mesurer l'utilisation de la mémoire avant
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Charger les données
        self.dashboard.load_data()

        # Mesurer l'utilisation de la mémoire après
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Vérifier que l'augmentation de la mémoire est raisonnable
        self.assertLess(memory_increase, 500)  # Moins de 500 MB

    def test_loading_performance(self):
        """Test les performances de chargement."""
        self._create_large_dataset()

        # Mesurer le temps de chargement
        start_time = time.time()
        self.dashboard.load_data()
        load_time = time.time() - start_time

        # Vérifier que le chargement est rapide
        self.assertLess(load_time, 10.0)  # Moins de 10 secondes

    def test_metrics_calculation_performance(self):
        """Test les performances du calcul des métriques."""
        self._create_large_dataset()
        self.dashboard.load_data()

        # Mesurer le temps de calcul des métriques
        start_time = time.time()
        self.dashboard.calculate_metrics()
        calculation_time = time.time() - start_time

        # Vérifier que le calcul est rapide
        self.assertLess(calculation_time, 5.0)  # Moins de 5 secondes

    def test_concurrent_refresh(self):
        """Test les performances lors de rafraîchissements concurrents."""
        self._create_large_dataset()
        self.dashboard.load_data()

        # Simuler des rafraîchissements concurrents
        refresh_times = []
        for _ in range(10):
            start_time = time.time()
            self.dashboard.load_data()
            refresh_times.append(time.time() - start_time)

        # Vérifier que les temps de rafraîchissement sont cohérents
        mean_refresh_time = sum(refresh_times) / len(refresh_times)
        self.assertLess(mean_refresh_time, 5.0)  # Moyenne inférieure à 5 secondes

    def test_data_aggregation_performance(self):
        """Test les performances de l'agrégation des données."""
        self._create_large_dataset()
        self.dashboard.load_data()

        # Mesurer le temps d'agrégation des données
        start_time = time.time()
        self.dashboard.calculate_metrics()
        aggregation_time = time.time() - start_time

        # Vérifier que l'agrégation est rapide
        self.assertLess(aggregation_time, 3.0)  # Moins de 3 secondes

    def tearDown(self):
        """Nettoie l'environnement de test."""
        shutil.rmtree(self.test_dir)


if __name__ == "__main__":
    unittest.main()
