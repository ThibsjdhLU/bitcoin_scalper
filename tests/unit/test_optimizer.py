"""
Tests unitaires pour le module d'optimisation.
"""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.bitcoin_scalper.utils.optimizer import MLStrategy, StrategyOptimizer


class MockStrategy:
    """Stratégie mock pour les tests."""

    def __init__(self, param1: float = 0.5, param2: float = 0.5):
        self.param1 = param1
        self.param2 = param2

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Génère des signaux mock."""
        signals = pd.Series(0, index=data.index)
        signals[data["close"] > data["close"].shift(1)] = 1
        signals[data["close"] < data["close"].shift(1)] = -1
        return signals


class TestStrategyOptimizer(unittest.TestCase):
    """Tests pour l'optimiseur de stratégie."""

    def setUp(self):
        """Initialise les données de test."""
        # Créer des données de test
        self.data = pd.DataFrame(
            {
                "open": np.random.randn(100).cumsum() + 100,
                "high": np.random.randn(100).cumsum() + 102,
                "low": np.random.randn(100).cumsum() + 98,
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000, 5000, 100),
            }
        )

        # Définir les plages de paramètres
        self.param_ranges = {"param1": (0.0, 1.0), "param2": (0.0, 1.0)}

        # Créer l'optimiseur
        self.optimizer = StrategyOptimizer(
            data=self.data,
            strategy_class=MockStrategy,
            param_ranges=self.param_ranges,
            metric="sharpe",
            cv_splits=3,
            random_state=42,
        )

    def test_calculate_metric(self):
        """Teste le calcul des métriques."""
        # Créer des données de test
        returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
        positions = pd.Series([1, 1, -1, -1, 0])

        # Tester chaque métrique
        metrics = ["sharpe", "sortino", "calmar", "profit"]
        for metric in metrics:
            self.optimizer.metric = metric
            score = self.optimizer._calculate_metric(returns, positions)
            self.assertIsInstance(score, float)

    def test_evaluate_params(self):
        """Teste l'évaluation des paramètres."""
        # Définir des paramètres de test
        params = {"param1": 0.5, "param2": 0.5}

        # Split les données
        train_data = self.data.iloc[:50]
        test_data = self.data.iloc[50:]

        # Évaluer les paramètres
        train_score, test_score = self.optimizer._evaluate_params(
            params, train_data, test_data
        )

        # Vérifier les résultats
        self.assertIsInstance(train_score, float)
        self.assertIsInstance(test_score, float)

    def test_grid_search(self):
        """Teste la recherche sur grille."""
        # Définir une grille de paramètres
        param_grid = {"param1": [0.0, 0.5, 1.0], "param2": [0.0, 0.5, 1.0]}

        # Lancer la recherche sur grille
        results = self.optimizer.grid_search(param_grid, verbose=False)

        # Vérifier les résultats
        self.assertIn("best_params", results)
        self.assertIn("train_score", results)
        self.assertIn("test_score", results)
        self.assertIn("history", results)

        # Vérifier les meilleurs paramètres
        best_params = results["best_params"]
        self.assertIn("param1", best_params)
        self.assertIn("param2", best_params)

    def test_random_search(self):
        """Teste la recherche aléatoire."""
        # Lancer la recherche aléatoire
        results = self.optimizer.random_search(n_iter=10, verbose=False)

        # Vérifier les résultats
        self.assertIn("best_params", results)
        self.assertIn("train_score", results)
        self.assertIn("test_score", results)
        self.assertIn("history", results)

        # Vérifier les meilleurs paramètres
        best_params = results["best_params"]
        self.assertIn("param1", best_params)
        self.assertIn("param2", best_params)

    def test_differential_evolution(self):
        """Teste l'évolution différentielle."""
        # Lancer l'évolution différentielle
        results = self.optimizer.differential_evolution(
            max_iter=10, popsize=5, verbose=False
        )

        # Vérifier les résultats
        self.assertIn("best_params", results)
        self.assertIn("train_score", results)
        self.assertIn("test_score", results)
        self.assertIn("convergence", results)

        # Vérifier les meilleurs paramètres
        best_params = results["best_params"]
        self.assertIn("param1", best_params)
        self.assertIn("param2", best_params)


class TestMLStrategy(unittest.TestCase):
    """Tests pour la stratégie ML."""

    def setUp(self):
        """Initialise les données de test."""
        # Créer des données de test
        self.data = pd.DataFrame(
            {
                "open": np.random.randn(100).cumsum() + 100,
                "high": np.random.randn(100).cumsum() + 102,
                "low": np.random.randn(100).cumsum() + 98,
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000, 5000, 100),
            }
        )

        # Créer la stratégie
        self.strategy = MLStrategy(
            model_type="rf", feature_window=10, prediction_window=5, random_state=42
        )

    def test_create_features(self):
        """Teste la création des features."""
        features = self.strategy._create_features(self.data)

        # Vérifier les features
        expected_features = [
            "returns",
            "log_returns",
            "sma_5",
            "ema_5",
            "sma_10",
            "ema_10",
            "sma_20",
            "ema_20",
            "sma_50",
            "ema_50",
            "volatility_5",
            "volatility_10",
            "volatility_20",
            "volume_ma5",
            "volume_ma20",
            "rsi",
            "macd",
        ]

        for feature in expected_features:
            self.assertIn(feature, features.columns)

    def test_create_labels(self):
        """Teste la création des labels."""
        labels = self.strategy._create_labels(self.data)

        # Vérifier les labels
        self.assertEqual(len(labels), len(self.data))
        self.assertTrue(all(label in [0, 1] for label in labels))

    def test_train(self):
        """Teste l'entraînement du modèle."""
        # Entraîner le modèle
        metrics = self.strategy.train(self.data)

        # Vérifier les métriques
        expected_metrics = ["accuracy", "precision", "recall", "f1"]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)

    def test_predict(self):
        """Teste les prédictions."""
        # Préparer les données d'entraînement
        train_data = pd.DataFrame(
            {
                "open": np.random.randn(100).cumsum() + 100,
                "high": np.random.randn(100).cumsum() + 102,
                "low": np.random.randn(100).cumsum() + 98,
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000, 5000, 100),
            }
        )

        # Entraîner le modèle
        metrics = self.strategy.train(train_data)
        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)

        # Préparer les données de test avec suffisamment de données pour les features
        test_data = pd.DataFrame(
            {
                "open": np.random.randn(50).cumsum() + 100,  # Augmenté à 50 points
                "high": np.random.randn(50).cumsum() + 102,
                "low": np.random.randn(50).cumsum() + 98,
                "close": np.random.randn(50).cumsum() + 100,
                "volume": np.random.randint(1000, 5000, 50),
            }
        )

        # Générer les features pour les données de test
        test_features = self.strategy._create_features(test_data)

        # Faire des prédictions
        predictions = self.strategy.predict(test_data)

        # Vérifier les prédictions
        self.assertEqual(
            len(predictions), len(test_features)
        )  # Vérifier contre les features
        self.assertTrue(all(p in [0, 1] for p in predictions))

    def test_different_models(self):
        """Teste différents types de modèles."""
        model_types = [
            "rf",
            "svm",
        ]  # Exclure xgb et lgb qui nécessitent des dépendances

        for model_type in model_types:
            strategy = MLStrategy(
                model_type=model_type,
                feature_window=10,
                prediction_window=5,
                random_state=42,
            )

            # Préparer les données
            X = self.data[["open", "high", "low", "close", "volume"]].values
            y = (self.data["close"].shift(-1) > self.data["close"]).astype(int)
            y = y[:-1]  # Supprimer la dernière ligne car pas de valeur future
            X = X[:-1]  # Supprimer la dernière ligne pour correspondre à y

            # Entraîner et prédire
            metrics = strategy.train(
                pd.DataFrame(X, columns=["open", "high", "low", "close", "volume"])
            )
            self.assertIsInstance(metrics, dict)
            self.assertIn("accuracy", metrics)
            self.assertIn("precision", metrics)
            self.assertIn("recall", metrics)
            self.assertIn("f1", metrics)


if __name__ == "__main__":
    unittest.main()
