"""
Module d'optimisation des paramètres et d'apprentissage automatique.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import differential_evolution
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler


class StrategyOptimizer:
    """
    Optimiseur de paramètres pour les stratégies de trading.

    Utilise différentes méthodes d'optimisation :
    - Grid Search : recherche exhaustive sur une grille de paramètres
    - Random Search : recherche aléatoire dans l'espace des paramètres
    - Differential Evolution : algorithme évolutionnaire pour l'optimisation globale
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_class: Any,
        param_ranges: Dict[str, Tuple[float, float]],
        metric: str = "sharpe",
        cv_splits: int = 5,
        random_state: int = 42,
    ):
        """
        Initialise l'optimiseur.

        Args:
            data: DataFrame avec les données OHLCV
            strategy_class: Classe de la stratégie à optimiser
            param_ranges: Plages de valeurs pour chaque paramètre
            metric: Métrique à optimiser ('sharpe', 'sortino', 'calmar', 'profit')
            cv_splits: Nombre de splits pour la validation croisée
            random_state: Graine aléatoire pour la reproductibilité
        """
        self.data = data
        self.strategy_class = strategy_class
        self.param_ranges = param_ranges
        self.param_names = list(param_ranges.keys())
        self.metric = metric
        self.cv_splits = cv_splits
        self.random_state = random_state

        # Validation croisée temporelle
        self.tscv = TimeSeriesSplit(n_splits=cv_splits)

        # Historique d'optimisation
        self.optimization_history = []

    def _calculate_metric(self, returns: pd.Series, positions: pd.Series) -> float:
        """
        Calcule la métrique d'évaluation.

        Args:
            returns: Série des rendements
            positions: Série des positions (1: long, -1: short, 0: neutre)

        Returns:
            float: Valeur de la métrique
        """
        # Calculer les rendements de la stratégie
        strategy_returns = returns * positions.shift(1)

        if self.metric == "sharpe":
            # Ratio de Sharpe annualisé
            if len(strategy_returns) == 0 or strategy_returns.std() == 0:
                return 0.0
            return np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()

        elif self.metric == "sortino":
            # Ratio de Sortino annualisé
            downside_returns = strategy_returns[strategy_returns < 0]
            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return 0.0
            return np.sqrt(252) * strategy_returns.mean() / downside_returns.std()

        elif self.metric == "calmar":
            # Ratio de Calmar
            cumulative_returns = (1 + strategy_returns).cumprod()
            max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
            if max_drawdown == 0:
                return 0.0
            return strategy_returns.mean() * 252 / max_drawdown

        else:  # 'profit'
            # Rendement total
            return (1 + strategy_returns).prod() - 1

    def _evaluate_params(
        self,
        params: Dict[str, float],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> Tuple[float, float]:
        """
        Évalue une combinaison de paramètres.

        Args:
            params: Paramètres à évaluer
            train_data: Données d'entraînement
            test_data: Données de test

        Returns:
            Tuple[float, float]: Score sur train et test
        """
        try:
            # Créer une instance de la stratégie avec les paramètres
            strategy = self.strategy_class(
                data_fetcher=None,  # Non utilisé pour l'optimisation
                order_executor=None,  # Non utilisé pour l'optimisation
                symbols=["BTCUSD"],  # Symbole fixe pour l'optimisation
                timeframe=None,  # Non utilisé pour l'optimisation
                params=params,
                is_optimizing=True,  # Désactive les logs pendant l'optimisation
            )

            # S'assurer que les colonnes requises sont présentes
            required_columns = ["open", "high", "low", "close", "volume"]

            # Ajouter les colonnes manquantes avec des valeurs par défaut
            for df in [train_data, test_data]:
                for col in required_columns:
                    if col not in df.columns:
                        if col == "volume":
                            df = df.copy()  # Créer une copie explicite
                            df[col] = 0  # Valeur par défaut pour le volume
                        else:
                            df = df.copy()  # Créer une copie explicite
                            df[col] = df[
                                "close"
                            ]  # Utiliser close pour les autres colonnes

            # Calculer les signaux sur train et test
            train_signals = strategy.generate_signals("BTCUSD", train_data)
            test_signals = strategy.generate_signals("BTCUSD", test_data)

            # Convertir les signaux en Series pandas si nécessaire
            if isinstance(train_signals, list):
                train_signals = pd.Series(train_signals, index=train_data.index)
            if isinstance(test_signals, list):
                test_signals = pd.Series(test_signals, index=test_data.index)

            # Calculer les rendements
            train_returns = train_data["close"].pct_change()
            test_returns = test_data["close"].pct_change()

            # Calculer les métriques
            train_score = self._calculate_metric(train_returns, train_signals)
            test_score = self._calculate_metric(test_returns, test_signals)

            return train_score, test_score

        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation des paramètres: {str(e)}")
            return float("-inf"), float("-inf")

    def grid_search(
        self, param_grid: Dict[str, List[float]], verbose: bool = True
    ) -> Dict:
        """
        Effectue une recherche sur grille des meilleurs paramètres.

        Args:
            param_grid: Grille de paramètres à tester
            verbose: Afficher les résultats intermédiaires

        Returns:
            Dict: Meilleurs paramètres et scores
        """
        best_params = None
        best_score = float("-inf")
        best_test_score = float("-inf")

        # Générer toutes les combinaisons de paramètres
        param_combinations = []
        for param_name, param_values in param_grid.items():
            param_combinations.append([(param_name, val) for val in param_values])

        from itertools import product

        all_combinations = [dict(combo) for combo in product(*param_combinations)]

        # Évaluer chaque combinaison
        for params in all_combinations:
            cv_train_scores = []
            cv_test_scores = []

            # Validation croisée temporelle
            for train_idx, test_idx in self.tscv.split(self.data):
                train_data = self.data.iloc[train_idx]
                test_data = self.data.iloc[test_idx]

                train_score, test_score = self._evaluate_params(
                    params, train_data, test_data
                )

                cv_train_scores.append(train_score)
                cv_test_scores.append(test_score)

            # Calculer les scores moyens
            mean_train_score = np.mean(cv_train_scores)
            mean_test_score = np.mean(cv_test_scores)

            # Sauvegarder l'historique
            self.optimization_history.append(
                {
                    "params": params,
                    "train_score": mean_train_score,
                    "test_score": mean_test_score,
                }
            )

            if mean_test_score > best_test_score:
                best_params = params
                best_score = mean_train_score
                best_test_score = mean_test_score

                if verbose:
                    logger.info(
                        f"Nouveaux meilleurs paramètres trouvés : {params}\n"
                        f"Score train : {mean_train_score:.4f}\n"
                        f"Score test : {mean_test_score:.4f}"
                    )

        return {
            "best_params": best_params,
            "train_score": best_score,
            "test_score": best_test_score,
            "history": self.optimization_history,
        }

    def random_search(self, n_iter: int = 100, verbose: bool = True) -> Dict:
        """
        Effectue une recherche aléatoire des meilleurs paramètres.

        Args:
            n_iter: Nombre d'itérations
            verbose: Afficher les résultats intermédiaires

        Returns:
            Dict: Meilleurs paramètres et scores
        """
        best_params = None
        best_score = float("-inf")
        best_test_score = float("-inf")

        for i in range(n_iter):
            # Générer des paramètres aléatoires
            params = {
                name: np.random.uniform(low, high)
                for name, (low, high) in self.param_ranges.items()
            }

            cv_train_scores = []
            cv_test_scores = []

            # Validation croisée temporelle
            for train_idx, test_idx in self.tscv.split(self.data):
                train_data = self.data.iloc[train_idx]
                test_data = self.data.iloc[test_idx]

                train_score, test_score = self._evaluate_params(
                    params, train_data, test_data
                )

                cv_train_scores.append(train_score)
                cv_test_scores.append(test_score)

            # Calculer les scores moyens
            mean_train_score = np.mean(cv_train_scores)
            mean_test_score = np.mean(cv_test_scores)

            # Sauvegarder l'historique
            self.optimization_history.append(
                {
                    "params": params,
                    "train_score": mean_train_score,
                    "test_score": mean_test_score,
                }
            )

            if mean_test_score > best_test_score:
                best_params = params
                best_score = mean_train_score
                best_test_score = mean_test_score

                if verbose:
                    logger.info(
                        f"Itération {i+1}/{n_iter}\n"
                        f"Nouveaux meilleurs paramètres trouvés : {params}\n"
                        f"Score train : {mean_train_score:.4f}\n"
                        f"Score test : {mean_test_score:.4f}"
                    )

        return {
            "best_params": best_params,
            "train_score": best_score,
            "test_score": best_test_score,
            "history": self.optimization_history,
        }

    def _objective(self, params):
        """Fonction objectif pour l'optimisation."""
        # Convertir les paramètres en dictionnaire
        param_dict = dict(zip(self.param_ranges.keys(), params))

        # Créer et tester la stratégie
        strategy = self.strategy_class(**param_dict)
        signals = strategy.generate_signals(self.data)

        # Calculer la performance
        returns = self.data["close"].pct_change()
        strategy_returns = returns * signals

        # Calculer le ratio de Sharpe
        if len(strategy_returns) > 0:
            sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
            return -sharpe  # Minimiser le négatif du Sharpe
        return 0

    def differential_evolution(
        self, max_iter: int = 100, popsize: int = 20, verbose: bool = False
    ) -> Dict:
        """
        Optimise les paramètres avec l'évolution différentielle.

        Args:
            max_iter: Nombre maximum d'itérations
            popsize: Taille de la population
            verbose: Afficher les logs

        Returns:
            Dict: Meilleurs paramètres et scores
        """
        bounds = [(0, 1) for _ in range(len(self.param_names))]

        result = differential_evolution(
            func=self._objective,
            bounds=bounds,
            maxiter=max_iter,
            popsize=popsize,
            disp=verbose,
        )

        best_params = dict(zip(self.param_names, result.x))
        train_score = -result.fun  # Convertir en score positif

        # Calculer le score de test
        test_score = self._evaluate_params(
            best_params,
            self.data.iloc[: int(len(self.data) * 0.8)],
            self.data.iloc[int(len(self.data) * 0.8) :],
        )[1]

        return {
            "best_params": best_params,
            "train_score": train_score,
            "test_score": test_score,
            "convergence": result.success,
        }

    def calculate_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calcule les métriques de performance."""
        if len(trades) == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
            }

        winning_trades = trades[trades["profit"] > 0]
        losing_trades = trades[trades["profit"] < 0]

        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

        total_profit = (
            winning_trades["profit"].sum() if len(winning_trades) > 0 else 0.0
        )
        total_loss = (
            abs(losing_trades["profit"].sum()) if len(losing_trades) > 0 else 0.0
        )
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        average_win = (
            winning_trades["profit"].mean() if len(winning_trades) > 0 else 0.0
        )
        average_loss = losing_trades["profit"].mean() if len(losing_trades) > 0 else 0.0

        # Calcul du drawdown
        cumulative_returns = trades["profit"].cumsum()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0

        # Calcul du ratio de Sharpe
        returns = trades["profit"].pct_change(fill_method=None).dropna()
        sharpe_ratio = (
            returns.mean() / returns.std()
            if len(returns) > 0 and returns.std() > 0
            else 0.0
        )

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": average_win,
            "average_loss": average_loss,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
        }

    def _calculate_sharpe_ratio(self, strategy_returns: np.ndarray) -> float:
        """
        Calcule le ratio de Sharpe.

        Args:
            strategy_returns: Retours de la stratégie

        Returns:
            float: Ratio de Sharpe
        """
        if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
            return 0.0

        return np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()


class MLStrategy:
    """
    Stratégie de trading basée sur l'apprentissage automatique.

    Utilise différents modèles de ML pour prédire les mouvements de prix :
    - Random Forest
    - XGBoost
    - LightGBM
    - Support Vector Machine
    """

    def __init__(
        self,
        data_fetcher=None,
        order_executor=None,
        symbols=None,
        timeframe=None,
        params=None,
        is_optimizing=False,
        model_type: str = "rf",
        feature_window: int = 20,
        prediction_window: int = 5,
        random_state: int = 42,
    ):
        """
        Initialise la stratégie ML.

        Args:
            data_fetcher: Objet pour récupérer les données
            order_executor: Objet pour exécuter les ordres
            symbols: Liste des symboles à trader
            timeframe: Timeframe des données
            params: Paramètres de la stratégie
            is_optimizing: Mode optimisation
            model_type: Type de modèle ('rf', 'xgb', 'lgb', 'svm')
            feature_window: Fenêtre pour les features
            prediction_window: Fenêtre de prédiction
            random_state: Graine aléatoire
        """
        self.data_fetcher = data_fetcher
        self.order_executor = order_executor
        self.symbols = symbols or ["BTCUSD"]
        self.timeframe = timeframe
        self.params = params or {}
        self.is_optimizing = is_optimizing

        # Paramètres ML
        self.model_type = model_type
        self.feature_window = feature_window
        self.prediction_window = prediction_window
        self.random_state = random_state

        # Initialiser le modèle
        self.model = None
        self.scaler = StandardScaler()
        self._init_model()

        logger.info(f"Stratégie ML initialisée avec le modèle {model_type}")

    def _init_model(self):
        """Initialise le modèle ML selon le type choisi."""
        if self.model_type == "rf":
            from sklearn.ensemble import RandomForestClassifier

            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=self.random_state
            )

        elif self.model_type == "xgb":
            import xgboost as xgb

            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
            )

        elif self.model_type == "lgb":
            import lightgbm as lgb

            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
            )

        else:  # 'svm'
            from sklearn.svm import SVC

            self.model = SVC(
                kernel="rbf", probability=True, random_state=self.random_state
            )

        logger.info(f"Modèle ML initialisé: {self.model_type}")

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les features pour le modèle ML.

        Args:
            data: DataFrame avec les données OHLCV

        Returns:
            DataFrame: Features pour le modèle
        """
        try:
            # Convertir numpy array en DataFrame si nécessaire
            if isinstance(data, np.ndarray):
                # Vérifier le nombre de colonnes
                if data.shape[1] == 7:
                    columns = [
                        "time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "spread",
                    ]
                else:
                    columns = ["time", "open", "high", "low", "close", "volume"]
                data = pd.DataFrame(data, columns=columns)

            # S'assurer que toutes les colonnes requises sont présentes
            required_columns = ["open", "high", "low", "close", "volume"]
            for col in required_columns:
                if col not in data.columns:
                    if col == "volume":
                        data = data.copy()
                        data[col] = 0
                    else:
                        data = data.copy()
                        data[col] = data["close"]

            # Calculer les rendements
            returns = data["close"].pct_change()

            # Moyennes mobiles
            ma_short = data["close"].rolling(window=5).mean()
            ma_medium = data["close"].rolling(window=20).mean()
            ma_long = data["close"].rolling(window=50).mean()

            # Volatilité
            volatility = returns.rolling(window=20).std()

            # RSI
            rsi = self._calculate_rsi(data["close"])

            # MACD
            macd, signal = self._calculate_macd(data["close"])

            # Créer le DataFrame de features
            features = pd.DataFrame(
                {
                    "returns": returns,
                    "ma_short": ma_short,
                    "ma_medium": ma_medium,
                    "ma_long": ma_long,
                    "volatility": volatility,
                    "rsi": rsi,
                    "macd": macd,
                    "signal": signal,
                }
            )

            # Supprimer les lignes avec des valeurs NaN
            features = features.dropna()

            return features

        except Exception as e:
            logger.error(f"Erreur lors de la création des features: {str(e)}")
            return pd.DataFrame()

    def _create_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Crée les labels pour l'entraînement.

        Args:
            data: DataFrame OHLCV

        Returns:
            np.ndarray: Labels (0 ou 1)
        """
        # Calculer les rendements futurs
        future_returns = (
            data["close"].shift(-self.prediction_window) / data["close"] - 1
        )

        # Créer les labels (1 si rendement positif, 0 sinon)
        labels = (future_returns > 0).astype(int)

        return labels

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcule le RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calcule le MACD."""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Entraîne le modèle sur les données.

        Args:
            data: DataFrame OHLCV

        Returns:
            Dict[str, float]: Métriques de performance
        """
        # Créer les features et labels
        X = self._create_features(data)
        y = self._create_labels(data)

        # Supprimer les NaN
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        if len(X) == 0 or len(y) == 0:
            logger.warning("Pas assez de données pour l'entraînement")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Diviser en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Normaliser les features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Entraîner le modèle
        self.model.fit(X_train_scaled, y_train)

        # Faire des prédictions
        y_pred = self.model.predict(X_test_scaled)

        # Calculer les métriques
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        logger.info(f"Métriques d'entraînement: {metrics}")
        return metrics

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fait des prédictions sur de nouvelles données.

        Args:
            data: DataFrame avec les données OHLCV

        Returns:
            np.ndarray: Prédictions (1: hausse, 0: baisse)
        """
        try:
            # Créer les features
            features = self._create_features(data)

            if features.empty:
                return np.array([])

            # Normaliser les features
            X = self.scaler.transform(features)

            # Faire les prédictions
            predictions = self.model.predict(X)

            return predictions

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            return np.array([])

    def generate_signals(self, symbol: str, data: pd.DataFrame) -> pd.Series:
        """
        Génère les signaux de trading.

        Args:
            symbol: Symbole à trader
            data: DataFrame avec les données OHLCV

        Returns:
            pd.Series: Signaux de trading (1: achat, -1: vente, 0: neutre)
        """
        try:
            # Faire les prédictions
            predictions = self.predict(data)

            if len(predictions) == 0:
                return pd.Series(0, index=data.index)

            # Convertir les prédictions en signaux
            signals = pd.Series(0, index=data.index)
            signals.iloc[-len(predictions) :] = np.where(predictions == 1, 1, -1)

            return signals

        except Exception as e:
            logger.error(f"Erreur lors de la génération des signaux: {str(e)}")
            return pd.Series(0, index=data.index)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retourne l'importance des features.

        Returns:
            Dict[str, float]: Importance des features
        """
        if not hasattr(self.model, "feature_importances_"):
            return {}

        features = [
            "returns",
            "ma_short",
            "ma_medium",
            "ma_long",
            "volatility",
            "rsi",
            "macd",
            "signal",
        ]

        importance = dict(zip(features, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
