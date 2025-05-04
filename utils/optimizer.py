"""
Module d'optimisation des paramètres et d'apprentissage automatique.
"""
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.optimize import differential_evolution
from loguru import logger

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
        metric: str = 'sharpe',
        cv_splits: int = 5,
        random_state: int = 42
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
        self.metric = metric
        self.cv_splits = cv_splits
        self.random_state = random_state
        
        # Validation croisée temporelle
        self.tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        # Historique d'optimisation
        self.optimization_history = []
        
    def _calculate_metric(
        self,
        returns: pd.Series,
        positions: pd.Series
    ) -> float:
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
        
        if self.metric == 'sharpe':
            # Ratio de Sharpe annualisé
            return np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
            
        elif self.metric == 'sortino':
            # Ratio de Sortino annualisé
            downside_returns = strategy_returns[strategy_returns < 0]
            return np.sqrt(252) * strategy_returns.mean() / downside_returns.std()
            
        elif self.metric == 'calmar':
            # Ratio de Calmar
            cumulative_returns = (1 + strategy_returns).cumprod()
            max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
            return strategy_returns.mean() * 252 / max_drawdown
            
        else:  # 'profit'
            # Rendement total
            return (1 + strategy_returns).prod() - 1
            
    def _evaluate_params(
        self,
        params: Dict[str, float],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Évalue un ensemble de paramètres sur les données.
        
        Args:
            params: Dictionnaire des paramètres
            train_data: Données d'entraînement
            test_data: Données de test
            
        Returns:
            Tuple[float, float]: Score sur train et test
        """
        # Instancier la stratégie avec les paramètres
        strategy = self.strategy_class(**params)
        
        # Calculer les signaux sur train et test
        train_signals = strategy.generate_signals(train_data)
        test_signals = strategy.generate_signals(test_data)
        
        # Calculer les rendements
        train_returns = train_data['close'].pct_change()
        test_returns = test_data['close'].pct_change()
        
        # Calculer les métriques
        train_score = self._calculate_metric(train_returns, train_signals)
        test_score = self._calculate_metric(test_returns, test_signals)
        
        return train_score, test_score
        
    def grid_search(
        self,
        param_grid: Dict[str, List[float]],
        verbose: bool = True
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
        best_score = float('-inf')
        best_test_score = float('-inf')
        
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
                    params,
                    train_data,
                    test_data
                )
                
                cv_train_scores.append(train_score)
                cv_test_scores.append(test_score)
                
            # Calculer les scores moyens
            mean_train_score = np.mean(cv_train_scores)
            mean_test_score = np.mean(cv_test_scores)
            
            # Sauvegarder l'historique
            self.optimization_history.append({
                'params': params,
                'train_score': mean_train_score,
                'test_score': mean_test_score
            })
            
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
            'best_params': best_params,
            'train_score': best_score,
            'test_score': best_test_score,
            'history': self.optimization_history
        }
        
    def random_search(
        self,
        n_iter: int = 100,
        verbose: bool = True
    ) -> Dict:
        """
        Effectue une recherche aléatoire des meilleurs paramètres.
        
        Args:
            n_iter: Nombre d'itérations
            verbose: Afficher les résultats intermédiaires
            
        Returns:
            Dict: Meilleurs paramètres et scores
        """
        best_params = None
        best_score = float('-inf')
        best_test_score = float('-inf')
        
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
                    params,
                    train_data,
                    test_data
                )
                
                cv_train_scores.append(train_score)
                cv_test_scores.append(test_score)
                
            # Calculer les scores moyens
            mean_train_score = np.mean(cv_train_scores)
            mean_test_score = np.mean(cv_test_scores)
            
            # Sauvegarder l'historique
            self.optimization_history.append({
                'params': params,
                'train_score': mean_train_score,
                'test_score': mean_test_score
            })
            
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
            'best_params': best_params,
            'train_score': best_score,
            'test_score': best_test_score,
            'history': self.optimization_history
        }
        
    def differential_evolution(
        self,
        max_iter: int = 100,
        popsize: int = 15,
        verbose: bool = True
    ) -> Dict:
        """
        Optimise les paramètres avec l'algorithme d'évolution différentielle.
        
        Args:
            max_iter: Nombre maximum d'itérations
            popsize: Taille de la population
            verbose: Afficher les résultats intermédiaires
            
        Returns:
            Dict: Meilleurs paramètres et scores
        """
        # Préparer les bornes pour l'optimisation
        bounds = [
            (low, high) for _, (low, high) in self.param_ranges.items()
        ]
        param_names = list(self.param_ranges.keys())
        
        def objective(x):
            # Convertir le vecteur en dictionnaire de paramètres
            params = dict(zip(param_names, x))
            
            cv_scores = []
            # Validation croisée temporelle
            for train_idx, test_idx in self.tscv.split(self.data):
                train_data = self.data.iloc[train_idx]
                test_data = self.data.iloc[test_idx]
                
                train_score, test_score = self._evaluate_params(
                    params,
                    train_data,
                    test_data
                )
                cv_scores.append(test_score)
                
            # Retourner le négatif du score moyen (minimisation)
            return -np.mean(cv_scores)
            
        # Lancer l'optimisation
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iter,
            popsize=popsize,
            seed=self.random_state,
            disp=verbose
        )
        
        # Convertir le résultat en dictionnaire
        best_params = dict(zip(param_names, result.x))
        
        # Évaluer les meilleurs paramètres
        cv_train_scores = []
        cv_test_scores = []
        
        for train_idx, test_idx in self.tscv.split(self.data):
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]
            
            train_score, test_score = self._evaluate_params(
                best_params,
                train_data,
                test_data
            )
            
            cv_train_scores.append(train_score)
            cv_test_scores.append(test_score)
            
        return {
            'best_params': best_params,
            'train_score': np.mean(cv_train_scores),
            'test_score': np.mean(cv_test_scores),
            'convergence': result.convergence
        }
        
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
        model_type: str = 'rf',
        feature_window: int = 20,
        prediction_window: int = 5,
        random_state: int = 42
    ):
        """
        Initialise la stratégie ML.
        
        Args:
            model_type: Type de modèle ('rf', 'xgb', 'lgb', 'svm')
            feature_window: Fenêtre pour le calcul des features
            prediction_window: Fenêtre de prédiction
            random_state: Graine aléatoire
        """
        self.model_type = model_type
        self.feature_window = feature_window
        self.prediction_window = prediction_window
        self.random_state = random_state
        
        # Initialiser le modèle
        self._init_model()
        
        # Scaler pour normaliser les features
        self.scaler = StandardScaler()
        
    def _init_model(self):
        """Initialise le modèle ML selon le type choisi."""
        if self.model_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            )
            
        elif self.model_type == 'xgb':
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            )
            
        elif self.model_type == 'lgb':
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            )
            
        else:  # 'svm'
            from sklearn.svm import SVC
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            )
            
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les features pour le modèle ML.
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            pd.DataFrame: Features calculées
        """
        df = pd.DataFrame()
        
        # Features de prix
        df['returns'] = data['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Moyennes mobiles
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = data['close'].rolling(period).mean()
            df[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            
        # Volatilité
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            
        # Volume
        df['volume_ma5'] = data['volume'].rolling(5).mean()
        df['volume_ma20'] = data['volume'].rolling(20).mean()
        
        # Momentum
        df['rsi'] = self._calculate_rsi(data['close'])
        df['macd'], _, _ = self._calculate_macd(data['close'])
        
        # Supprimer les NaN
        df = df.dropna()
        
        return df
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcule le RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcule le MACD."""
        fast_ema = prices.ewm(span=fast).mean()
        slow_ema = prices.ewm(span=slow).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
        
    def _create_labels(self, data: pd.DataFrame) -> pd.Series:
        """
        Crée les labels pour l'apprentissage supervisé.
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            pd.Series: Labels (1: hausse, 0: baisse)
        """
        future_returns = data['close'].shift(-self.prediction_window).pct_change(self.prediction_window)
        return (future_returns > 0).astype(int)
        
    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Entraîne le modèle ML.
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Dict[str, float]: Métriques d'évaluation
        """
        # Créer les features et labels
        X = self._create_features(data)
        y = self._create_labels(data)
        
        # Supprimer les dernières lignes sans label
        X = X[:-self.prediction_window]
        y = y[:-self.prediction_window]
        
        # Split train/test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Normaliser les features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Entraîner le modèle
        self.model.fit(X_train, y_train)
        
        # Évaluer le modèle
        y_pred = self.model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Génère des prédictions.
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            pd.Series: Prédictions (1: long, -1: short, 0: neutre)
        """
        # Créer les features
        X = self._create_features(data)
        
        # Normaliser les features
        X = self.scaler.transform(X)
        
        # Prédire les probabilités
        proba = self.model.predict_proba(X)
        
        # Convertir en signaux (-1, 0, 1)
        signals = pd.Series(0, index=data.index)
        signals.iloc[-len(X):] = np.where(
            proba[:, 1] > 0.6, 1,  # Probabilité de hausse > 60%
            np.where(proba[:, 0] > 0.6, -1, 0)  # Probabilité de baisse > 60%
        )
        
        return signals 