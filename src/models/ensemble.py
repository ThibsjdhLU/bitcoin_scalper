"""
Module pour le meta-labeling et le stacking des modèles
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Callable, Optional, Tuple
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class MetaLabeler:
    """
    Méta-étiqueteur pour la classification des signaux de trading.
    """
    
    def __init__(self, base_threshold: float = 0.5):
        """
        Initialise le méta-étiqueteur.
        
        Args:
            base_threshold (float): Seuil de base pour la classification
        """
        self.base_threshold = base_threshold
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prépare les features pour le modèle.
        
        Args:
            data (pd.DataFrame): Données brutes
            
        Returns:
            np.ndarray: Features normalisées
        """
        # Calcul des indicateurs techniques de base
        features = pd.DataFrame()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        sma = data['close'].rolling(window=20).mean()
        std = data['close'].rolling(window=20).std()
        features['bb_upper'] = sma + (std * 2)
        features['bb_lower'] = sma - (std * 2)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma
        
        # Normalisation
        return self.scaler.fit_transform(features.fillna(0))
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Entraîne le méta-étiqueteur.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
        """
        self.classifier.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les labels pour de nouvelles données.
        
        Args:
            X (np.ndarray): Features
            
        Returns:
            np.ndarray: Labels prédits
        """
        return self.classifier.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les probabilités pour chaque classe.
        
        Args:
            X (np.ndarray): Features
            
        Returns:
            np.ndarray: Probabilités prédites
        """
        return self.classifier.predict_proba(X)

class ModelStacker:
    """
    Stacking de modèles pour la prédiction des mouvements de prix.
    """
    
    def __init__(self, n_models: int = 3):
        """
        Initialise le stacker de modèles.
        
        Args:
            n_models (int): Nombre de modèles de base
        """
        self.n_models = n_models
        self.base_models = [
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42 + i
            )
            for i in range(n_models)
        ]
        self.meta_model = GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare les features pour l'entraînement.
        
        Args:
            data (pd.DataFrame): Données brutes
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features et target
        """
        # Features techniques
        features = pd.DataFrame()
        
        # Momentum
        features['returns'] = data['close'].pct_change()
        features['momentum'] = features['returns'].rolling(window=10).mean()
        
        # Volatilité
        features['volatility'] = features['returns'].rolling(window=20).std()
        
        # Volume relatif
        features['rel_volume'] = data['volume'] / data['volume'].rolling(window=20).mean()
        
        # Target : rendement futur
        target = features['returns'].shift(-1)
        
        # Nettoyage et normalisation
        features = features.dropna()
        target = target.dropna()
        
        X = self.scaler.fit_transform(features)
        y = target.values
        
        return X, y
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Entraîne le stacker.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target
        """
        # Entraînement des modèles de base
        base_predictions = np.zeros((X.shape[0], self.n_models))
        
        for i, model in enumerate(self.base_models):
            model.fit(X, y)
            base_predictions[:, i] = model.predict(X)
        
        # Entraînement du méta-modèle
        self.meta_model.fit(base_predictions, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fait des prédictions avec le stacker.
        
        Args:
            X (np.ndarray): Features
            
        Returns:
            np.ndarray: Prédictions
        """
        # Prédictions des modèles de base
        base_predictions = np.zeros((X.shape[0], self.n_models))
        
        for i, model in enumerate(self.base_models):
            base_predictions[:, i] = model.predict(X)
        
        # Prédiction finale avec le méta-modèle
        return self.meta_model.predict(base_predictions)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les probabilités des labels
        
        Args:
            X (np.ndarray): Features
            
        Returns:
            np.ndarray: Probabilités des prédictions
        """
        predictions = np.zeros((len(X), len(self.models)))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict_proba(X)[:, 1]
            
        return self.final_model.predict_proba(predictions)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Obtient l'importance des features du modèle final
        
        Returns:
            Dict[str, float]: Importance des features
        """
        if hasattr(self.final_model, 'feature_importances_'):
            return {
                f'model_{i}': importance 
                for i, importance in enumerate(self.final_model.feature_importances_)
            }
        return {} 