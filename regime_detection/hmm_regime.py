"""
Module de détection des régimes de marché utilisant les modèles de Markov cachés (HMM)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from hmmlearn import hmm

class HMMRegimeDetector:
    """
    Classe pour la détection des régimes de marché utilisant les HMM
    """
    
    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        """
        Initialise le détecteur de régimes HMM
        
        Args:
            n_regimes (int): Nombre de régimes à détecter
            random_state (int): État aléatoire pour la reproductibilité
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            random_state=random_state,
            covariance_type="full"
        )
        
    def fit(self, data: pd.DataFrame) -> None:
        """
        Entraîne le modèle HMM sur les données de prix
        
        Args:
            data (pd.DataFrame): Données de prix
        """
        features = self._prepare_features(data)
        self.model.fit(features)
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prédit les régimes pour les données données
        
        Args:
            data (pd.DataFrame): Données de prix
            
        Returns:
            np.ndarray: Régimes prédits
        """
        features = self._prepare_features(data)
        return self.model.predict(features)
        
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calcule les probabilités de chaque régime
        
        Args:
            data (pd.DataFrame): Données de prix
            
        Returns:
            np.ndarray: Probabilités des régimes
        """
        features = self._prepare_features(data)
        return self.model.predict_proba(features)
        
    def get_regime_parameters(self) -> Dict:
        """
        Récupère les paramètres des régimes détectés
        
        Returns:
            dict: Paramètres des régimes
        """
        return {
            'means': self.model.means_,
            'covars': self.model.covars_,
            'transition_matrix': self.model.transmat_,
            'startprob': self.model.startprob_
        }
        
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prépare les caractéristiques pour l'entraînement
        
        Args:
            data (pd.DataFrame): Données de prix
            
        Returns:
            np.ndarray: Caractéristiques préparées
        """
        returns = np.log(data['close'] / data['close'].shift(1)).dropna()
        volatility = returns.rolling(window=20).std().dropna()
        momentum = returns.rolling(window=10).mean().dropna()
        
        features = np.column_stack([
            returns.values.reshape(-1, 1),
            volatility.values.reshape(-1, 1),
            momentum.values.reshape(-1, 1)
        ])
        
        return features 