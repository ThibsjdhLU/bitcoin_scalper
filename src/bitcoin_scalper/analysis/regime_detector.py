"""
Module principal de détection des régimes de marché
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .gmm_regime import GMMRegimeDetector
from .hmm_regime import HMMRegimeDetector


class RegimeDetector:
    """
    Classe principale pour la détection des régimes de marché
    """

    def __init__(self, method: str = "hmm", **kwargs):
        """
        Initialise le détecteur de régimes

        Args:
            method (str): Méthode de détection ('hmm' ou 'gmm')
            **kwargs: Arguments supplémentaires pour les détecteurs
        """
        self.method = method.lower()
        if self.method == "hmm":
            self.detector = HMMRegimeDetector(**kwargs)
        elif self.method == "gmm":
            self.detector = GMMRegimeDetector(**kwargs)
        else:
            raise ValueError(f"Méthode de détection non supportée: {method}")

    def fit(self, data: pd.DataFrame) -> None:
        """
        Entraîne le modèle de détection

        Args:
            data (pd.DataFrame): Données de prix
        """
        self.detector.fit(data)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prédit les régimes pour les données données

        Args:
            data (pd.DataFrame): Données de prix

        Returns:
            np.ndarray: Régimes prédits
        """
        return self.detector.predict(data)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calcule les probabilités de chaque régime

        Args:
            data (pd.DataFrame): Données de prix

        Returns:
            np.ndarray: Probabilités des régimes
        """
        return self.detector.predict_proba(data)

    def get_regime_parameters(self) -> Dict:
        """
        Récupère les paramètres des régimes détectés

        Returns:
            dict: Paramètres des régimes
        """
        return self.detector.get_regime_parameters()

    def analyze_regime(self, data: pd.DataFrame) -> Dict:
        """
        Analyse les caractéristiques du régime actuel

        Args:
            data (pd.DataFrame): Données de prix

        Returns:
            dict: Caractéristiques du régime
        """
        regime = self.predict(data)[-1]
        proba = self.predict_proba(data)[-1]

        return {
            "current_regime": regime,
            "regime_probability": proba[regime],
            "all_probabilities": proba,
            "parameters": self.get_regime_parameters(),
        }

    def get_current_regime(self, data: pd.DataFrame) -> Tuple[int, float]:
        """
        Détermine le régime actuel et sa probabilité

        Args:
            data (pd.DataFrame): Données de prix récentes

        Returns:
            Tuple[int, float]: (régime actuel, probabilité)
        """
        probas = self.predict_proba(data)
        current_regime = np.argmax(probas[-1])
        current_prob = probas[-1][current_regime]
        return current_regime, current_prob

    def get_regime_name(self, regime: int) -> str:
        """
        Obtient le nom du régime à partir de son index

        Args:
            regime (int): Index du régime

        Returns:
            str: Nom du régime
        """
        return self.detector.regime_names[regime]

    def analyze_regime_stability(self, data: pd.DataFrame, window: int = 20) -> Dict:
        """
        Analyse la stabilité des régimes sur une fenêtre glissante

        Args:
            data (pd.DataFrame): Données de prix
            window (int): Taille de la fenêtre glissante

        Returns:
            dict: Statistiques de stabilité des régimes
        """
        regimes = self.predict(data)
        stability = {}

        for regime in range(self.detector.n_regimes):
            regime_mask = regimes == regime
            regime_changes = np.diff(regime_mask.astype(int))
            stability[self.get_regime_name(regime)] = {
                "frequency": np.mean(regime_mask),
                "avg_duration": np.mean(np.diff(np.where(regime_changes)[0]))
                if len(np.where(regime_changes)[0]) > 1
                else 0,
                "volatility": np.std(regime_changes) if len(regime_changes) > 0 else 0,
            }

        return stability
