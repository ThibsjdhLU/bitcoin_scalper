"""
Module de calibration des probabilités pour modèles de classification.
Propose Platt scaling (régression logistique) et isotonic regression.
Conforme aux standards PEP8, sécurité, et documentation automatique.
"""
from typing import Optional, Union
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import numpy as np

class ProbabilityCalibrator:
    """
    Classe pour calibrer les probabilités d'un modèle de classification.
    Permet d'utiliser Platt scaling (sigmoid) ou isotonic regression.
    """
    def __init__(self, method: str = "sigmoid"):
        """
        Initialise le calibrateur.
        Args:
            method (str): 'sigmoid' (Platt scaling) ou 'isotonic'.
        """
        if method not in ["sigmoid", "isotonic"]:
            raise ValueError("method doit être 'sigmoid' ou 'isotonic'")
        self.method = method
        self.calibrator: Optional[CalibratedClassifierCV] = None
        self.is_fitted = False

    def fit(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entraîne le calibrateur sur les données fournies.
        Args:
            model: Modèle de classification déjà entraîné.
            X: Features.
            y: Labels.
        """
        self.calibrator = CalibratedClassifierCV(base_estimator=model, method=self.method, cv="prefit")
        self.calibrator.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les probabilités calibrées.
        Args:
            X: Features.
        Returns:
            np.ndarray: Probabilités calibrées.
        """
        if not self.is_fitted or self.calibrator is None:
            raise RuntimeError("Le calibrateur doit être entraîné avant la prédiction.")
        return self.calibrator.predict_proba(X)

    def save(self, path: str) -> None:
        """
        Sauvegarde le calibrateur sur disque.
        Args:
            path: Chemin du fichier.
        """
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ProbabilityCalibrator":
        """
        Charge un calibrateur depuis le disque.
        Args:
            path: Chemin du fichier.
        Returns:
            ProbabilityCalibrator
        """
        return joblib.load(path) 