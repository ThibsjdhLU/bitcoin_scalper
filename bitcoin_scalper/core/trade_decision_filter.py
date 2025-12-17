"""
Module de filtrage des décisions de trade basé sur la probabilité calibrée et l'entropie de Shannon.
Permet d'appliquer un seuil dynamique et de refuser les trades en zone d'incertitude ou de confusion.
Conforme aux standards PEP8, sécurité, et documentation automatique.
"""
from typing import Tuple, Optional, List, Union
import numpy as np
import logging

class TradeDecisionFilter:
    """
    Filtre les décisions de trade selon la probabilité calibrée et un seuil dynamique.
    Intègre un filtre d'entropie de Shannon pour détecter la confusion du modèle.
    Refuse les trades en zone d'incertitude.
    """
    def __init__(self, lower_bound: float = 0.45, upper_bound: float = 0.55, dynamic: bool = False, window_size: int = 100, max_entropy: float = 0.8):
        """
        Initialise le filtre.
        Args:
            lower_bound (float): Borne inférieure de la zone d'incertitude.
            upper_bound (float): Borne supérieure de la zone d'incertitude.
            dynamic (bool): Si True, adapte les bornes selon la distribution récente.
            window_size (int): Taille de la fenêtre pour le seuil dynamique.
            max_entropy (float): Seuil maximal d'entropie accepté (0.0 à 1.0+).
                                 Pour 2 classes, max est 1.0. 0.8 est un bon seuil par défaut.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dynamic = dynamic
        self.window_size = window_size
        self.max_entropy = max_entropy
        self.recent_probas: List[float] = []
        self.logger = logging.getLogger("TradeDecisionFilter")

    def update_dynamic_bounds(self):
        """
        Met à jour dynamiquement les bornes selon la distribution récente des probabilités.
        """
        if len(self.recent_probas) >= self.window_size:
            arr = np.array(self.recent_probas[-self.window_size:])
            self.lower_bound = float(np.quantile(arr, 0.05))
            self.upper_bound = float(np.quantile(arr, 0.95))

    def calculate_entropy(self, probs: Union[np.ndarray, List[float]]) -> float:
        """
        Calcule l'entropie de Shannon de la distribution de probabilités.
        H(X) = - sum(p * log2(p))
        """
        probs = np.array(probs, dtype=float)
        # Éviter log(0)
        probs = np.clip(probs, 1e-9, 1.0)
        # Normalisation si nécessaire
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def filter(self, proba: float, probs: Optional[Union[np.ndarray, List[float]]] = None) -> Tuple[bool, str]:
        """
        Applique le filtre sur la probabilité calibrée et l'entropie (si fournie).
        Args:
            proba (float): Probabilité calibrée du trade (classe dominante).
            probs (array, optional): Distribution complète des probabilités pour calcul d'entropie.
        Returns:
            (bool, str): (True si trade accepté, False sinon, raison)
        """
        # 1. Mise à jour dynamique
        self.recent_probas.append(proba)
        if self.dynamic:
            self.update_dynamic_bounds()

        # 2. Filtre d'incertitude (seuil de proba)
        if self.lower_bound < proba < self.upper_bound:
            self.logger.info(f"Trade refusé : proba={proba:.3f} en zone d'incertitude [{self.lower_bound:.2f}, {self.upper_bound:.2f}]")
            return False, f"Refusé : proba en zone d'incertitude [{self.lower_bound:.2f}, {self.upper_bound:.2f}]"

        # 3. Filtre d'entropie (si probs fourni)
        if probs is not None:
            entropy = self.calculate_entropy(probs)
            if entropy > self.max_entropy:
                self.logger.info(f"Trade refusé : Entropie trop élevée ({entropy:.2f} > {self.max_entropy})")
                return False, f"Refusé : Modèle confus (Entropie {entropy:.2f} > {self.max_entropy})"

        self.logger.info(f"Trade accepté : proba={proba:.3f} hors zone d'incertitude.")
        return True, "Accepté"
