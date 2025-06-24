"""
Module de filtrage des décisions de trade basé sur la probabilité calibrée.
Permet d'appliquer un seuil dynamique et de refuser les trades en zone d'incertitude.
Conforme aux standards PEP8, sécurité, et documentation automatique.
"""
from typing import Tuple, Optional, List
import numpy as np
import logging

class TradeDecisionFilter:
    """
    Filtre les décisions de trade selon la probabilité calibrée et un seuil dynamique.
    Refuse les trades en zone d'incertitude.
    """
    def __init__(self, lower_bound: float = 0.45, upper_bound: float = 0.55, dynamic: bool = False, window_size: int = 100):
        """
        Initialise le filtre.
        Args:
            lower_bound (float): Borne inférieure de la zone d'incertitude.
            upper_bound (float): Borne supérieure de la zone d'incertitude.
            dynamic (bool): Si True, adapte les bornes selon la distribution récente.
            window_size (int): Taille de la fenêtre pour le seuil dynamique.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dynamic = dynamic
        self.window_size = window_size
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

    def filter(self, proba: float) -> Tuple[bool, str]:
        """
        Applique le filtre sur la probabilité calibrée.
        Args:
            proba (float): Probabilité calibrée du trade.
        Returns:
            (bool, str): (True si trade accepté, False sinon, raison)
        """
        self.recent_probas.append(proba)
        if self.dynamic:
            self.update_dynamic_bounds()
        if self.lower_bound < proba < self.upper_bound:
            self.logger.info(f"Trade refusé : proba={proba:.3f} en zone d'incertitude [{self.lower_bound:.2f}, {self.upper_bound:.2f}]")
            return False, f"Refusé : proba en zone d'incertitude [{self.lower_bound:.2f}, {self.upper_bound:.2f}]"
        self.logger.info(f"Trade accepté : proba={proba:.3f} hors zone d'incertitude.")
        return True, "Accepté" 