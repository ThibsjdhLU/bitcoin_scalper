"""
Module pour la gestion du risque et le rééquilibrage cross-asset
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from arch import arch_model

class DynamicCorrelation:
    """
    Gestionnaire de corrélation dynamique entre les actifs.
    """
    
    def __init__(self, window: int = 60):
        """
        Initialise le gestionnaire de corrélation.
        
        Args:
            window (int): Fenêtre de calcul de la corrélation
        """
        self.window = window
        self.correlations = None
    
    def calculate_correlations(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule la matrice de corrélation entre les actifs.
        
        Args:
            prices (pd.DataFrame): Prix des actifs
            
        Returns:
            pd.DataFrame: Matrice de corrélation
        """
        returns = prices.pct_change()
        self.correlations = returns.rolling(window=self.window).corr()
        return self.correlations
    
    def check_correlations(self, positions: Optional[List[Dict]] = None) -> bool:
        """
        Vérifie si les corrélations sont dans des limites acceptables.
        
        Args:
            positions (List[Dict], optional): Positions ouvertes
            
        Returns:
            bool: True si le risque de corrélation est trop élevé
        """
        if positions is None or len(positions) < 2:
            return False
            
        if self.correlations is None:
            return False
            
        # Vérifie les corrélations entre les positions ouvertes
        symbols = [pos['symbol'] for pos in positions]
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                if abs(self.correlations.iloc[-1, i, j]) > 0.8:  # Seuil de corrélation
                    return True
        
        return False

class KellyCriterion:
    """
    Calculateur de taille de position basé sur le critère de Kelly.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialise le calculateur de Kelly.
        
        Args:
            risk_free_rate (float): Taux sans risque annuel
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_position_size(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Calcule la fraction optimale du capital à risquer.
        
        Args:
            win_rate (float): Taux de réussite historique
            win_loss_ratio (float): Ratio gains/pertes moyen
            
        Returns:
            float: Fraction du capital à risquer
        """
        # Formule du critère de Kelly
        q = 1 - win_rate
        kelly_fraction = (win_rate * win_loss_ratio - q) / win_loss_ratio
        
        # Limite la fraction pour plus de sécurité
        return max(0, min(kelly_fraction, 0.2))  # Maximum 20% du capital

class PortfolioRebalancer:
    """
    Gestionnaire de rééquilibrage du portefeuille.
    """
    
    def __init__(self, target_volatility: float = 0.15, rebalance_threshold: float = 0.1):
        """
        Initialise le gestionnaire de rééquilibrage.
        
        Args:
            target_volatility (float): Volatilité cible du portefeuille
            rebalance_threshold (float): Seuil de déclenchement du rééquilibrage
        """
        self.target_volatility = target_volatility
        self.rebalance_threshold = rebalance_threshold
        self.current_weights = None
    
    def calculate_optimal_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calcule les poids optimaux pour le portefeuille.
        
        Args:
            returns (pd.DataFrame): Rendements historiques
            
        Returns:
            Dict[str, float]: Poids optimaux par actif
        """
        # Calcul de la matrice de covariance
        cov_matrix = returns.cov()
        
        # Calcul des rendements moyens
        mean_returns = returns.mean()
        
        # Optimisation simple basée sur la volatilité cible
        weights = {}
        total_risk = 0
        
        for asset in returns.columns:
            weight = self.target_volatility / (returns[asset].std() * np.sqrt(252))
            weights[asset] = min(weight, 0.5)  # Maximum 50% par actif
            total_risk += weights[asset]
        
        # Normalisation des poids
        if total_risk > 0:
            for asset in weights:
                weights[asset] /= total_risk
        
        self.current_weights = weights
        return weights
    
    def should_rebalance(self, current_positions: Optional[List[Dict]] = None) -> bool:
        """
        Détermine si un rééquilibrage est nécessaire.
        
        Args:
            current_positions (List[Dict], optional): Positions actuelles
            
        Returns:
            bool: True si un rééquilibrage est nécessaire
        """
        if self.current_weights is None or current_positions is None:
            return False
            
        # Calcul des poids actuels
        total_value = sum(pos['value'] for pos in current_positions)
        current_weights = {
            pos['symbol']: pos['value'] / total_value
            for pos in current_positions
        }
        
        # Vérifie si les déviations dépassent le seuil
        for symbol, target_weight in self.current_weights.items():
            if symbol in current_weights:
                deviation = abs(current_weights[symbol] - target_weight)
                if deviation > self.rebalance_threshold:
                    return True
        
        return False 