"""
Générateur de scénarios de marché hostiles pour l'adversarial testing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class MarketGenerator:
    """
    Classe pour générer des scénarios de marché hostiles
    """
    
    def __init__(self, base_data=None):
        """
        Initialise le générateur de marché
        
        Args:
            base_data (pd.DataFrame): Données de base pour la génération
        """
        self.base_data = base_data
        
    def generate_slippage_scenario(self, slippage_percent=0.5, duration_minutes=30):
        """
        Génère un scénario avec slippage extrême
        
        Args:
            slippage_percent (float): Pourcentage de slippage
            duration_minutes (int): Durée du slippage en minutes
            
        Returns:
            pd.DataFrame: Données avec slippage
        """
        if self.base_data is None:
            raise ValueError("Données de base non fournies")
            
        data = self.base_data.copy()
        
        # Trouver un point aléatoire pour commencer le slippage
        start_idx = np.random.randint(0, len(data) - duration_minutes)
        
        # Appliquer le slippage
        for i in range(start_idx, min(start_idx + duration_minutes, len(data))):
            # Slippage aléatoire positif ou négatif
            direction = np.random.choice([-1, 1])
            slippage = data.iloc[i]['close'] * (slippage_percent / 100) * direction
            
            # Mettre à jour les prix
            data.iloc[i, data.columns.get_loc('close')] += slippage
            data.iloc[i, data.columns.get_loc('high')] = max(data.iloc[i]['high'], data.iloc[i]['close'])
            data.iloc[i, data.columns.get_loc('low')] = min(data.iloc[i]['low'], data.iloc[i]['close'])
            
        return data
        
    def generate_gap_scenario(self, gap_percent=1.0, gap_direction='up'):
        """
        Génère un scénario avec gap de prix
        
        Args:
            gap_percent (float): Pourcentage du gap
            gap_direction (str): Direction du gap ('up' ou 'down')
            
        Returns:
            pd.DataFrame: Données avec gap
        """
        if self.base_data is None:
            raise ValueError("Données de base non fournies")
            
        data = self.base_data.copy()
        
        # Trouver un point aléatoire pour le gap
        gap_idx = np.random.randint(0, len(data) - 1)
        
        # Calculer le gap
        gap = data.iloc[gap_idx]['close'] * (gap_percent / 100)
        
        # Appliquer le gap
        if gap_direction == 'up':
            data.iloc[gap_idx+1:, data.columns.get_loc('open')] += gap
            data.iloc[gap_idx+1:, data.columns.get_loc('high')] += gap
            data.iloc[gap_idx+1:, data.columns.get_loc('low')] += gap
            data.iloc[gap_idx+1:, data.columns.get_loc('close')] += gap
        else:
            data.iloc[gap_idx+1:, data.columns.get_loc('open')] -= gap
            data.iloc[gap_idx+1:, data.columns.get_loc('high')] -= gap
            data.iloc[gap_idx+1:, data.columns.get_loc('low')] -= gap
            data.iloc[gap_idx+1:, data.columns.get_loc('close')] -= gap
            
        return data
        
    def generate_volatility_spike(self, volatility_multiplier=3.0, duration_minutes=60):
        """
        Génère un scénario avec pic de volatilité
        
        Args:
            volatility_multiplier (float): Multiplicateur de volatilité
            duration_minutes (int): Durée du pic en minutes
            
        Returns:
            pd.DataFrame: Données avec pic de volatilité
        """
        if self.base_data is None:
            raise ValueError("Données de base non fournies")
            
        data = self.base_data.copy()
        
        # Trouver un point aléatoire pour commencer le pic
        start_idx = np.random.randint(0, len(data) - duration_minutes)
        
        # Calculer la volatilité de base
        base_volatility = data['close'].pct_change().std()
        
        # Appliquer le pic de volatilité
        for i in range(start_idx, min(start_idx + duration_minutes, len(data))):
            # Générer un mouvement aléatoire avec volatilité augmentée
            direction = np.random.choice([-1, 1])
            movement = direction * base_volatility * volatility_multiplier * data.iloc[i]['close']
            
            # Mettre à jour les prix
            data.iloc[i, data.columns.get_loc('close')] += movement
            data.iloc[i, data.columns.get_loc('high')] = max(data.iloc[i]['high'], data.iloc[i]['close'])
            data.iloc[i, data.columns.get_loc('low')] = min(data.iloc[i]['low'], data.iloc[i]['close'])
            
        return data
        
    def generate_liquidity_crisis(self, volume_reduction=0.7, duration_minutes=120):
        """
        Génère un scénario de crise de liquidité
        
        Args:
            volume_reduction (float): Réduction du volume (0-1)
            duration_minutes (int): Durée de la crise en minutes
            
        Returns:
            pd.DataFrame: Données avec crise de liquidité
        """
        if self.base_data is None:
            raise ValueError("Données de base non fournies")
            
        data = self.base_data.copy()
        
        # Trouver un point aléatoire pour commencer la crise
        start_idx = np.random.randint(0, len(data) - duration_minutes)
        
        # Appliquer la réduction de volume
        for i in range(start_idx, min(start_idx + duration_minutes, len(data))):
            data.iloc[i, data.columns.get_loc('volume')] *= (1 - volume_reduction)
            
        return data 