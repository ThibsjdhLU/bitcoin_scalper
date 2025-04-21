"""
Module pour le calcul des Bandes de Bollinger
"""

import numpy as np
import pandas as pd
from typing import List, Union, Dict, Tuple

class BollingerBands:
    """Classe pour le calcul des Bandes de Bollinger"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialise le calculateur des Bandes de Bollinger
        
        Args:
            period (int): Période pour la moyenne mobile
            std_dev (float): Nombre d'écarts-types pour les bandes
        """
        self.period = period
        self.std_dev = std_dev
        
    def calculate(self, data: Union[pd.DataFrame, List[Dict], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule les Bandes de Bollinger pour les données fournies
        
        Args:
            data: Données de prix (doit contenir les prix de clôture)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (Bande supérieure, Moyenne mobile, Bande inférieure)
        """
        try:
            # Conversion des données en DataFrame si nécessaire
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, np.ndarray):
                df = pd.DataFrame(data, columns=['close'])
            else:
                df = data.copy()
            
            # Calcul de la moyenne mobile
            middle_band = df['close'].rolling(window=self.period).mean()
            
            # Calcul de l'écart-type
            rolling_std = df['close'].rolling(window=self.period).std()
            
            # Calcul des bandes
            upper_band = middle_band + (rolling_std * self.std_dev)
            lower_band = middle_band - (rolling_std * self.std_dev)
            
            return upper_band.values, middle_band.values, lower_band.values
            
        except Exception as e:
            print(f"Erreur lors du calcul des Bandes de Bollinger: {str(e)}")
            return np.zeros(len(data)), np.zeros(len(data)), np.zeros(len(data))
            
    def get_signal(self, price: float, upper: float, lower: float) -> str:
        """
        Détermine le signal basé sur les Bandes de Bollinger
        
        Args:
            price (float): Prix actuel
            upper (float): Bande supérieure
            lower (float): Bande inférieure
            
        Returns:
            str: Signal généré ('buy', 'sell' ou 'neutral')
        """
        if price >= upper:
            return 'sell'
        elif price <= lower:
            return 'buy'
        else:
            return 'neutral' 