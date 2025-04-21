"""
Module pour le calcul de l'indicateur RSI (Relative Strength Index)
"""

import numpy as np
import pandas as pd
from typing import List, Union, Dict

class RSI:
    """Classe pour le calcul du RSI"""
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        """
        Initialise le calculateur RSI
        
        Args:
            period (int): Période pour le calcul du RSI
            overbought (float): Niveau de surachat
            oversold (float): Niveau de survente
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
    def calculate(self, data: Union[pd.DataFrame, List[Dict], np.ndarray]) -> np.ndarray:
        """
        Calcule le RSI pour les données fournies
        
        Args:
            data: Données de prix (doit contenir les prix de clôture)
            
        Returns:
            np.ndarray: Valeurs RSI calculées
        """
        try:
            # Conversion des données en DataFrame si nécessaire
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, np.ndarray):
                df = pd.DataFrame(data, columns=['close'])
            else:
                df = data.copy()
            
            # Calcul des variations de prix
            delta = df['close'].diff()
            
            # Séparation des gains et pertes
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            
            # Calcul des moyennes mobiles des gains et pertes
            avg_gain = gain.rolling(window=self.period).mean()
            avg_loss = loss.rolling(window=self.period).mean()
            
            # Calcul de la force relative
            rs = avg_gain / avg_loss
            
            # Calcul du RSI
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50).values
            
        except Exception as e:
            print(f"Erreur lors du calcul du RSI: {str(e)}")
            return np.array([50] * len(data))  # Valeur neutre en cas d'erreur
            
    def get_signal(self, value: float) -> str:
        """
        Détermine le signal basé sur la valeur RSI
        
        Args:
            value (float): Valeur RSI actuelle
            
        Returns:
            str: Signal généré ('buy', 'sell' ou 'neutral')
        """
        if value <= self.oversold:
            return 'buy'
        elif value >= self.overbought:
            return 'sell'
        else:
            return 'neutral' 