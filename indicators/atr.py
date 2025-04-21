"""
Module pour le calcul de l'ATR (Average True Range)
"""

import numpy as np
import pandas as pd
from typing import List, Union, Dict

class ATR:
    """Classe pour le calcul de l'ATR"""
    
    def __init__(self, period: int = 14):
        """
        Initialise le calculateur ATR
        
        Args:
            period (int): Période pour le calcul de l'ATR
        """
        self.period = period
        
    def calculate(self, data: Union[pd.DataFrame, List[Dict], np.ndarray]) -> np.ndarray:
        """
        Calcule l'ATR pour les données fournies
        
        Args:
            data: Données de prix (doit contenir high, low, close)
            
        Returns:
            np.ndarray: Valeurs ATR calculées
        """
        try:
            # Conversion des données en DataFrame si nécessaire
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, np.ndarray):
                df = pd.DataFrame(data, columns=['high', 'low', 'close'])
            else:
                df = data.copy()
            
            # Calcul du True Range
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calcul de l'ATR
            atr = tr.rolling(window=self.period).mean()
            
            return atr.fillna(tr).values
            
        except Exception as e:
            print(f"Erreur lors du calcul de l'ATR: {str(e)}")
            return np.zeros(len(data))
            
    def get_volatility_level(self, atr: float, avg_atr: float) -> str:
        """
        Détermine le niveau de volatilité basé sur l'ATR
        
        Args:
            atr (float): Valeur ATR actuelle
            avg_atr (float): Moyenne de l'ATR sur une période plus longue
            
        Returns:
            str: Niveau de volatilité ('high', 'normal', 'low')
        """
        ratio = atr / avg_atr if avg_atr > 0 else 1.0
        
        if ratio >= 1.5:
            return 'high'
        elif ratio <= 0.5:
            return 'low'
        else:
            return 'normal' 