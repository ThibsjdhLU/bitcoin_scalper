"""
Module pour le calcul de l'indicateur MACD (Moving Average Convergence Divergence)
"""

import numpy as np
import pandas as pd
from typing import List, Union, Dict, Tuple

class MACD:
    """Classe pour le calcul du MACD"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialise le calculateur MACD
        
        Args:
            fast_period (int): Période de la moyenne mobile rapide
            slow_period (int): Période de la moyenne mobile lente
            signal_period (int): Période de la ligne de signal
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def calculate(self, data: Union[pd.DataFrame, List[Dict], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule le MACD pour les données fournies
        
        Args:
            data: Données de prix (doit contenir les prix de clôture)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (MACD, Signal, Histogramme)
        """
        try:
            # Conversion des données en DataFrame si nécessaire
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, np.ndarray):
                df = pd.DataFrame(data, columns=['close'])
            else:
                df = data.copy()
            
            # Calcul des moyennes mobiles exponentielles
            ema_fast = df['close'].ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = df['close'].ewm(span=self.slow_period, adjust=False).mean()
            
            # Calcul du MACD
            macd = ema_fast - ema_slow
            
            # Calcul de la ligne de signal
            signal = macd.ewm(span=self.signal_period, adjust=False).mean()
            
            # Calcul de l'histogramme
            histogram = macd - signal
            
            return macd.values, signal.values, histogram.values
            
        except Exception as e:
            print(f"Erreur lors du calcul du MACD: {str(e)}")
            return np.zeros(len(data)), np.zeros(len(data)), np.zeros(len(data))
            
    def get_signal(self, macd: float, signal: float, prev_macd: float, prev_signal: float) -> str:
        """
        Détermine le signal basé sur les valeurs MACD
        
        Args:
            macd (float): Valeur MACD actuelle
            signal (float): Valeur de la ligne de signal actuelle
            prev_macd (float): Valeur MACD précédente
            prev_signal (float): Valeur de la ligne de signal précédente
            
        Returns:
            str: Signal généré ('buy', 'sell' ou 'neutral')
        """
        # Croisement haussier
        if prev_macd <= prev_signal and macd > signal:
            return 'buy'
        # Croisement baissier
        elif prev_macd >= prev_signal and macd < signal:
            return 'sell'
        else:
            return 'neutral' 