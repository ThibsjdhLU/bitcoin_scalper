"""
Module d'analyse fractale et de détection des vagues d'Elliott
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN

class FractalAnalyzer:
    """
    Analyseur de motifs fractals sur les marchés financiers.
    """
    
    def __init__(self, window: int = 20):
        """
        Initialise l'analyseur de fractals.
        
        Args:
            window (int): Taille de la fenêtre d'analyse
        """
        self.window = window
        self.last_support = None
        self.last_resistance = None
    
    def find_fractal_points(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """
        Trouve les points fractals (supports et résistances).
        
        Args:
            data (pd.DataFrame): Données OHLCV
            
        Returns:
            Tuple[List[float], List[float]]: Points de support et de résistance
        """
        highs = data['high'].values
        lows = data['low'].values
        
        supports = []
        resistances = []
        
        for i in range(2, len(data) - 2):
            # Fractal haut
            if (highs[i] > highs[i-2] and 
                highs[i] > highs[i-1] and 
                highs[i] > highs[i+1] and 
                highs[i] > highs[i+2]):
                resistances.append(highs[i])
            
            # Fractal bas
            if (lows[i] < lows[i-2] and 
                lows[i] < lows[i-1] and 
                lows[i] < lows[i+1] and 
                lows[i] < lows[i+2]):
                supports.append(lows[i])
        
        return supports, resistances
    
    def analyze(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analyse les motifs fractals sur les données.
        
        Args:
            data (pd.DataFrame, optional): Données OHLCV
            
        Returns:
            Dict: Résultats de l'analyse
        """
        if data is None:
            return {
                'support_level': self.last_support,
                'resistance_level': self.last_resistance
            }
        
        supports, resistances = self.find_fractal_points(data)
        
        if supports:
            self.last_support = supports[-1]
        if resistances:
            self.last_resistance = resistances[-1]
        
        return {
            'support_level': self.last_support,
            'resistance_level': self.last_resistance,
            'all_supports': supports,
            'all_resistances': resistances
        }

class ElliottWaveAnalyzer:
    """
    Analyseur de vagues d'Elliott sur les marchés financiers.
    """
    
    def __init__(self, min_wave_length: int = 10):
        """
        Initialise l'analyseur de vagues d'Elliott.
        
        Args:
            min_wave_length (int): Longueur minimale d'une vague
        """
        self.min_wave_length = min_wave_length
        self.current_wave = 0
        self.wave_points = []
    
    def find_pivot_points(self, data: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Trouve les points pivots dans les données.
        
        Args:
            data (pd.DataFrame): Données OHLCV
            
        Returns:
            List[Tuple[int, float]]: Liste des points pivots (index, prix)
        """
        prices = data['close'].values
        pivots = []
        
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:  # Sommet local
                pivots.append((i, prices[i]))
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:  # Creux local
                pivots.append((i, prices[i]))
        
        return pivots
    
    def identify_waves(self, pivots: List[Tuple[int, float]]) -> List[Dict]:
        """
        Identifie les vagues d'Elliott dans les points pivots.
        
        Args:
            pivots (List[Tuple[int, float]]): Points pivots
            
        Returns:
            List[Dict]: Liste des vagues identifiées
        """
        waves = []
        current_trend = None
        wave_start = None
        
        for i in range(1, len(pivots)):
            prev_point = pivots[i-1]
            curr_point = pivots[i]
            
            # Détection de la tendance
            trend = 'up' if curr_point[1] > prev_point[1] else 'down'
            
            if current_trend != trend:
                if wave_start is not None:
                    wave_length = curr_point[0] - wave_start[0]
                    if wave_length >= self.min_wave_length:
                        waves.append({
                            'start': wave_start,
                            'end': prev_point,
                            'trend': current_trend,
                            'length': wave_length
                        })
                wave_start = prev_point
                current_trend = trend
        
        return waves
    
    def analyze(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analyse les vagues d'Elliott sur les données.
        
        Args:
            data (pd.DataFrame, optional): Données OHLCV
            
        Returns:
            Dict: Résultats de l'analyse
        """
        if data is None:
            return {
                'wave_count': self.current_wave,
                'wave_points': self.wave_points
            }
        
        pivots = self.find_pivot_points(data)
        waves = self.identify_waves(pivots)
        
        self.current_wave = len(waves)
        self.wave_points = [wave['end'] for wave in waves]
        
        return {
            'wave_count': self.current_wave,
            'waves': waves,
            'wave_points': self.wave_points
        } 