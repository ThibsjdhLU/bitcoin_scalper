"""
Module contenant les indicateurs techniques.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

class TechnicalIndicators:
    """Classe pour le calcul des indicateurs techniques."""
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """
        Calcule la Simple Moving Average (SMA).
        
        Args:
            data: Série de données
            period: Période de calcul
            
        Returns:
            pd.Series: SMA
        """
        return data.rolling(window=period).mean()
        
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """
        Calcule la Exponential Moving Average (EMA).
        
        Args:
            data: Série de données
            period: Période de calcul
            
        Returns:
            pd.Series: EMA
        """
        return data.ewm(span=period, adjust=False).mean()
        
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calcule les bandes de Bollinger.
        
        Args:
            data: Série de données
            period: Période de calcul
            std_dev: Nombre d'écarts-types
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: Upper band, Middle band, Lower band
        """
        middle_band = TechnicalIndicators.calculate_sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return upper_band, middle_band, lower_band
        
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcule le Relative Strength Index (RSI).
        
        Args:
            data: Série de données
            period: Période de calcul
            
        Returns:
            pd.Series: RSI
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    @staticmethod
    def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calcule le MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Série de données
            fast_period: Période courte
            slow_period: Période longue
            signal_period: Période du signal
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: MACD line, Signal line, Histogram
        """
        fast_ema = TechnicalIndicators.calculate_ema(data, fast_period)
        slow_ema = TechnicalIndicators.calculate_ema(data, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
        
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcule l'Average True Range (ATR).
        
        Args:
            high: Série des prix hauts
            low: Série des prix bas
            close: Série des prix de clôture
            period: Période de calcul
            
        Returns:
            pd.Series: ATR
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
        
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calcule l'oscillateur stochastique.
        
        Args:
            high: Série des prix hauts
            low: Série des prix bas
            close: Série des prix de clôture
            k_period: Période %K
            d_period: Période %D
            
        Returns:
            Tuple[pd.Series, pd.Series]: %K, %D
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d 