import numpy as np
import pandas as pd
from typing import Tuple, List

class TechnicalIndicators:
    """Classe pour calculer les indicateurs techniques"""
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calcule le RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calcule le MACD (Moving Average Convergence Divergence)"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Calcule les bandes de Bollinger"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band.iloc[-1], sma.iloc[-1], lower_band.iloc[-1]

    def calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """Calcule les niveaux de support et résistance"""
        prices = data['close'].values
        support = min(prices[-window:])
        resistance = max(prices[-window:])
        return support, resistance 