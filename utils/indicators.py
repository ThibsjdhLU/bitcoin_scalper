"""
Module d'indicateurs techniques pour l'analyse de trading.
Fournit des fonctions pour calculer divers indicateurs techniques.
"""
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

def calculate_ema(
    data: Union[pd.Series, List[float], np.ndarray],
    period: int,
    smoothing: float = 2.0
) -> pd.Series:
    """
    Calcule l'Exponential Moving Average (EMA).
    
    Args:
        data: Série de prix ou liste/array de prix
        period: Période de l'EMA
        smoothing: Facteur de lissage (par défaut 2.0)
        
    Returns:
        pd.Series: EMA calculée
    """
    # Convertir les données en Series pandas si nécessaire
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # S'assurer que les données sont numériques
    data = pd.to_numeric(data, errors='coerce')
    
    # Calculer l'EMA
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(
    data: pd.Series,
    period: int
) -> pd.Series:
    """
    Calcule le Simple Moving Average (SMA).
    
    Args:
        data: Série de prix
        period: Période du SMA
        
    Returns:
        pd.Series: SMA calculé
    """
    return data.rolling(window=period).mean()

def calculate_rsi(
    data: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calcule le Relative Strength Index (RSI).
    
    Args:
        data: Série de prix
        period: Période du RSI (par défaut 14)
        
    Returns:
        pd.Series: RSI calculé
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Remplacer les valeurs NaN par 50 (zone neutre)
    rsi = rsi.fillna(50)
    
    # S'assurer que les valeurs sont entre 0 et 100
    rsi = rsi.clip(0, 100)
    
    return rsi

def calculate_macd(
    data: Union[pd.Series, List[float], np.ndarray],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcule le Moving Average Convergence Divergence (MACD).
    
    Args:
        data: Série de prix ou liste/array de prix
        fast_period: Période rapide (par défaut 12)
        slow_period: Période lente (par défaut 26)
        signal_period: Période du signal (par défaut 9)
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (MACD, Signal, Histogram)
    """
    # Convertir les données en Series pandas si nécessaire
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    
    # S'assurer que les données sont numériques
    data = pd.to_numeric(data, errors='coerce')
    
    # Calculer les EMAs
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)
    
    # Calculer le MACD
    macd = fast_ema - slow_ema
    
    # Calculer la ligne de signal
    signal = calculate_ema(macd, signal_period)
    
    # Calculer l'histogramme
    histogram = macd - signal
    
    # S'assurer que toutes les séries ont le même index
    macd = pd.Series(macd, index=data.index)
    signal = pd.Series(signal, index=data.index)
    histogram = pd.Series(histogram, index=data.index)
    
    return macd, signal, histogram

def calculate_bollinger_bands(
    data: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcule les Bandes de Bollinger.
    
    Args:
        data: Série de prix
        period: Période de la moyenne mobile (par défaut 20)
        num_std: Nombre d'écarts-types (par défaut 2.0)
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (Bande supérieure, Moyenne, Bande inférieure)
    """
    middle_band = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    # S'assurer que les bandes sont dans le bon ordre
    upper_band = upper_band.bfill()
    middle_band = middle_band.bfill()
    lower_band = lower_band.bfill()
    
    return upper_band, middle_band, lower_band

def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calcule l'Average True Range (ATR).
    
    Args:
        high: Série des prix hauts
        low: Série des prix bas
        close: Série des prix de clôture
        period: Période de l'ATR (par défaut 14)
        
    Returns:
        pd.Series: ATR calculé
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Remplacer les valeurs NaN par la première valeur valide
    atr = atr.bfill()
    
    # S'assurer que l'ATR est positif
    atr = atr.clip(lower=0)
    
    return atr

def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calcule l'oscillateur stochastique.
    
    Args:
        high: Série des prix hauts
        low: Série des prix bas
        close: Série des prix de clôture
        k_period: Période %K (par défaut 14)
        d_period: Période %D (par défaut 3)
        
    Returns:
        Tuple[pd.Series, pd.Series]: (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    
    # Remplacer les valeurs NaN par 50 (zone neutre)
    k = k.fillna(50)
    d = d.fillna(50)
    
    # S'assurer que les valeurs sont entre 0 et 100
    k = k.clip(0, 100)
    d = d.clip(0, 100)
    
    return k, d

def calculate_volume_profile(
    price: pd.Series,
    volume: pd.Series,
    num_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule le profil de volume.
    
    Args:
        price: Série des prix
        volume: Série des volumes
        num_bins: Nombre de bins (par défaut 10)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (Prix des bins, Volumes des bins)
    """
    bins = np.linspace(price.min(), price.max(), num_bins)
    volumes = np.zeros_like(bins)
    
    for i in range(len(bins) - 1):
        mask = (price >= bins[i]) & (price < bins[i + 1])
        volumes[i] = volume[mask].sum()
    
    return bins, volumes

def calculate_support_resistance(
    price: pd.Series,
    window: int = 20,
    threshold: float = 0.02
) -> Tuple[List[float], List[float]]:
    """
    Identifie les niveaux de support et résistance.
    
    Args:
        price: Série des prix
        window: Fenêtre de recherche (par défaut 20)
        threshold: Seuil de détection (par défaut 0.02)
        
    Returns:
        Tuple[List[float], List[float]]: (Supports, Résistances)
    """
    supports = []
    resistances = []
    
    for i in range(window, len(price) - window):
        window_data = price[i - window:i + window]
        
        if price.iloc[i] == window_data.min():
            supports.append(price.iloc[i])
        elif price.iloc[i] == window_data.max():
            resistances.append(price.iloc[i])
    
    # Fusionner les niveaux proches
    supports = _merge_levels(supports, threshold)
    resistances = _merge_levels(resistances, threshold)
    
    return supports, resistances

def _merge_levels(levels: List[float], threshold: float) -> List[float]:
    """
    Fusionne les niveaux proches.
    
    Args:
        levels: Liste des niveaux
        threshold: Seuil de fusion
        
    Returns:
        List[float]: Niveaux fusionnés
    """
    if not levels:
        return []
    
    levels = sorted(levels)
    merged = [levels[0]]
    
    for level in levels[1:]:
        if abs(level - merged[-1]) / merged[-1] > threshold:
            merged.append(level)
    
    return merged 