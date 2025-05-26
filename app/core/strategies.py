import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional

class MeanReversionStrategy:
    """
    Stratégie de mean reversion simple : entre en position inverse après un écart extrême (z-score).
    """
    def __init__(self, z_thresh: float = 2.0):
        self.z_thresh = z_thresh
    def generate_signal(self, df: pd.DataFrame, price_col: str = "close") -> np.ndarray:
        z = (df[price_col] - df[price_col].rolling(20).mean()) / (df[price_col].rolling(20).std() + 1e-9)
        signal = np.where(z > self.z_thresh, -1, np.where(z < -self.z_thresh, 1, 0))
        return signal

class MomentumStrategy:
    """
    Stratégie momentum : suit la tendance sur n périodes.
    """
    def __init__(self, window: int = 10):
        self.window = window
    def generate_signal(self, df: pd.DataFrame, price_col: str = "close") -> np.ndarray:
        ret = df[price_col].pct_change(self.window)
        signal = np.where(ret > 0, 1, np.where(ret < 0, -1, 0))
        return signal

class BreakoutStrategy:
    """
    Stratégie breakout : entre sur cassure de plus haut/bas n périodes.
    """
    def __init__(self, window: int = 20):
        self.window = window
    def generate_signal(self, df: pd.DataFrame, price_col: str = "close") -> np.ndarray:
        high = df[price_col].rolling(self.window).max()
        low = df[price_col].rolling(self.window).min()
        signal = np.where(df[price_col] >= high, 1, np.where(df[price_col] <= low, -1, 0))
        return signal

class ArbitrageDummyStrategy:
    """
    Stratégie d'arbitrage cross-asset (exemple fictif, à adapter selon les datas multi-actifs).
    """
    def generate_signal(self, df: pd.DataFrame, ref_col: str = "btc_close", hedge_col: str = "eth_close") -> np.ndarray:
        spread = df[ref_col] - df[hedge_col]
        z = (spread - spread.rolling(20).mean()) / (spread.rolling(20).std() + 1e-9)
        signal = np.where(z > 2, -1, np.where(z < -2, 1, 0))
        return signal

class AdaptivePositionSizing:
    """
    Taille de position adaptative selon volatilité, capital, risk per trade.
    """
    def __init__(self, risk_per_trade: float = 0.01):
        self.risk_per_trade = risk_per_trade
    def compute_size(self, capital: float, stop_loss: float, atr: float) -> float:
        risk_amount = capital * self.risk_per_trade
        size = risk_amount / (stop_loss * atr + 1e-9)
        return max(0.0, size)

class DynamicStop:
    """
    Stop loss dynamique basé sur ATR ou volatilité.
    """
    def __init__(self, atr_mult: float = 2.0):
        self.atr_mult = atr_mult
    def compute_stop(self, entry_price: float, atr: float, direction: int) -> float:
        if direction == 1:
            return entry_price - self.atr_mult * atr
        elif direction == -1:
            return entry_price + self.atr_mult * atr
        else:
            return entry_price 