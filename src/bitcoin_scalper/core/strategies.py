import numpy as np
import pandas as pd

class MeanReversionStrategy:
    def __init__(self, z_thresh=1.5, window=10):
        self.z_thresh = z_thresh
        self.window = window
    def generate_signal(self, df: pd.DataFrame) -> np.ndarray:
        # Signal fictif pour le test
        return np.zeros(len(df), dtype=int)

class MomentumStrategy:
    def __init__(self, window=5):
        self.window = window
    def generate_signal(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df), dtype=int)

class BreakoutStrategy:
    def __init__(self, window=10):
        self.window = window
    def generate_signal(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df), dtype=int)

class ArbitrageDummyStrategy:
    def __init__(self):
        pass
    def generate_signal(self, df: pd.DataFrame, ref_col="btc_close", hedge_col="eth_close") -> np.ndarray:
        return np.zeros(len(df), dtype=int)

class AdaptivePositionSizing:
    def __init__(self, risk_per_trade=0.02):
        self.risk_per_trade = risk_per_trade
    def compute_size(self, capital, stop_loss, atr):
        return 1.0

class DynamicStop:
    def __init__(self, atr_mult=2.5):
        self.atr_mult = atr_mult
    def compute_stop(self, entry_price, atr, direction):
        if direction == 1:
            return entry_price - self.atr_mult * atr
        elif direction == -1:
            return entry_price + self.atr_mult * atr
        return entry_price 