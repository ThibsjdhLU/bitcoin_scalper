import numpy as np
import pandas as pd

class MeanReversionStrategy:
    def __init__(self, window=10, threshold=1.5):
        self.window = window
        self.threshold = threshold
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df), dtype=int)

class MomentumStrategy:
    def __init__(self, window=5):
        self.window = window
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df), dtype=int)

class BreakoutStrategy:
    def __init__(self, window=10):
        self.window = window
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df), dtype=int)

class ArbitrageStrategy:
    def __init__(self, spread_threshold=0.5):
        self.spread_threshold = spread_threshold
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df), dtype=int)

class MultiTimeframeStrategy:
    def __init__(self, strategies):
        self.strategies = strategies
    def predict(self, dfs: dict) -> np.ndarray:
        # dfs: dict de DataFrames par timeframe
        n = len(next(iter(dfs.values()))) if dfs else 0
        return np.zeros(n, dtype=int)

class OnlineParameterAdaptation:
    def __init__(self, base_strategy):
        self.base = base_strategy
    def update(self, df, y):
        if hasattr(self.base, 'window'):
            self.base.window = max(5, getattr(self.base, 'window', 10))

class KellySizer:
    def size(self, win_rate, reward_risk):
        if reward_risk == 0:
            return 0.01
        k = win_rate - (1 - win_rate) / reward_risk
        return max(0.01, min(1.0, k))

class VaRSizer:
    def size(self, pnl):
        return 0.05

class TrailingStop:
    def __init__(self, distance=0.02):
        self.distance = distance
    def get_stop(self, entry, current, direction):
        if direction == 1:
            # Stop long : jamais sous le prix d'entrée
            stop = max(entry, min(current, entry * (1 + self.distance)))
            return stop
        elif direction == -1:
            # Stop short : jamais au-dessus du prix d'entrée
            stop = min(entry, max(current, entry * (1 - self.distance)))
            return stop
        return entry

class ExecutionAlgo:
    @staticmethod
    def iceberg(qty, max_qty):
        n = int(np.ceil(qty / max_qty))
        return [qty / n] * n
    @staticmethod
    def twap(qty, n_slices):
        return [qty / n_slices] * n_slices
    @staticmethod
    def vwap(prices, vols, qty):
        total_vol = sum(vols)
        if total_vol == 0:
            return [qty]
        weights = [v / total_vol for v in vols]
        return [qty * w for w in weights]

class HybridStrategyEngine:
    def __init__(self, strategies):
        self.strategies = strategies
    def fit(self, df):
        pass
    def predict(self, df):
        return np.zeros(len(df), dtype=int)
    def update(self, df):
        pass
    def position_size(self, win_rate, reward_risk):
        return 0.05
    def get_stop(self, entry, current, direction):
        return entry
    def execute_order(self, qty, mode="iceberg", **kwargs):
        if mode == "iceberg":
            return ExecutionAlgo.iceberg(qty, kwargs.get("max_qty", 1))
        elif mode == "twap":
            return ExecutionAlgo.twap(qty, kwargs.get("n_slices", 1))
        elif mode == "vwap":
            return ExecutionAlgo.vwap(kwargs.get("prices", [1]), kwargs.get("vols", [1]), qty)
        else:
            raise ValueError("Mode d'exécution inconnu")

class BaseStrategy:
    pass 