import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

class BaseStrategy:
    """
    Classe de base pour toutes les stratégies.
    """
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        pass
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
    def update(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        pass

class MeanReversionStrategy(BaseStrategy):
    """
    Stratégie mean reversion simple basée sur la déviation du prix par rapport à une moyenne mobile.
    """
    def __init__(self, window: int = 20, threshold: float = 2.0):
        self.window = window
        self.threshold = threshold
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ma = X['close'].rolling(self.window).mean()
        std = X['close'].rolling(self.window).std()
        signal = (X['close'] - ma) / std
        return np.where(signal < -self.threshold, 1, np.where(signal > self.threshold, -1, 0))

class MomentumStrategy(BaseStrategy):
    """
    Stratégie momentum basée sur le rendement cumulé.
    """
    def __init__(self, window: int = 10):
        self.window = window
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        returns = X['close'].pct_change(self.window)
        return np.where(returns > 0, 1, np.where(returns < 0, -1, 0))

class BreakoutStrategy(BaseStrategy):
    """
    Stratégie breakout basée sur les plus hauts/bas récents.
    """
    def __init__(self, window: int = 20):
        self.window = window
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        high = X['high'].rolling(self.window).max()
        low = X['low'].rolling(self.window).min()
        signal = np.where(X['close'] > high.shift(1), 1, np.where(X['close'] < low.shift(1), -1, 0))
        return signal

class ArbitrageStrategy(BaseStrategy):
    """
    Stratégie d'arbitrage simple entre deux actifs (spread trading).
    """
    def __init__(self, spread_threshold: float = 1.0):
        self.spread_threshold = spread_threshold
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # X doit contenir 'asset1', 'asset2'
        spread = X['asset1'] - X['asset2']
        return np.where(spread > self.spread_threshold, -1, np.where(spread < -self.spread_threshold, 1, 0))

class MultiTimeframeStrategy(BaseStrategy):
    """
    Stratégie combinant signaux multi-timeframe (ex: 1min, 5min, 1h).
    """
    def __init__(self, strategies: Dict[str, BaseStrategy]):
        self.strategies = strategies  # ex: {'1min': MeanReversionStrategy(), ...}
    def predict(self, X: Dict[str, pd.DataFrame]) -> np.ndarray:
        # X: {timeframe: DataFrame}
        signals = [s.predict(X[tf]) for tf, s in self.strategies.items() if tf in X]
        return np.sign(np.sum(signals, axis=0))

class OnlineParameterAdaptation:
    """
    Adaptation en ligne des paramètres d'une stratégie (ex: window, threshold).
    """
    def __init__(self, base_strategy: BaseStrategy):
        self.base_strategy = base_strategy
    def update(self, X: pd.DataFrame, y: pd.Series):
        # Exemple: ajuster window selon la volatilité
        vol = X['close'].rolling(20).std().iloc[-1]
        if hasattr(self.base_strategy, 'window'):
            self.base_strategy.window = max(5, int(20 * (1 + vol)))

class KellySizer:
    """
    Position sizing adaptatif selon le critère de Kelly modifié.
    """
    def __init__(self, risk_free: float = 0.0):
        self.risk_free = risk_free
    def size(self, winrate: float, winloss: float) -> float:
        k = winrate - (1 - winrate) / winloss if winloss > 0 else 0.01
        return max(0.01, min(k, 1.0))

class VaRSizer:
    """
    Position sizing basé sur la Value-at-Risk dynamique.
    """
    def size(self, pnl_series: pd.Series, alpha: float = 0.05) -> float:
        var = np.percentile(pnl_series, 100 * alpha)
        return max(0.01, min(0.1, 0.05 / abs(var) if var != 0 else 0.01))

class TrailingStop:
    """
    Stop dynamique multi-niveaux (trailing stop).
    """
    def __init__(self, distance: float = 0.01):
        self.distance = distance
    def get_stop(self, entry_price: float, current_price: float, direction: int) -> float:
        if direction > 0:
            return max(entry_price, current_price - self.distance * entry_price)
        elif direction < 0:
            return min(entry_price, current_price + self.distance * entry_price)
        else:
            return entry_price

class ExecutionAlgo:
    """
    Algorithmes d'exécution : iceberg, TWAP, VWAP.
    """
    @staticmethod
    def iceberg(total_qty: float, max_qty: float) -> List[float]:
        n = int(np.ceil(total_qty / max_qty))
        return [min(max_qty, total_qty - i * max_qty) for i in range(n)]
    @staticmethod
    def twap(total_qty: float, n_slices: int) -> List[float]:
        return [total_qty / n_slices] * n_slices
    @staticmethod
    def vwap(prices: List[float], vols: List[float], total_qty: float) -> List[float]:
        weights = np.array(vols) / np.sum(vols)
        return list(total_qty * weights)

class HybridStrategyEngine:
    """
    Moteur de stratégies hybrides/dynamiques pour trading algorithmique.
    """
    def __init__(self, strategies: List[BaseStrategy], sizer: Optional[Any] = None, stop: Optional[Any] = None, exec_algo: Optional[Any] = None):
        self.strategies = strategies
        self.sizer = sizer or KellySizer()
        self.stop = stop or TrailingStop()
        self.exec_algo = exec_algo or ExecutionAlgo()
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        for s in self.strategies:
            s.fit(X, y)
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        signals = np.array([s.predict(X) for s in self.strategies])
        return np.sign(np.sum(signals, axis=0))
    def update(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        for s in self.strategies:
            s.update(X, y)
    def position_size(self, winrate: float, winloss: float, pnl_series: Optional[pd.Series] = None) -> float:
        if isinstance(self.sizer, KellySizer):
            return self.sizer.size(winrate, winloss)
        elif isinstance(self.sizer, VaRSizer) and pnl_series is not None:
            return self.sizer.size(pnl_series)
        return 0.01
    def get_stop(self, entry_price: float, current_price: float, direction: int) -> float:
        return self.stop.get_stop(entry_price, current_price, direction)
    def execute_order(self, total_qty: float, mode: str = "iceberg", **kwargs) -> List[float]:
        if mode == "iceberg":
            return self.exec_algo.iceberg(total_qty, kwargs.get("max_qty", 1.0))
        elif mode == "twap":
            return self.exec_algo.twap(total_qty, kwargs.get("n_slices", 5))
        elif mode == "vwap":
            return self.exec_algo.vwap(kwargs["prices"], kwargs["vols"], total_qty)
        else:
            raise ValueError(f"Mode d'exécution non supporté : {mode}") 