import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from typing import List, Dict, Any, Optional

class FeatureEngineering:
    """
    Calcul vectorisé d'indicateurs techniques et extraction de features dérivées pour pipeline ML.
    Support multi-timeframe, API flexible, pas d'IO (purement transformation DataFrame).
    """
    def __init__(self, timeframes: Optional[List[str]] = None):
        self.timeframes = timeframes or ["1min"]

    def add_indicators(self, df: pd.DataFrame, price_col: str = "close", high_col: str = "high", low_col: str = "low", volume_col: str = "volume", prefix: str = "") -> pd.DataFrame:
        """Ajoute les indicateurs techniques principaux au DataFrame OHLCV."""
        df = df.copy()
        required_cols = [price_col, high_col, low_col, volume_col]
        if df.empty or not all(col in df.columns for col in required_cols):
            return df
        # RSI
        df[f"{prefix}rsi"] = RSIIndicator(df[price_col], window=14, fillna=True).rsi()
        # MACD
        macd = MACD(df[price_col], fillna=True)
        df[f"{prefix}macd"] = macd.macd()
        df[f"{prefix}macd_signal"] = macd.macd_signal()
        df[f"{prefix}macd_diff"] = macd.macd_diff()
        # EMA/SMA
        df[f"{prefix}ema_20"] = EMAIndicator(df[price_col], window=20, fillna=True).ema_indicator()
        df[f"{prefix}sma_20"] = SMAIndicator(df[price_col], window=20, fillna=True).sma_indicator()
        # Bollinger Bands
        bb = BollingerBands(df[price_col], window=20, window_dev=2, fillna=True)
        df[f"{prefix}bb_high"] = bb.bollinger_hband()
        df[f"{prefix}bb_low"] = bb.bollinger_lband()
        df[f"{prefix}bb_width"] = bb.bollinger_wband()
        # ATR
        if len(df) >= 14:
            atr = AverageTrueRange(df[high_col], df[low_col], df[price_col], window=14, fillna=True)
            df[f"{prefix}atr"] = atr.average_true_range()
        else:
            df[f"{prefix}atr"] = np.nan
        # VWAP (rolling)
        df[f"{prefix}vwap"] = (df[price_col] * df[volume_col]).rolling(window=20, min_periods=1).sum() / df[volume_col].rolling(window=20, min_periods=1).sum()
        return df

    def add_features(self, df: pd.DataFrame, price_col: str = "close", volume_col: str = "volume", prefix: str = "") -> pd.DataFrame:
        """Ajoute des features dérivées (retours, volatilité, ratios, etc)."""
        df = df.copy()
        required_cols = [price_col, volume_col]
        if df.empty or not all(col in df.columns for col in required_cols):
            return df
        # Retours log et simples
        df[f"{prefix}return"] = df[price_col].pct_change().fillna(0)
        df[f"{prefix}log_return"] = np.log(df[price_col] / df[price_col].shift(1)).fillna(0)
        # Volatilité rolling
        df[f"{prefix}volatility_20"] = df[f"{prefix}return"].rolling(window=20, min_periods=1).std().fillna(0)
        # Ratio volume/prix
        df[f"{prefix}vol_price_ratio"] = df[volume_col] / (df[price_col] + 1e-9)
        return df

    def multi_timeframe(self, dfs: Dict[str, pd.DataFrame], price_col: str = "close", high_col: str = "high", low_col: str = "low", volume_col: str = "volume") -> pd.DataFrame:
        """Fusionne et concatène les features multi-timeframe (clé = timeframe, valeur = DataFrame OHLCV)."""
        base = None
        for tf, df in dfs.items():
            df_feat = self.add_indicators(df, price_col, high_col, low_col, volume_col, prefix=f"{tf}_")
            df_feat = self.add_features(df_feat, price_col, volume_col, prefix=f"{tf}_")
            if base is None:
                base = df_feat
            else:
                # Fusion sur l'index (timestamp)
                base = base.join(df_feat, rsuffix=f"_{tf}", how="outer")
        return base

"""
Exemple d'utilisation :
fe = FeatureEngineering(["1min", "5min"])
df_1m = fe.add_indicators(df_1m)
df_1m = fe.add_features(df_1m)
features = fe.multi_timeframe({"1min": df_1m, "5min": df_5m})
""" 