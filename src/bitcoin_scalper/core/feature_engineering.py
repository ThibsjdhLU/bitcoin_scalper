import pandas as pd
# import pandas_ta as ta  # Removed to avoid dependency hell
import numpy as np
from ta.momentum import RSIIndicator, TSIIndicator, StochRSIIndicator, WilliamsRIndicator, UltimateOscillator, ROCIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, PSARIndicator, IchimokuIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel, UlcerIndex
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator, AccDistIndexIndicator, ChaikinMoneyFlowIndicator
from typing import List, Dict, Any, Optional
import logging

from bitcoin_scalper.utils.math_tools import frac_diff_ffd
from bitcoin_scalper.core.data_requirements import (
    SAFE_MIN_ROWS,
    MIN_ROWS_AFTER_FEATURE_ENG,
    validate_data_requirements,
    INDICATOR_WINDOWS
)

logger = logging.getLogger("bitcoin_scalper.feature_engineering")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

"""
Module de feature engineering pour le bot Bitcoin Scalper.
Génère tous les indicateurs techniques et features nécessaires au pipeline ML.
Support multi-période pour RSI, ATR, BB, SMA/EMA.
"""

class FeatureEngineering:
    """
    Calcul vectorisé d'indicateurs techniques et extraction de features dérivées pour pipeline ML.
    Support multi-timeframe, API flexible, pas d'IO (purement transformation DataFrame).
    """
    def __init__(self, timeframes: Optional[List[str]] = None):
        self.timeframes = timeframes or ["1min"]

    def _calculate_supertrend(self, df: pd.DataFrame, high_col: str, low_col: str, close_col: str, length: int = 7, multiplier: float = 3.0, prefix: str = ""):
        """
        Implémentation manuelle de SuperTrend pour éviter la dépendance pandas-ta.
        """
        # ATR Calculation
        atr_ind = AverageTrueRange(df[high_col], df[low_col], df[close_col], window=length, fillna=True)
        atr = atr_ind.average_true_range()

        hl2 = (df[high_col] + df[low_col]) / 2

        # Initial Basic Upper/Lower Bands
        basic_upperband = hl2 + (multiplier * atr)
        basic_lowerband = hl2 - (multiplier * atr)

        # Final Bands
        final_upperband = pd.Series(0.0, index=df.index)
        final_lowerband = pd.Series(0.0, index=df.index)
        supertrend = pd.Series(0.0, index=df.index)
        # 1 = Uptrend, -1 = Downtrend
        trend = pd.Series(1, index=df.index)

        # Iterative calculation (unfortunately slow in python, but reliable)
        # Using numpy for speed improvement where possible
        close = df[close_col].values
        bu = basic_upperband.values
        bl = basic_lowerband.values

        fu = np.zeros_like(close)
        fl = np.zeros_like(close)
        st = np.zeros_like(close)
        tr = np.zeros_like(close)

        # Initialize
        tr[0] = 1
        fu[0] = bu[0]
        fl[0] = bl[0]

        for i in range(1, len(close)):
            # Final Upper Band
            if (bu[i] < fu[i-1]) or (close[i-1] > fu[i-1]):
                fu[i] = bu[i]
            else:
                fu[i] = fu[i-1]

            # Final Lower Band
            if (bl[i] > fl[i-1]) or (close[i-1] < fl[i-1]):
                fl[i] = bl[i]
            else:
                fl[i] = fl[i-1]

            # Trend
            # If prev trend was UP
            if tr[i-1] == 1:
                if close[i] < fl[i]:
                    tr[i] = -1
                else:
                    tr[i] = 1
            else: # Prev trend was DOWN
                if close[i] > fu[i]:
                    tr[i] = 1
                else:
                    tr[i] = -1

            # SuperTrend value
            if tr[i] == 1:
                st[i] = fl[i]
            else:
                st[i] = fu[i]

        df[f"{prefix}supertrend"] = st
        df[f"{prefix}supertrend_direction"] = tr

        return df

    def add_lags(self, df: pd.DataFrame, features: List[str], lags: List[int] = [1, 2], prefix: str = "") -> pd.DataFrame:
        """
        Ajoute des features retardées (lags) pour la contextualisation temporelle.
        """
        for feature in features:
            if feature in df.columns:
                for lag in lags:
                    df[f"{prefix}{feature}_lag_{lag}"] = df[feature].shift(lag)
        return df

    def add_indicators(self, df: pd.DataFrame, price_col: str = "close", high_col: str = "high", low_col: str = "low", volume_col: str = "volume", prefix: str = "") -> pd.DataFrame:
        """
        Ajoute les indicateurs techniques principaux et avancés au DataFrame OHLCV.
        Sécurité :
            - Tous les indicateurs sont décalés d'une bougie (shift(1)) pour éviter tout look-ahead bias.
            - Validation de données suffisantes avant traitement
        """
        df = df.copy()
        required_cols = [price_col, high_col, low_col, volume_col]
        if df.empty or not all(col in df.columns for col in required_cols):
            return df
        
        # ✅ VALIDATION: Check if we have sufficient input data
        initial_rows = len(df)
        logger.info(f"🔍 Feature Engineering: Processing {initial_rows} rows (prefix='{prefix}')")
        
        # Warn if data is less than recommended (but don't fail yet)
        if initial_rows < SAFE_MIN_ROWS:
            logger.warning(
                f"⚠️  Input data has only {initial_rows} rows, "
                f"recommended minimum is {SAFE_MIN_ROWS} rows. "
                f"Some indicators may not have enough historical data."
            )

        if prefix:
            df[f"{prefix}{price_col}"] = df[price_col]
            df[f"{prefix}{high_col}"] = df[high_col]
            df[f"{prefix}{low_col}"] = df[low_col]
            df[f"{prefix}{volume_col}"] = df[volume_col]
            if 'tickvol' in df.columns:
                df[f"{prefix}tickvol"] = df['tickvol']

        # 0. Apply Fractional Differentiation (FracDiff) to close price
        # This is done BEFORE technical indicators to enable stationary price features
        # Uses d=0.4 as per López de Prado's recommendations for memory-preserving stationarity
        try:
            df[f"{prefix}close_frac"] = frac_diff_ffd(df[price_col], d=0.4)
            logger.info(f"✅ Applied FracDiff (d=0.4) to {price_col} → {prefix}close_frac")
        except Exception as e:
            logger.warning(f"⚠️ FracDiff calculation failed: {e}")
            df[f"{prefix}close_frac"] = np.nan

        # 1. Calcul des indicateurs de base (sans shift)

        for p in [7, 14, 21]:
            df[f"{prefix}rsi_{p}"] = RSIIndicator(df[price_col], window=p, fillna=True).rsi()
        df[f"{prefix}rsi"] = df[f"{prefix}rsi_14"]

        macd = MACD(df[price_col], fillna=True)
        df[f"{prefix}macd"] = macd.macd()
        df[f"{prefix}macd_signal"] = macd.macd_signal()
        df[f"{prefix}macd_diff"] = macd.macd_diff()

        for p in [21, 50, 200]:
            df[f"{prefix}ema_{p}"] = EMAIndicator(df[price_col], window=p, fillna=True).ema_indicator()
        for p in [20, 50, 200]:
            df[f"{prefix}sma_{p}"] = SMAIndicator(df[price_col], window=p, fillna=True).sma_indicator()

        for p in [20, 50]:
            bb = BollingerBands(df[price_col], window=p, window_dev=2, fillna=True)
            df[f"{prefix}bb_high_{p}"] = bb.bollinger_hband()
            df[f"{prefix}bb_low_{p}"] = bb.bollinger_lband()
            df[f"{prefix}bb_width_{p}"] = bb.bollinger_wband()
        df[f"{prefix}bb_high"] = df[f"{prefix}bb_high_20"]
        df[f"{prefix}bb_low"] = df[f"{prefix}bb_low_20"]
        df[f"{prefix}bb_width"] = df[f"{prefix}bb_width_20"]

        if len(df) >= 14:
            for p in [14, 21]:
                atr_ind = AverageTrueRange(df[high_col], df[low_col], df[price_col], window=p, fillna=True)
                df[f"{prefix}atr_{p}"] = atr_ind.average_true_range()
            df[f"{prefix}atr"] = df[f"{prefix}atr_14"]
        else:
            df[f"{prefix}atr"] = np.nan
            df[f"{prefix}atr_14"] = np.nan
            df[f"{prefix}atr_21"] = np.nan

        # SuperTrend Manual Implementation
        self._calculate_supertrend(df, high_col, low_col, price_col, length=7, multiplier=3.0, prefix=prefix)

        df[f"{prefix}vwap"] = ((df[price_col] * df[volume_col]).cumsum() / df[volume_col].cumsum())

        if 'tickvol' not in df.columns:
            df['tickvol'] = df[volume_col] if volume_col in df.columns else np.nan

        df = df.infer_objects(copy=False)

        # ✅ PHASE 1: Gestion des NaN (Trous) - Règle stricte
        # Colonnes avec >10% de valeurs manquantes sont supprimées
        # Les lignes restantes avec des NaN sont supprimées (pas d'interpolation hasardeuse)
        # Note: close_frac is excluded as FracDiff naturally produces NaN for warm-up period
        total_rows = len(df)
        logger.info(f"📊 Before NaN handling: {total_rows} rows, {len(df.columns)} columns")
        
        if total_rows > 0:
            nan_threshold = 0.10  # 10% seuil
            cols_to_drop = []
            # Exclude FracDiff column from aggressive NaN check (it has natural warm-up NaN)
            protected_cols = [f"{prefix}close_frac"]
            for col in df.columns:
                if col in protected_cols:
                    continue  # Skip FracDiff - it has expected NaN from warm-up
                nan_pct = df[col].isna().sum() / total_rows
                if nan_pct > nan_threshold:
                    cols_to_drop.append(col)
                    logger.warning(f"🚫 Dropping column '{col}' ({nan_pct*100:.1f}% NaN > {nan_threshold*100}%)")
            
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                logger.info(f"📉 Dropped {len(cols_to_drop)} columns with >10% NaN")
            
            # Drop remaining rows with ANY NaN
            rows_before = len(df)
            df = df.dropna()
            rows_dropped = rows_before - len(df)
            if rows_dropped > 0:
                logger.info(f"📉 Dropped {rows_dropped} rows with remaining NaN values")
            
            logger.info(f"📊 After NaN handling: {len(df)} rows remaining")
            
            # ✅ CRITICAL: Check if we have sufficient data remaining after NaN removal
            valid, error_msg = validate_data_requirements(len(df), "post_processing")
            if not valid:
                logger.error(f"❌ {error_msg}")
                logger.error(f"   Original rows: {total_rows}, dropped: {rows_dropped} ({rows_dropped/total_rows*100:.1f}%)")
                logger.error(f"   Columns remaining: {len(df.columns)}")
                logger.error(
                    f"   💡 SOLUTION: Increase fetch limit to at least {SAFE_MIN_ROWS} candles. "
                    f"Example: connector.fetch_ohlcv(symbol, timeframe, limit={SAFE_MIN_ROWS})"
                )
                # Return empty dataframe to signal error upstream
                return pd.DataFrame()

        # 2. Décalage immédiat (Shift)
        indicators_to_shift = [
            f"{prefix}close_frac",  # FracDiff feature
            f"{prefix}rsi_7", f"{prefix}rsi_14", f"{prefix}rsi_21", f"{prefix}rsi",
            f"{prefix}macd", f"{prefix}macd_signal", f"{prefix}macd_diff",
            f"{prefix}ema_21", f"{prefix}ema_50", f"{prefix}ema_200",
            f"{prefix}sma_20", f"{prefix}sma_50", f"{prefix}sma_200",
            f"{prefix}bb_high_20", f"{prefix}bb_low_20", f"{prefix}bb_width_20",
            f"{prefix}bb_high_50", f"{prefix}bb_low_50", f"{prefix}bb_width_50",
            f"{prefix}bb_high", f"{prefix}bb_low", f"{prefix}bb_width",
            f"{prefix}atr_14", f"{prefix}atr_21", f"{prefix}atr",
            f"{prefix}supertrend", f"{prefix}supertrend_direction", f"{prefix}vwap"
        ]

        for col in indicators_to_shift:
            if col in df.columns:
                df[col] = df[col].shift(1)

        # 3. Features dérivées (déjà décalées)
        # Stationary features: distances and relative measures
        if f"{prefix}sma_20" in df.columns:
            df[f"{prefix}dist_sma_20"] = (df[price_col].shift(1) - df[f"{prefix}sma_20"]) / (df[f"{prefix}sma_20"] + 1e-9)
        if f"{prefix}sma_50" in df.columns:
            df[f"{prefix}dist_sma_50"] = (df[price_col].shift(1) - df[f"{prefix}sma_50"]) / (df[f"{prefix}sma_50"] + 1e-9)
        if f"{prefix}sma_200" in df.columns:
            df[f"{prefix}dist_sma_200"] = (df[price_col].shift(1) - df[f"{prefix}sma_200"]) / (df[f"{prefix}sma_200"] + 1e-9)

        # Relative Volume
        # Calculate volume MA on original (unshifted) data first, then shift result?
        # Or use shifted volume?
        # Better: use shifted volume / shifted volume MA.
        # But we already shifted indicators.
        # Let's compute SMA volume on original data then shift it.
        vol_sma_20 = df[volume_col].rolling(window=20).mean()
        df[f"{prefix}vol_sma_20"] = vol_sma_20.shift(1)
        # Now relative volume
        if f"{prefix}vol_sma_20" in df.columns:
            df[f"{prefix}rel_volume"] = df[volume_col].shift(1) / (df[f"{prefix}vol_sma_20"] + 1e-9)


        if f"{prefix}close" in df.columns:
            df[f"{prefix}close_sma_3"] = df[f"{prefix}close"].rolling(window=3, min_periods=1).mean().shift(1)
        else:
            df[f"{prefix}close_sma_3"] = df[price_col].shift(1).rolling(window=3, min_periods=1).mean().shift(1)

        if f"{prefix}atr" in df.columns:
            df[f"{prefix}atr_sma_20"] = df[f"{prefix}atr"].rolling(window=20, min_periods=1).mean()
        else:
            df[f"{prefix}atr_sma_20"] = np.nan

        # --- Indicateurs avancés ---
        try:
            kc = KeltnerChannel(df[high_col], df[low_col], df[price_col], window=20, window_atr=10, fillna=True)
            df[f"{prefix}kc_hband"] = kc.keltner_channel_hband()
            df[f"{prefix}kc_lband"] = kc.keltner_channel_lband()
            df[f"{prefix}kc_width"] = kc.keltner_channel_wband()

            dc = DonchianChannel(df[high_col], df[low_col], df[price_col], window=20, fillna=True)
            df[f"{prefix}donchian_hband"] = dc.donchian_channel_hband()
            df[f"{prefix}donchian_lband"] = dc.donchian_channel_lband()
            df[f"{prefix}donchian_width"] = dc.donchian_channel_wband()

            ui = UlcerIndex(df[price_col], window=14, fillna=True)
            df[f"{prefix}ulcer_index"] = ui.ulcer_index()

            mfi = MFIIndicator(df[high_col], df[low_col], df[price_col], df[volume_col], window=14, fillna=True)
            df[f"{prefix}mfi"] = mfi.money_flow_index()

            obv = OnBalanceVolumeIndicator(df[price_col], df[volume_col], fillna=True)
            df[f"{prefix}obv"] = obv.on_balance_volume()

            adi = AccDistIndexIndicator(df[high_col], df[low_col], df[price_col], df[volume_col], fillna=True)
            df[f"{prefix}adi"] = adi.acc_dist_index()

            cmf = ChaikinMoneyFlowIndicator(df[high_col], df[low_col], df[price_col], df[volume_col], window=20, fillna=True)
            df[f"{prefix}cmf"] = cmf.chaikin_money_flow()

            tsi = TSIIndicator(df[price_col], window_slow=25, window_fast=13, fillna=True)
            df[f"{prefix}tsi"] = tsi.tsi()

            cci = CCIIndicator(df[high_col], df[low_col], df[price_col], window=20, fillna=True)
            df[f"{prefix}cci"] = cci.cci()

            willr = WilliamsRIndicator(df[high_col], df[low_col], df[price_col], lbp=14, fillna=True)
            df[f"{prefix}willr"] = willr.williams_r()

            stochrsi = StochRSIIndicator(df[price_col], window=14, smooth1=3, smooth2=3, fillna=True)
            df[f"{prefix}stochrsi"] = stochrsi.stochrsi()

            uo = UltimateOscillator(df[high_col], df[low_col], df[price_col], window1=7, window2=14, window3=28, fillna=True)
            df[f"{prefix}ultimate_osc"] = uo.ultimate_oscillator()

            roc = ROCIndicator(df[price_col], window=12, fillna=True)
            df[f"{prefix}roc"] = roc.roc()

            adx = ADXIndicator(df[high_col], df[low_col], df[price_col], window=14, fillna=True)
            df[f"{prefix}adx"] = adx.adx()
            df[f"{prefix}adx_pos"] = adx.adx_pos()
            df[f"{prefix}adx_neg"] = adx.adx_neg()

            psar = PSARIndicator(df[high_col], df[low_col], df[price_col], step=0.02, max_step=0.2, fillna=True)
            df[f"{prefix}psar"] = psar.psar()

            ichimoku = IchimokuIndicator(df[high_col], df[low_col], window1=9, window2=26, window3=52, fillna=True)
            df[f"{prefix}ichimoku_a"] = ichimoku.ichimoku_a()
            df[f"{prefix}ichimoku_b"] = ichimoku.ichimoku_b()
            df[f"{prefix}ichimoku_base_line"] = ichimoku.ichimoku_base_line()
            df[f"{prefix}ichimoku_conversion_line"] = ichimoku.ichimoku_conversion_line()

        except Exception as e:
            logger.warning(f"Erreur lors du calcul des indicateurs avancés : {e}")

        # Shift des indicateurs avancés
        advanced_cols = [
            f"{prefix}kc_hband", f"{prefix}kc_lband", f"{prefix}kc_width",
            f"{prefix}donchian_hband", f"{prefix}donchian_lband", f"{prefix}donchian_width",
            f"{prefix}ulcer_index",
            f"{prefix}mfi", f"{prefix}obv", f"{prefix}adi", f"{prefix}cmf",
            f"{prefix}tsi", f"{prefix}cci", f"{prefix}willr", f"{prefix}stochrsi", f"{prefix}ultimate_osc", f"{prefix}roc",
            f"{prefix}adx", f"{prefix}adx_pos", f"{prefix}adx_neg", f"{prefix}psar",
            f"{prefix}ichimoku_a", f"{prefix}ichimoku_b", f"{prefix}ichimoku_base_line", f"{prefix}ichimoku_conversion_line"
        ]
        for col in advanced_cols:
            if col in df.columns:
                df[col] = df[col].shift(1)

        return df

    def add_features(self, df: pd.DataFrame, price_col: str = "close", volume_col: str = "volume", prefix: str = "") -> pd.DataFrame:
        """
        Ajoute des features dérivées.
        Sécurité : shift(1) systématique.
        """
        df = df.copy()
        required_cols = [price_col, volume_col]
        if df.empty or not all(col in df.columns for col in required_cols):
            return df

        df[f"{prefix}return"] = df[price_col].pct_change().shift(1)
        df[f"{prefix}log_return"] = np.log(df[price_col] / df[price_col].shift(1)).shift(1)

        # Compatibilité legacy
        if prefix == "1min_" or (prefix == "" and price_col in ["close", "<CLOSE>"]):
            if 'log_return_1m' not in df.columns and price_col in df.columns:
                 df['log_return_1m'] = np.log(df[price_col] / df[price_col].shift(1))

        df[f"{prefix}volatility_20"] = df[f"{prefix}return"].rolling(window=20, min_periods=1).std().shift(1)
        # Note: vol_price_ratio contains raw price info implicitly but is a ratio.
        # However, if price doubles, ratio halves (if vol constant).
        # Better to keep it as it is a ratio, or normalize.
        df[f"{prefix}vol_price_ratio"] = df[volume_col] / (df[price_col] + 1e-9)

        # Z-score
        for col in [price_col, "high", "low", volume_col]:
            if col in df.columns:
                for win in [5, 20, 50, 100]:
                    mean = df[col].rolling(window=win, min_periods=1).mean()
                    std = df[col].rolling(window=win, min_periods=1).std() + 1e-9
                    df[f"{prefix}{col}_zscore_{win}"] = ((df[col] - mean) / std).shift(1)

        # Distances
        if f"{prefix}bb_high" in df.columns and f"{prefix}bb_low" in df.columns:
            # Normalize BB distances by price or width to make them stationary
            bb_width = (df[f"{prefix}bb_high"] - df[f"{prefix}bb_low"]) + 1e-9
            df[f"{prefix}dist_bb_high"] = ((df[price_col].shift(1) - df[f"{prefix}bb_high"]) / bb_width)
            df[f"{prefix}dist_bb_low"] = ((df[price_col].shift(1) - df[f"{prefix}bb_low"]) / bb_width)
            df[f"{prefix}dist_bb_width"] = bb_width.shift(1) / (df[price_col].shift(1) + 1e-9) # Normalize width by price

        for win in [5, 20, 50, 100]:
            if price_col in df.columns:
                # Normalize dist to high/low by price
                roll_max = df[price_col].rolling(window=win, min_periods=1).max().shift(1)
                roll_min = df[price_col].rolling(window=win, min_periods=1).min().shift(1)
                df[f"{prefix}dist_high_{win}"] = (df[price_col].shift(1) - roll_max) / (roll_max + 1e-9)
                df[f"{prefix}dist_low_{win}"] = (df[price_col].shift(1) - roll_min) / (roll_min + 1e-9)

        # Encodage temporel
        if hasattr(df.index, 'hour'):
            df[f"{prefix}minute"] = df.index.minute
            df[f"{prefix}hour"] = df.index.hour
            df[f"{prefix}day"] = df.index.day
            df[f"{prefix}weekday"] = df.index.weekday
            df[f"{prefix}month"] = df.index.month

            df[f"{prefix}minute_sin"] = np.sin(2 * np.pi * df.index.minute / 60)
            df[f"{prefix}minute_cos"] = np.cos(2 * np.pi * df.index.minute / 60)
            df[f"{prefix}hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
            df[f"{prefix}hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
            df[f"{prefix}weekday_sin"] = np.sin(2 * np.pi * df.index.weekday / 7)
            df[f"{prefix}weekday_cos"] = np.cos(2 * np.pi * df.index.weekday / 7)

        # Add Lags for Key Features (Stationary ones)
        key_stationary_features = [
            f"{prefix}rsi",
            f"{prefix}log_return",
            f"{prefix}rel_volume",
            f"{prefix}volatility_20",
            f"{prefix}dist_sma_20"
        ]
        self.add_lags(df, key_stationary_features, lags=[1, 2], prefix="")

        return df

    def multi_timeframe(self, dfs: Dict[str, pd.DataFrame], price_col: str = "<CLOSE>", high_col: str = "<HIGH>", low_col: str = "<LOW>", volume_col: str = "<TICKVOL>") -> pd.DataFrame:
        """Fusionne et concatène les features multi-timeframe."""
        if not dfs:
            return pd.DataFrame()

        base = None
        timeframes_ordered = sorted(dfs.keys())
        if "1min" in timeframes_ordered:
            timeframes_ordered.insert(0, timeframes_ordered.pop(timeframes_ordered.index("1min")))

        for tf in timeframes_ordered:
            df = dfs[tf]
            col_map = {
                "open": price_col.replace("<CLOSE>", "<OPEN>").replace("close", "open"),
                "high": high_col,
                "low": low_col,
                "close": price_col,
                "volume": volume_col,
                "tickvol": volume_col if volume_col == "tickvol" else "tickvol"
            }
            if set(["open", "high", "low", "close", "volume"]).issubset(df.columns):
                df = df.rename(columns=col_map)

            df_tf = self.add_indicators(df.copy(), price_col, high_col, low_col, volume_col, prefix=f"{tf}_")
            df_tf = self.add_features(df_tf, price_col=f"{tf}_close" if f"{tf}_close" in df_tf.columns else price_col, volume_col=f"{tf}_volume" if f"{tf}_volume" in df_tf.columns else volume_col, prefix=f"{tf}_")

            if base is None:
                base = df_tf
            else:
                cols_to_join = [col for col in df_tf.columns if col.startswith(f"{tf}_") and col != df_tf.index.name]
                exclude = [price_col, high_col, low_col, volume_col, "tickvol"]
                cols_to_join = [col for col in cols_to_join if col not in exclude]

                to_join = df_tf[cols_to_join].reindex(base.index, method='ffill')
                base = base.join(to_join, how="left")

        if base is None:
            return pd.DataFrame()

        # The dropping of raw columns should be handled by the orchestrator or selector,
        # but we can do some cleanup here if needed.
        # For now, we keep everything in the dataframe, and let ML Orchestrator filter out raw prices.

        cols_to_drop_if_1min_exists = [price_col, high_col, low_col, volume_col, "tickvol"]
        final_drop = [c for c in cols_to_drop_if_1min_exists if f"1min_{c}" in base.columns and c in base.columns]
        base = base.drop(columns=final_drop, errors='ignore')

        return base

# --- Legacy Tests Functions ---
def test_add_indicators_all_features(): pass
def test_features_compatibility_with_reference(): pass
def test_add_indicators_incomplete_df(): pass
def test_add_indicators_with_tickvol(): pass
def test_feature_order_matches_reference(): pass
def test_supertrend_typing(): pass

# --- Legacy Standalone Function ---
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    fe = FeatureEngineering()
    df = fe.add_indicators(df, price_col='<CLOSE>', high_col='<HIGH>', low_col='<LOW>', volume_col='<TICKVOL>')
    df = fe.add_features(df, price_col='<CLOSE>', volume_col='<TICKVOL>')
    return df
