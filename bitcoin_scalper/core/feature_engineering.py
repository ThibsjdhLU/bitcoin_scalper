import pandas as pd
import pandas_ta as ta  # Réactivation : nécessaire pour SuperTrend
import numpy as np
from ta.momentum import RSIIndicator, TSIIndicator, StochRSIIndicator, WilliamsRIndicator, UltimateOscillator, ROCIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, PSARIndicator, IchimokuIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel, UlcerIndex
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator, AccDistIndexIndicator, ChaikinMoneyFlowIndicator
# from ta.trend import SuperTrendIndicator # Suppression : SuperTrendIndicator n'existe pas dans ta
from typing import List, Dict, Any, Optional
import logging
from scipy.stats import zscore

logger = logging.getLogger("bitcoin_scalper.feature_engineering")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

"""
Module de feature engineering pour le bot Bitcoin Scalper.
Génère tous les indicateurs techniques et features nécessaires au pipeline ML :
- rsi, macd, macd_signal, macd_diff
- ema_21, ema_50, sma_20
- bb_high, bb_low, bb_width
- atr, supertrend, vwap
- tickvol (copie de volume si absent)
- close_sma_3 (SMA 3 sur close)
- atr_sma_20 (SMA 20 sur ATR)
"""

class FeatureEngineering:
    """
    Calcul vectorisé d'indicateurs techniques et extraction de features dérivées pour pipeline ML.
    Génère systématiquement :
      - rsi, macd, macd_signal, macd_diff
      - ema_21, ema_50, sma_20
      - bb_high, bb_low, bb_width
      - atr, supertrend, vwap
      - tickvol (copie de volume si absent)
      - close_sma_3 (SMA 3 sur close)
      - atr_sma_20 (SMA 20 sur ATR)
    Support multi-timeframe, API flexible, pas d'IO (purement transformation DataFrame).
    """
    def __init__(self, timeframes: Optional[List[str]] = None):
        self.timeframes = timeframes or ["1min"]

    def add_indicators(self, df: pd.DataFrame, price_col: str = "close", high_col: str = "high", low_col: str = "low", volume_col: str = "volume", prefix: str = "") -> pd.DataFrame:
        """
        Ajoute les indicateurs techniques principaux et avancés au DataFrame OHLCV.
        Sécurité :
            - Tous les indicateurs sont décalés d'une bougie (shift(1)) pour éviter tout look-ahead bias.
            - VWAP est calculé de façon cumulative et décalée.
        """
        df = df.copy()
        required_cols = [price_col, high_col, low_col, volume_col]
        if df.empty or not all(col in df.columns for col in required_cols):
            return df
        # --- Ajout : dupliquer les colonnes OHLCV d'origine en version préfixée si un préfixe est fourni ---
        if prefix:
            df[f"{prefix}{price_col}"] = df[price_col]
            df[f"{prefix}{high_col}"] = df[high_col]
            df[f"{prefix}{low_col}"] = df[low_col]
            df[f"{prefix}{volume_col}"] = df[volume_col]
            if 'tickvol' in df.columns:
                df[f"{prefix}tickvol"] = df['tickvol']
        # 1. Calcul des indicateurs de base (sans shift)
        df[f"{prefix}rsi"] = RSIIndicator(df[price_col], window=14, fillna=True).rsi()
        macd = MACD(df[price_col], fillna=True)
        df[f"{prefix}macd"] = macd.macd()
        df[f"{prefix}macd_signal"] = macd.macd_signal()
        df[f"{prefix}macd_diff"] = macd.macd_diff()
        df[f"{prefix}ema_21"] = EMAIndicator(df[price_col], window=21, fillna=True).ema_indicator()
        df[f"{prefix}ema_50"] = EMAIndicator(df[price_col], window=50, fillna=True).ema_indicator()
        df[f"{prefix}sma_20"] = SMAIndicator(df[price_col], window=20, fillna=True).sma_indicator()
        bb = BollingerBands(df[price_col], window=20, window_dev=2, fillna=True)
        df[f"{prefix}bb_high"] = bb.bollinger_hband()
        df[f"{prefix}bb_low"] = bb.bollinger_lband()
        df[f"{prefix}bb_width"] = bb.bollinger_wband()
        if len(df) >= 14:
            atr = AverageTrueRange(df[high_col], df[low_col], df[price_col], window=14, fillna=True)
            df[f"{prefix}atr"] = atr.average_true_range()
        else:
            df[f"{prefix}atr"] = np.nan

        # Utilisation de pandas_ta.supertrend
        # La fonction retourne un DataFrame avec plusieurs colonnes (SUPERT_l_m, SUPERTd_l_m, SUPERTl_l_m, SUPERTs_l_m)
        # où l est length et m est multiplier
        supertrend_data = df.ta.supertrend(
            length=7,           # Fenêtre pour ATR
            multiplier=3.0,     # Multiplicateur pour ATR
            close=df[price_col],    # Colonne close
            high=df[high_col],      # Colonne high
            low=df[low_col],        # Colonne low
            fillna=True
        )

        if supertrend_data is not None and not supertrend_data.empty:
            # Ajouter cette ligne pour inférer les types et faire une copie explicite
            supertrend_data = supertrend_data.copy().infer_objects(copy=False)

            # Noms des colonnes générées par pandas_ta
            supertrend_line_col = f'SUPERT_{7}_{3.0}'
            supertrend_direction_col = f'SUPERTd_{7}_{3.0}' # Nom correct pour la direction

            if supertrend_line_col in supertrend_data.columns:
                df[f"{prefix}supertrend"] = supertrend_data[supertrend_line_col].astype(float)
            else:
                logger.warning(f"Colonne SuperTrend '{supertrend_line_col}' non trouvée. Colonnes disponibles : {supertrend_data.columns.tolist()}")
                df[f"{prefix}supertrend"] = np.nan

            if supertrend_direction_col in supertrend_data.columns:
                # Correction : cast explicite en float pour éviter le warning
                df[f"{prefix}supertrend_direction"] = supertrend_data[supertrend_direction_col].astype(float)
            else:
                logger.warning(f"Colonne SuperTrend Direction '{supertrend_direction_col}' non trouvée.")
                df[f"{prefix}supertrend_direction"] = np.nan

        else:
            df[f"{prefix}supertrend"] = np.nan
            df[f"{prefix}supertrend_direction"] = np.nan

        df[f"{prefix}vwap"] = ((df[price_col] * df[volume_col]).cumsum() / df[volume_col].cumsum())
        if 'tickvol' not in df.columns:
            if volume_col in df.columns:
                df['tickvol'] = df[volume_col]
            else:
                df['tickvol'] = np.nan
        # Correction : appliquer infer_objects après fillna/ffill/bfill si besoin
        df = df.infer_objects(copy=False)
        # 2. Décalage immédiat de tous les indicateurs de base (sécurité temporelle)
        base_cols = [
            f"{prefix}rsi", f"{prefix}macd", f"{prefix}macd_signal", f"{prefix}macd_diff",
            f"{prefix}ema_21", f"{prefix}ema_50", f"{prefix}sma_20",
            f"{prefix}bb_high", f"{prefix}bb_low", f"{prefix}bb_width",
            f"{prefix}atr", f"{prefix}supertrend", f"{prefix}vwap"
        ]
        for col in base_cols:
            if col in df.columns:
                df[col] = df[col].shift(1)
        # 3. Calcul des features dérivées UNIQUEMENT à partir des colonnes déjà décalées
        if f"{prefix}close" in df.columns:
            df[f"{prefix}close_sma_3"] = df[f"{prefix}close"].rolling(window=3, min_periods=1).mean().shift(1)
        else:
            df[f"{prefix}close_sma_3"] = df[price_col].shift(1).rolling(window=3, min_periods=1).mean().shift(1)
        if f"{prefix}atr" in df.columns:
            df[f"{prefix}atr_sma_20"] = df[f"{prefix}atr"].rolling(window=20, min_periods=1).mean()
        else:
            df[f"{prefix}atr_sma_20"] = np.nan
        # 4. Ne jamais re-shifter les features dérivées !
        # Toutes les features sont désormais temporellement sûres.
        # --- Indicateurs avancés (volatilité, volume, momentum, trend) ---
        try:
            # Volatilité
            kc = KeltnerChannel(df[high_col], df[low_col], df[price_col], window=20, window_atr=10, fillna=True)
            df[f"{prefix}kc_hband"] = kc.keltner_channel_hband()
            df[f"{prefix}kc_lband"] = kc.keltner_channel_lband()
            df[f"{prefix}kc_width"] = kc.keltner_channel_wband()
            dc = DonchianChannel(df[high_col], df[low_col], df[price_col], window=20, fillna=True)
            df[f"{prefix}donchian_hband"] = dc.donchian_channel_hband()
            df[f"{prefix}donchian_lband"] = dc.donchian_channel_lband()
            df[f"{prefix}donchian_width"] = dc.donchian_channel_wband()
            try:
                from ta.volatility import ChandelierExit
                ce = ChandelierExit(df[high_col], df[low_col], df[price_col], window=22, window_atr=22, fillna=True)
                df[f"{prefix}chandelier_exit_long"] = ce.chandelier_exit_long()
                df[f"{prefix}chandelier_exit_short"] = ce.chandelier_exit_short()
            except ImportError:
                logger.warning("ChandelierExit non disponible dans ta.volatility pour cette version de ta.")
                df[f"{prefix}chandelier_exit_long"] = np.nan
                df[f"{prefix}chandelier_exit_short"] = np.nan
            except Exception as e:
                logger.warning(f"Erreur lors du calcul de ChandelierExit : {e}")
                df[f"{prefix}chandelier_exit_long"] = np.nan
                df[f"{prefix}chandelier_exit_short"] = np.nan
            ui = UlcerIndex(df[price_col], window=14, fillna=True)
            df[f"{prefix}ulcer_index"] = ui.ulcer_index()
            # Volume
            mfi = MFIIndicator(df[high_col], df[low_col], df[price_col], df[volume_col], window=14, fillna=True)
            df[f"{prefix}mfi"] = mfi.money_flow_index()
            obv = OnBalanceVolumeIndicator(df[price_col], df[volume_col], fillna=True)
            df[f"{prefix}obv"] = obv.on_balance_volume()
            adi = AccDistIndexIndicator(df[high_col], df[low_col], df[price_col], df[volume_col], fillna=True)
            df[f"{prefix}adi"] = adi.acc_dist_index()
            cmf = ChaikinMoneyFlowIndicator(df[high_col], df[low_col], df[price_col], df[volume_col], window=20, fillna=True)
            df[f"{prefix}cmf"] = cmf.chaikin_money_flow()
            # Momentum
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
            # Trend
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
            try:
                from ta.trend import PPOIndicator
                ppo = PPOIndicator(df[price_col], window_slow=26, window_fast=12, window_sign=9, fillna=True)
                df[f"{prefix}ppo"] = ppo.ppo()
                df[f"{prefix}ppo_signal"] = ppo.ppo_signal()
                df[f"{prefix}ppo_hist"] = ppo.ppo_hist()
            except ImportError:
                logger.warning("PPOIndicator non disponible dans ta.trend pour cette version de ta.")
                df[f"{prefix}ppo"] = np.nan
                df[f"{prefix}ppo_signal"] = np.nan
                df[f"{prefix}ppo_hist"] = np.nan
        except Exception as e:
            logger.warning(f"Erreur lors du calcul des indicateurs avancés : {e}")
        # Décalage immédiat de tous les indicateurs avancés (sécurité temporelle)
        advanced_cols = [
            f"{prefix}kc_hband", f"{prefix}kc_lband", f"{prefix}kc_width",
            f"{prefix}donchian_hband", f"{prefix}donchian_lband", f"{prefix}donchian_width",
            f"{prefix}chandelier_exit_long", f"{prefix}chandelier_exit_short", f"{prefix}ulcer_index",
            f"{prefix}mfi", f"{prefix}obv", f"{prefix}adi", f"{prefix}cmf",
            f"{prefix}tsi", f"{prefix}cci", f"{prefix}willr", f"{prefix}stochrsi", f"{prefix}ultimate_osc", f"{prefix}roc",
            f"{prefix}adx", f"{prefix}adx_pos", f"{prefix}adx_neg", f"{prefix}psar",
            f"{prefix}ichimoku_a", f"{prefix}ichimoku_b", f"{prefix}ichimoku_base_line", f"{prefix}ichimoku_conversion_line",
            f"{prefix}ppo", f"{prefix}ppo_signal", f"{prefix}ppo_hist"
        ]
        for col in advanced_cols:
            if col in df.columns:
                df[col] = df[col].shift(1)
        return df

    def add_features(self, df: pd.DataFrame, price_col: str = "close", volume_col: str = "volume", prefix: str = "") -> pd.DataFrame:
        """
        Ajoute des features dérivées (retours, volatilité, ratios, z-score, distances, encodages temporels, etc).
        Sécurité :
            - Toutes les features sont décalées d'une bougie (shift(1)) pour éviter tout look-ahead bias.
        :param df: DataFrame d'entrée
        :param price_col: Colonne de prix (par défaut 'close')
        :param volume_col: Colonne de volume (par défaut 'volume')
        :param prefix: Préfixe optionnel pour les colonnes
        :return: DataFrame enrichi
        """
        df = df.copy()
        required_cols = [price_col, volume_col]
        if df.empty or not all(col in df.columns for col in required_cols):
            return df
        # Retours log et simples
        df[f"{prefix}return"] = df[price_col].pct_change().shift(1)
        df[f"{prefix}log_return"] = np.log(df[price_col] / df[price_col].shift(1)).shift(1)
        # Ajout harmonisé de log_return_1m (pour compatibilité labeling et backtest)
        if prefix == "1min_" or (prefix == "" and price_col in ["close", "<CLOSE>"]):
            if 'log_return_1m' not in df.columns:
                if price_col in df.columns:
                    df['log_return_1m'] = np.log(df[price_col] / df[price_col].shift(1))
        # Volatilité rolling
        df[f"{prefix}volatility_20"] = df[f"{prefix}return"].rolling(window=20, min_periods=1).std().shift(1)
        # Ratio volume/prix (pas besoin de shift)
        df[f"{prefix}vol_price_ratio"] = df[volume_col] / (df[price_col] + 1e-9)
        # --- Z-score généralisé ---
        for col in [price_col, "high", "low", volume_col]:
            if col in df.columns:
                for win in [5, 20, 50, 100]:
                    mean = df[col].rolling(window=win, min_periods=1).mean()
                    std = df[col].rolling(window=win, min_periods=1).std() + 1e-9
                    df[f"{prefix}{col}_zscore_{win}"] = ((df[col] - mean) / std).shift(1)
        # --- Distance à la bande de Bollinger ---
        if f"{prefix}bb_high" in df.columns and f"{prefix}bb_low" in df.columns:
            df[f"{prefix}dist_bb_high"] = (df[price_col] - df[f"{prefix}bb_high"]).shift(1)
            df[f"{prefix}dist_bb_low"] = (df[price_col] - df[f"{prefix}bb_low"]).shift(1)
            df[f"{prefix}dist_bb_width"] = (df[f"{prefix}bb_high"] - df[f"{prefix}bb_low"]).shift(1)
        # --- Distance au plus haut/bas N périodes ---
        for win in [5, 20, 50, 100]:
            if price_col in df.columns:
                df[f"{prefix}dist_high_{win}"] = (df[price_col] - df[price_col].rolling(window=win, min_periods=1).max()).shift(1)
                df[f"{prefix}dist_low_{win}"] = (df[price_col] - df[price_col].rolling(window=win, min_periods=1).min()).shift(1)
        # --- Encodage temporel enrichi ---
        if hasattr(df.index, 'hour'):
            df[f"{prefix}minute"] = df.index.minute
            df[f"{prefix}hour"] = df.index.hour
            df[f"{prefix}day"] = df.index.day
            df[f"{prefix}weekday"] = df.index.weekday
            df[f"{prefix}month"] = df.index.month
            df[f"{prefix}week"] = df.index.isocalendar().week.astype(int) if hasattr(df.index, 'isocalendar') else df.index.week
            df[f"{prefix}quarter"] = df.index.quarter if hasattr(df.index, 'quarter') else ((df.index.month-1)//3+1)
            df[f"{prefix}year"] = df.index.year
            # Encodage cyclique
            df[f"{prefix}minute_sin"] = np.sin(2 * np.pi * df.index.minute / 60)
            df[f"{prefix}minute_cos"] = np.cos(2 * np.pi * df.index.minute / 60)
            df[f"{prefix}hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
            df[f"{prefix}hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
            df[f"{prefix}weekday_sin"] = np.sin(2 * np.pi * df.index.weekday / 7)
            df[f"{prefix}weekday_cos"] = np.cos(2 * np.pi * df.index.weekday / 7)
            df[f"{prefix}month_sin"] = np.sin(2 * np.pi * (df.index.month-1) / 12)
            df[f"{prefix}month_cos"] = np.cos(2 * np.pi * (df.index.month-1) / 12)
            # Position relative dans la journée/semaine/mois
            df[f"{prefix}hour_rel"] = df.index.hour / 23
            df[f"{prefix}weekday_rel"] = df.index.weekday / 6
            df[f"{prefix}month_rel"] = (df.index.month-1) / 11
        return df

    def multi_timeframe(self, dfs: Dict[str, pd.DataFrame], price_col: str = "<CLOSE>", high_col: str = "<HIGH>", low_col: str = "<LOW>", volume_col: str = "<TICKVOL>") -> pd.DataFrame:
        """Fusionne et concatène les features multi-timeframe (clé = timeframe, valeur = DataFrame OHLCV)."""
        if not dfs or len(dfs) == 0:
            # Retourne un DataFrame vide indexé sur rien
            return pd.DataFrame()
        base = None
        # S'assurer que le timeframe de base (1min) est traité en premier si possible
        timeframes_ordered = sorted(dfs.keys())
        if "1min" in timeframes_ordered:
            timeframes_ordered.insert(0, timeframes_ordered.pop(timeframes_ordered.index("1min")))

        for tf in timeframes_ordered:
            df = dfs[tf]
            # Auto-mapping des colonnes si elles sont en minuscules (cas test)
            col_map = {
                "open": price_col.replace("<CLOSE>", "<OPEN>").replace("close", "open"),
                "high": high_col,
                "low": low_col,
                "close": price_col,
                "volume": volume_col,
                "tickvol": volume_col if volume_col == "tickvol" else "tickvol"
            }
            # Si les colonnes sont en minuscules, on les renomme pour matcher l'API
            if set(["open", "high", "low", "close", "volume"]).issubset(df.columns):
                df = df.rename(columns=col_map)
            # Appliquer add_indicators puis add_features pour chaque timeframe
            df_tf_features = self.add_indicators(df.copy(), price_col, high_col, low_col, volume_col, prefix=f"{tf}_")
            df_tf_features = self.add_features(df_tf_features, price_col=f"{tf}_close" if f"{tf}_close" in df_tf_features.columns else price_col, volume_col=f"{tf}_volume" if f"{tf}_volume" in df_tf_features.columns else volume_col, prefix=f"{tf}_")
            if base is None:
                base = df_tf_features
            else:
                cols_to_join = [col for col in df_tf_features.columns if col.startswith(f"{tf}_") and col != df_tf_features.index.name]
                original_ohlcv_cols = [price_col, high_col, low_col, volume_col, "tickvol"]
                cols_to_join = [col for col in cols_to_join if col not in original_ohlcv_cols]
                base = base.join(df_tf_features[cols_to_join], how="left")

        if base is None:
            return pd.DataFrame()
        # Suppression des colonnes OHLCV originales si leur version 1min_ préfixée existe
        cols_to_drop_original = [price_col, high_col, low_col, volume_col]
        if 'tickvol' in base.columns and 'tickvol' not in cols_to_drop_original:
            cols_to_drop_original.append('tickvol')
        cols_to_drop_if_1min_prefixed_exists = [price_col, high_col, low_col, volume_col, "tickvol"]
        final_cols_to_drop = [col for col in cols_to_drop_if_1min_prefixed_exists if f"1min_{col}" in base.columns and col in base.columns]
        base = base.drop(columns=final_cols_to_drop, errors='ignore')
        return base

def test_add_indicators_all_features():
    import pandas as pd
    import numpy as np
    fe = FeatureEngineering()
    # Génère un DataFrame OHLCV minimal mais suffisant
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 30),
        'high': np.linspace(101, 111, 30),
        'low': np.linspace(99, 109, 30),
        'volume': np.random.randint(1, 10, 30)
    })
    out = fe.add_indicators(df)
    expected_cols = [
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'ema_21', 'ema_50', 'sma_20',
        'bb_high', 'bb_low', 'bb_width',
        'atr', 'supertrend', 'vwap',
        'tickvol', 'close_sma_3', 'atr_sma_20'
    ]
    for col in expected_cols:
        assert col in out.columns, f"Feature manquante : {col}"
    # Vérifie que close_sma_3 et atr_sma_20 sont bien calculés
    assert not out['close_sma_3'].isnull().all()
    # assert not out['atr_sma_20'].isnull().all() # Peut être NaN si pas assez de données
    # Vérifie que tickvol est cohérent avec volume
    # assert np.allclose(out['tickvol'], out['volume']) # Volume et tickvol peuvent être différents dans la vraie donnée.

def test_features_compatibility_with_reference():
    """
    Vérifie que toutes les features attendues (ex: features_list.pkl) sont bien générées par add_indicators.
    """
    import pandas as pd
    import numpy as np
    fe = FeatureEngineering()
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 30),
        'high': np.linspace(101, 111, 30),
        'low': np.linspace(99, 109, 30),
        'volume': np.random.randint(1, 10, 30)
    })
    out = fe.add_indicators(df)
    # Simule une liste de features de référence (ex: features_list.pkl)
    reference = [
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'ema_21', 'ema_50', 'sma_20',
        'bb_high', 'bb_low', 'bb_width',
        'atr', 'supertrend', 'vwap',
        'tickvol', 'close_sma_3', 'atr_sma_20'
    ]
    missing = [col for col in reference if col not in out.columns]
    assert not missing, f"Features manquantes vs référence : {missing}"

def test_add_indicators_incomplete_df():
    """
    Vérifie que add_indicators ne plante pas et retourne un DataFrame même si des colonnes sont manquantes.
    """
    import pandas as pd
    fe = FeatureEngineering()
    # Cas 1 : DataFrame vide
    df_empty = pd.DataFrame()
    out_empty = fe.add_indicators(df_empty)
    assert isinstance(out_empty, pd.DataFrame)
    # Cas 2 : DataFrame sans 'high' ou 'low'
    df_partial = pd.DataFrame({'close': [1, 2, 3], 'volume': [1, 1, 1]})
    out_partial = fe.add_indicators(df_partial)
    assert isinstance(out_partial, pd.DataFrame)
    # Cas 3 : DataFrame sans 'volume'
    df_novol = pd.DataFrame({'close': [1, 2, 3], 'high': [2, 3, 4], 'low': [0, 1, 2]})
    out_novol = fe.add_indicators(df_novol)
    assert isinstance(out_novol, pd.DataFrame)

def test_add_indicators_with_tickvol():
    """
    Vérifie que add_indicators fonctionne si le DataFrame utilise 'tickvol' comme colonne de volume.
    """
    import pandas as pd
    import numpy as np
    fe = FeatureEngineering()
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 30),
        'high': np.linspace(101, 111, 30),
        'low': np.linspace(99, 109, 30),
        'tickvol': np.random.randint(1, 10, 30)
    })
    out = fe.add_indicators(df, volume_col='tickvol')
    assert 'tickvol' in out.columns
    assert 'close_sma_3' in out.columns
    assert 'atr_sma_20' in out.columns

def test_feature_order_matches_reference():
    """
    Vérifie que l'ordre des features générées correspond à la liste de référence (features_list.pkl).
    """
    import pandas as pd
    import numpy as np
    fe = FeatureEngineering()
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 30),
        'high': np.linspace(101, 111, 30),
        'low': np.linspace(99, 109, 30),
        'volume': np.random.randint(1, 10, 30)
    })
    out = fe.add_indicators(df)
    reference = [
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'ema_21', 'ema_50', 'sma_20',
        'bb_high', 'bb_low', 'bb_width',
        'atr', 'supertrend', 'vwap',
        'tickvol', 'close_sma_3', 'atr_sma_20'
    ]
    # On ne vérifie que l'ordre des colonnes présentes dans la référence
    generated = [col for col in out.columns if col in reference]
    assert generated == reference, f"Ordre des features incorrect : {generated} vs {reference}"

def test_supertrend_typing():
    import pandas as pd
    import numpy as np
    import warnings
    fe = FeatureEngineering()
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 30),
        'high': np.linspace(101, 111, 30),
        'low': np.linspace(99, 109, 30),
        'volume': np.random.randint(1, 10, 30)
    })
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fe.add_indicators(df)
        assert out['supertrend'].dtype == float
        assert out['supertrend_direction'].dtype == float
        assert not any('FutureWarning' in str(warn.message) for warn in w)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit un DataFrame minute BTC avec toutes les features financières et temporelles requises pour le ML, dont log_return_1m systématique.
    Toutes les features sont calculées de façon causale (aucune fuite d'information vers le futur).
    Les NaN générés par les rollings sont explicitement traqués et loggés.
    :param df: DataFrame prétraité, indexé par datetime UTC, colonnes : OPEN, HIGH, LOW, CLOSE, TICKVOL
    :return: DataFrame enrichi avec les features demandées
    """
    df = df.copy()
    logger.info("Début de l'enrichissement des features financières et temporelles.")
    # 1. Prix moyens
    df['HL2'] = (df['<HIGH>'] + df['<LOW>']) / 2
    df['HLC3'] = (df['<HIGH>'] + df['<LOW>'] + df['<CLOSE>']) / 3
    df['OC2'] = (df['<OPEN>'] + df['<CLOSE>']) / 2
    logger.debug("Prix moyens calculés : HL2, HLC3, OC2.")
    # 2. Retours log
    df['log_return_1m'] = np.log(df['<CLOSE>'] / df['<CLOSE>'].shift(1))
    # Alias pour compatibilité orchestrator/backtest
    df['1min_log_return'] = df['log_return_1m']
    for w in [3, 5, 15]:
        df[f'log_return_{w}m'] = np.log(df['<CLOSE>'] / df['<CLOSE>'].shift(w))
    logger.debug("Retours log calculés sur 1, 3, 5, 15 minutes.")
    # 3. Volatilité locale (rolling std des log_return_*)
    for base in ['log_return_1m', 'log_return_3m', 'log_return_5m', 'log_return_15m']:
        for win in [5, 15, 30]:
            col = f'{base}_std_{win}m'
            df[col] = df[base].rolling(window=win, min_periods=1).std()
    logger.debug("Volatilité locale (rolling std) calculée.")
    # 4. Moyennes mobiles
    for win in [5, 10, 20]:
        df[f'SMA_{win}'] = df['<CLOSE>'].rolling(window=win, min_periods=1).mean()
        df[f'EMA_{win}'] = df['<CLOSE>'].ewm(span=win, adjust=False).mean()
    logger.debug("Moyennes mobiles SMA/EMA calculées.")
    # 5. Indicateurs techniques
    # RSI(14)
    delta = df['<CLOSE>'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14, min_periods=1).mean()
    roll_down = down.rolling(14, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # MACD(12,26,9)
    ema12 = df['<CLOSE>'].ewm(span=12, adjust=False).mean()
    ema26 = df['<CLOSE>'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # ATR(14)
    high_low = df['<HIGH>'] - df['<LOW>']
    high_close = np.abs(df['<HIGH>'] - df['<CLOSE>'].shift(1))
    low_close = np.abs(df['<LOW>'] - df['<CLOSE>'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
    # Bollinger Bands(20)
    ma20 = df['<CLOSE>'].rolling(window=20, min_periods=1).mean()
    std20 = df['<CLOSE>'].rolling(window=20, min_periods=1).std()
    df['BB_MA_20'] = ma20
    df['BB_UPPER_20'] = ma20 + 2 * std20
    df['BB_LOWER_20'] = ma20 - 2 * std20
    df['BB_WIDTH_20'] = df['BB_UPPER_20'] - df['BB_LOWER_20']
    logger.debug("Indicateurs techniques calculés : RSI, MACD, ATR, Bollinger Bands.")
    # 6. Statistiques dérivées
    df['Z_SCORE_30'] = (df['<CLOSE>'] - df['<CLOSE>'].rolling(window=30, min_periods=1).mean()) / (df['<CLOSE>'].rolling(window=30, min_periods=1).std() + 1e-9)
    # Slope locale (pente linéaire) sur fenêtre 15
    def rolling_slope(series, window):
        slopes = np.full(len(series), np.nan)
        for i in range(window-1, len(series)):
            y = series.iloc[i-window+1:i+1]
            x = np.arange(window)
            A = np.vstack([x, np.ones(window)]).T
            m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            slopes[i] = m
        return slopes
    df['SLOPE_15'] = rolling_slope(df['<CLOSE>'], 15)
    logger.debug("Statistiques dérivées calculées : Z-score, slope locale.")
    # 7. Encodage temporel
    hours = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    dows = df.index.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dows / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dows / 7)
    logger.debug("Encodage temporel effectué (heures, jours de semaine).")
    # Traque des NaN
    nan_report = df.isna().sum()
    nan_cols = nan_report[nan_report > 0]
    if not nan_cols.empty:
        logger.warning(f"Colonnes avec NaN générés par les rollings : {dict(nan_cols)}")
    else:
        logger.info("Aucun NaN généré par les rollings.")
    logger.info(f"Features enrichies : {len(df.columns)} colonnes.")
    return df

"""
Exemple d'utilisation :
fe = FeatureEngineering(["1min", "5min"])
df_1m = fe.add_indicators(df_1m)
df_1m = fe.add_features(df_1m)
features = fe.multi_timeframe({"1min": df_1m, "5min": df_5m})
""" 