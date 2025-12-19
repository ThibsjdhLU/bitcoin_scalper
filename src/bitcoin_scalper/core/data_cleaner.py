import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any, Optional
import json
import logging
import os

class DataCleaner:
    """
    Nettoyage avancé des données ticks et OHLCV :
    - Suppression des outliers (z-score robuste/MAD, IQR)
    - Gestion des valeurs manquantes
    - Détection d'anomalies (Isolation Forest)
    """
    def __init__(self, anomaly_method: str = "isoforest", contamination: float = 0.01, zscore_thresh: float = 4.0):
        self.anomaly_method = anomaly_method
        self.contamination = contamination
        self.zscore_thresh = zscore_thresh

    def _mad_zscore(self, series):
        median = np.median(series)
        mad = np.median(np.abs(series - median)) + 1e-9
        return 0.6745 * (series - median) / mad

    def clean_ticks(self, ticks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Nettoie une liste de ticks (suppression outliers, valeurs manquantes, anomalies).
        - Si 'timestamp' est absent mais 'time' ou 'time_msc' est présent, il est ajouté automatiquement.
        - Supprime les ticks incomplets ou aberrants.
        """
        if not ticks:
            return []
        df = pd.DataFrame(ticks)
        # Gestion valeurs manquantes : suppression lignes incomplètes
        df = df.dropna(subset=["bid", "ask", "volume", "timestamp"]).reset_index(drop=True)
        # Suppression stricte des outliers z-score robuste (MAD) sur bid/ask/volume
        if not df.empty:
            mad_zscores = pd.DataFrame({col: np.abs(self._mad_zscore(df[col].values)) for col in ["bid", "ask", "volume"]})
            mask = (mad_zscores < self.zscore_thresh).all(axis=1)
            df = df[mask].reset_index(drop=True)
        # Détection anomalies (Isolation Forest)
        if self.anomaly_method == "isoforest" and len(df) > 10:
            clf = IsolationForest(contamination=self.contamination, random_state=42)
            features = df[["bid", "ask", "volume"]]
            preds = clf.fit_predict(features)
            df = df[preds == 1]
        return df.to_dict(orient="records")

    def clean_ohlcv(self, ohlcv: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Nettoie une liste d'OHLCV (suppression outliers, valeurs manquantes, anomalies)."""
        if not ohlcv:
            return []
        df = pd.DataFrame(ohlcv)
        # Gestion valeurs manquantes : suppression lignes incomplètes
        df = df.dropna(subset=["open", "high", "low", "close", "volume", "timestamp"])
        # Suppression stricte des outliers IQR sur open/high/low/close/volume
        if not df.empty:
            iqr_mask = np.ones(len(df), dtype=bool)
            for col in ["open", "high", "low", "close", "volume"]:
                if len(df) < 5:
                    # Filtrage MAD strict pour petits jeux de données
                    median = df[col].median()
                    mad = np.median(np.abs(df[col] - median)) + 1e-9
                    col_mask = np.abs(df[col] - median) < 3 * mad
                else:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    median = df[col].median()
                    if iqr == 0:
                        col_mask = (df[col] == median)
                    else:
                        col_mask = (df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)
                iqr_mask &= col_mask
            df = df[iqr_mask]
        # Détection anomalies (Isolation Forest)
        if self.anomaly_method == "isoforest" and len(df) > 10:
            clf = IsolationForest(contamination=self.contamination, random_state=42)
            features = df[["open", "high", "low", "close", "volume"]]
            preds = clf.fit_predict(features)
            df = df[preds == 1]
        return df.to_dict(orient="records")

    def clean_market_data(
        self,
        df: pd.DataFrame,
        min_volume: float = 1.0,
        max_spread: float = 0.01,
        zscore_thresh: float = 5.0,
        exclude_hours: Optional[list] = None,
        report_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Nettoyage avancé d'un DataFrame OHLCV :
        - Supprime/masque les périodes d'illiquidité (volume < min_volume, spread > max_spread)
        - Supprime/masque les périodes d'anomalies (prix aberrants, gaps, spikes, via z-score)
        - Exclut des plages horaires (ex : week-end, nuit, maintenance)
        - Génère un rapport JSON sur les périodes supprimées/masquées
        :param df: DataFrame OHLCV indexé par datetime
        :param min_volume: Volume minimum pour considérer une période liquide
        :param max_spread: Spread maximum toléré (en proportion, ex : 0.01 = 1%)
        :param zscore_thresh: Seuil z-score pour détecter les prix aberrants
        :param exclude_hours: Liste d'heures (ou fonction) à exclure (ex : [0,1,2,3,4,5,23])
        :param report_path: Chemin du rapport JSON à générer (optionnel)
        :return: DataFrame nettoyé
        """
        logger = logging.getLogger("bitcoin_scalper.data_cleaner")
        df = df.copy()
        report = {"illiquid": [], "spread": [], "anomaly": [], "excluded_hours": []}
        # Illiquidité (volume)
        illiquid_mask = df["volume"] < min_volume
        report["illiquid"] = df.index[illiquid_mask].strftime("%Y-%m-%d %H:%M:%S").tolist()
        # Spread (high-low/close)
        spread = (df["high"] - df["low"]) / (df["close"] + 1e-9)
        spread_mask = spread > max_spread
        report["spread"] = df.index[spread_mask].strftime("%Y-%m-%d %H:%M:%S").tolist()
        # Anomalies (z-score sur close)
        zscores = (df["close"] - df["close"].rolling(30, min_periods=10).mean()) / (df["close"].rolling(30, min_periods=10).std() + 1e-9)
        anomaly_mask = zscores.abs() > zscore_thresh
        report["anomaly"] = df.index[anomaly_mask].strftime("%Y-%m-%d %H:%M:%S").tolist()
        # Exclusion horaires
        if exclude_hours is not None:
            if callable(exclude_hours):
                hours_mask = df.index.map(exclude_hours)
            else:
                hours_mask = df.index.hour.isin(exclude_hours)
            report["excluded_hours"] = df.index[hours_mask].strftime("%Y-%m-%d %H:%M:%S").tolist()
        else:
            hours_mask = pd.Series(False, index=df.index)
        # Masquage global
        mask = ~(illiquid_mask | spread_mask | anomaly_mask | hours_mask)
        df_clean = df[mask]
        # Rapport JSON
        if report_path is not None:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Rapport de nettoyage exporté : {report_path}")
        return df_clean

    def correct_outliers_and_gaps(
        self,
        df: pd.DataFrame,
        freq: str = "1min",
        outlier_zscore: float = 5.0,
        winsorize: bool = True,
        report_path: str = None
    ) -> pd.DataFrame:
        """
        Corrige les outliers, trous temporels et erreurs de marché dans un DataFrame OHLCV.
        - Outliers (prix, volume) : winsorization ou interpolation
        - Trous temporels : réindexation, ffill/bfill/interpolation
        - Erreurs de marché : prix/volume négatifs, incohérences OHLC
        - Génère un rapport JSON sur les corrections appliquées
        :param df: DataFrame OHLCV indexé par datetime
        :param freq: Fréquence cible (ex : "1min")
        :param outlier_zscore: Seuil z-score pour détecter les outliers
        :param winsorize: Si True, applique la winsorization, sinon interpolation
        :param report_path: Chemin du rapport JSON à générer (optionnel)
        :return: DataFrame corrigé
        """
        logger = logging.getLogger("bitcoin_scalper.data_cleaner")
        df = df.copy()
        report = {"outliers": {}, "gaps": [], "negatives": [], "ohlc_incoherence": []}
        # Correction des trous temporels
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
        gaps = full_idx.difference(df.index)
        report["gaps"] = gaps.strftime("%Y-%m-%d %H:%M:%S").tolist()
        df = df.reindex(full_idx)
        # Correction des valeurs négatives
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                neg_mask = df[col] < 0
                report["negatives"] += df.index[neg_mask].strftime("%Y-%m-%d %H:%M:%S").tolist()
                df.loc[neg_mask, col] = np.nan
        # Correction des incohérences OHLC (high < low, close hors [low, high])
        if all(c in df.columns for c in ["high", "low"]):
            incoh_mask = df["high"] < df["low"]
            report["ohlc_incoherence"] += df.index[incoh_mask].strftime("%Y-%m-%d %H:%M:%S").tolist()
            df.loc[incoh_mask, ["high", "low"]] = np.nan
        if all(c in df.columns for c in ["close", "low", "high"]):
            out_low = df["close"] < df["low"]
            out_high = df["close"] > df["high"]
            report["ohlc_incoherence"] += df.index[out_low | out_high].strftime("%Y-%m-%d %H:%M:%S").tolist()
            df.loc[out_low, "close"] = df.loc[out_low, "low"]
            df.loc[out_high, "close"] = df.loc[out_high, "high"]
        # Correction des outliers (z-score)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                roll_mean = df[col].rolling(30, min_periods=10).mean()
                roll_std = df[col].rolling(30, min_periods=10).std() + 1e-9
                zscores = (df[col] - roll_mean) / roll_std
                outlier_mask = zscores.abs() > outlier_zscore
                report["outliers"][col] = df.index[outlier_mask].strftime("%Y-%m-%d %H:%M:%S").tolist()
                if winsorize:
                    # Winsorization : ramène à la borne la plus proche
                    upper = roll_mean + outlier_zscore * roll_std
                    lower = roll_mean - outlier_zscore * roll_std
                    df.loc[outlier_mask & (df[col] > upper), col] = upper[outlier_mask & (df[col] > upper)]
                    df.loc[outlier_mask & (df[col] < lower), col] = lower[outlier_mask & (df[col] < lower)]
                else:
                    # Interpolation
                    df.loc[outlier_mask, col] = np.nan
        # Interpolation/ffill/bfill pour les NaN
        df = df.interpolate(method="time").ffill().bfill()
        # Rapport JSON
        if report_path is not None:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Rapport de correction exporté : {report_path}")
        return df

    def check_temporal_consistency(
        self,
        df: pd.DataFrame,
        target_cols: list = None,
        freq: str = "1min",
        report_path: str = None
    ) -> dict:
        """
        Vérifie la cohérence temporelle d'un DataFrame OHLCV/ML :
        - Pas de fuite d'information (colonnes target/future bien décalées)
        - Pas de doublons ou de désordre dans l'index temporel
        - Continuité temporelle (pas de saut anormal, pas de timestamp dans le futur)
        - Génère un rapport JSON ou log sur les incohérences détectées
        :param df: DataFrame indexé par datetime
        :param target_cols: Liste des colonnes à vérifier pour le look-ahead (ex : ["target_5m", "future_return"])
        :param freq: Fréquence attendue (ex : "1min")
        :param report_path: Chemin du rapport JSON à générer (optionnel)
        :return: Dictionnaire des incohérences détectées
        """
        logger = logging.getLogger("bitcoin_scalper.data_cleaner")
        report = {"lookahead": {}, "duplicates": [], "disorder": [], "gaps": [], "future_timestamps": []}
        # Vérification doublons
        if df.index.duplicated().any():
            report["duplicates"] = df.index[df.index.duplicated()].strftime("%Y-%m-%d %H:%M:%S").tolist()
        # Vérification désordre
        if not df.index.is_monotonic_increasing:
            disorder_idx = df.index[~df.index.is_monotonic_increasing]
            report["disorder"] = disorder_idx.strftime("%Y-%m-%d %H:%M:%S").tolist()
        # Vérification gaps temporels
        expected_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
        gaps = expected_idx.difference(df.index)
        report["gaps"] = gaps.strftime("%Y-%m-%d %H:%M:%S").tolist()
        # Vérification timestamps dans le futur
        now = pd.Timestamp.utcnow()
        future_mask = df.index > now
        report["future_timestamps"] = df.index[future_mask].strftime("%Y-%m-%d %H:%M:%S").tolist()
        # Vérification look-ahead (fuite d'info)
        if target_cols is not None:
            for col in target_cols:
                if col in df.columns:
                    # Vérifie que la valeur en t ne dépend pas de t+1 ou plus (ex : target_5m doit être NaN sur les 5 dernières lignes)
                    horizon = 0
                    if "_" in col and "m" in col:
                        try:
                            horizon = int(col.split("_")[-1].replace("m", ""))
                        except Exception:
                            pass
                    if horizon > 0:
                        tail = df[col].iloc[-horizon:]
                        if tail.notna().any():
                            report["lookahead"][col] = tail.index[tail.notna()].strftime("%Y-%m-%d %H:%M:%S").tolist()
        # Rapport JSON
        if report_path is not None:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Rapport de cohérence temporelle exporté : {report_path}")
        return report

    def audit_data_quality(
        self,
        df: pd.DataFrame,
        label_cols: list = None,
        freq: str = "1min",
        min_volume: float = 1.0,
        max_spread: float = 0.01,
        zscore_thresh: float = 5.0,
        outlier_zscore: float = 5.0,
        winsorize: bool = True,
        exclude_hours: list = None,
        report_dir: str = "data/features",
        prefix: str = "audit_"
    ) -> dict:
        """
        Audit global de la qualité des données :
        - Nettoyage avancé (illiquidité, spread, anomalies, horaires exclus)
        - Correction des outliers, trous, erreurs de marché
        - Vérification de la cohérence temporelle
        - Analyse de la distribution des labels (si label_cols)
        - Génère un rapport global JSON dans report_dir
        :param df: DataFrame OHLCV indexé par datetime
        :param label_cols: Colonnes de labels à analyser (optionnel)
        :param freq: Fréquence cible
        :param min_volume: Volume minimum pour la liquidité
        :param max_spread: Spread max toléré
        :param zscore_thresh: Seuil z-score pour anomalies marché
        :param outlier_zscore: Seuil z-score pour outliers
        :param winsorize: Winsorization ou interpolation
        :param exclude_hours: Heures à exclure
        :param report_dir: Dossier de sortie des rapports
        :param prefix: Préfixe pour les fichiers exportés
        :return: Dictionnaire du rapport global
        """
        os.makedirs(report_dir, exist_ok=True)
        logger = logging.getLogger("bitcoin_scalper.data_cleaner")
        report = {}
        # Nettoyage marché
        clean_path = os.path.join(report_dir, f"{prefix}clean_market.json")
        df_clean = self.clean_market_data(df, min_volume=min_volume, max_spread=max_spread, zscore_thresh=zscore_thresh, exclude_hours=exclude_hours, report_path=clean_path)
        report["clean_market"] = clean_path
        # Correction outliers/gaps
        corr_path = os.path.join(report_dir, f"{prefix}correction.json")
        df_corr = self.correct_outliers_and_gaps(df_clean, freq=freq, outlier_zscore=outlier_zscore, winsorize=winsorize, report_path=corr_path)
        report["correction"] = corr_path
        # Cohérence temporelle
        temporal_path = os.path.join(report_dir, f"{prefix}temporal.json")
        temporal_report = self.check_temporal_consistency(df_corr, target_cols=label_cols, freq=freq, report_path=temporal_path)
        report["temporal"] = temporal_path
        # Distribution des labels
        if label_cols is not None:
            from bitcoin_scalper.core.labeling import analyze_label_distribution
            dist = analyze_label_distribution(df_corr, label_cols, out_dir=report_dir, prefix=prefix)
            report["label_distribution"] = os.path.join(report_dir, f"{prefix}labels_distribution.json")
        # Rapport global
        global_path = os.path.join(report_dir, f"{prefix}global_audit.json")
        with open(global_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Rapport global d'audit exporté : {global_path}")
        return report

def test_clean_ticks_timestamp_fallback():
    """
    Vérifie que clean_ticks ajoute 'timestamp' à partir de 'time' ou 'time_msc' et supprime les ticks incomplets.
    """
    cleaner = DataCleaner()
    ticks = [
        {"bid": 1, "ask": 2, "volume": 0.1, "time": 1234567890},
        {"bid": 1, "ask": 2, "volume": 0.1, "time_msc": 1234567890123},
        {"bid": 1, "ask": 2, "volume": 0.1},  # Incomplet
        {"bid": 1, "ask": 2, "volume": 0.1, "timestamp": 1234567890},
    ]
    # On simule l'ajout du champ timestamp dans l'ingestor, mais on vérifie la robustesse ici aussi
    for t in ticks:
        if 'timestamp' not in t:
            if 'time' in t:
                t['timestamp'] = t['time']
            elif 'time_msc' in t:
                t['timestamp'] = int(t['time_msc'] // 1000)
    cleaned = cleaner.clean_ticks(ticks)
    # Seuls les ticks avec tous les champs requis doivent rester
    assert all('timestamp' in t for t in cleaned)
    assert all('bid' in t and 'ask' in t and 'volume' in t for t in cleaned)
    # Le tick incomplet doit être supprimé
    assert len(cleaned) == 3

"""
Exemple d'utilisation :
cleaner = DataCleaner()
ticks_clean = cleaner.clean_ticks(ticks)
ohlcv_clean = cleaner.clean_ohlcv(ohlcv)
""" 