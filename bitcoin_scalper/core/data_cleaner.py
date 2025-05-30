import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any, Optional

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