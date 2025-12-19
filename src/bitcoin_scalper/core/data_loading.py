import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger("bitcoin_scalper.data_loading")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

class InvalidFrequencyError(Exception):
    """
    Exception levée si la fréquence temporelle du DataFrame n'est pas strictement de 1 minute.
    """
    pass

def load_minute_csv(
    csv_path: str,
    date_col: str = "<DATE>",
    time_col: str = "<TIME>",
    open_col: str = "<OPEN>",
    high_col: str = "<HIGH>",
    low_col: str = "<LOW>",
    close_col: str = "<CLOSE>",
    tickvol_col: str = "<TICKVOL>",
    vol_col: str = "<VOL>",
    spread_col: str = "<SPREAD>",
    report_missing: Optional[str] = None,
    fill_method: Optional[str] = None
) -> pd.DataFrame:
    """
    Charge et nettoie un fichier CSV minute pour le pipeline ML BTC.

    - Fusionne <DATE> et <TIME> en datetime indexé UTC
    - Supprime <VOL> et <SPREAD>
    - Convertit les types (float32 pour prix/volumes, int32 pour timestamp)
    - Supprime doublons et valeurs nulles (sauf si fill_method)
    - Contrôle la fréquence temporelle (1 min stricte)
    - Si report_missing est fourni, génère un rapport des trous temporels
    - Si fill_method est spécifié, remplit les NaN avant le contrôle de fréquence

    :param csv_path: Chemin du fichier CSV
    :param report_missing: Chemin du fichier de rapport des trous (ou None pour désactiver)
    :param fill_method: None (strict), 'ffill', 'bfill', 'interpolate'
    :return: DataFrame propre, indexé par datetime UTC, colonnes : OPEN, HIGH, LOW, CLOSE, TICKVOL
    """
    logger.info(f"Chargement du fichier CSV : {csv_path}")
    # Détection automatique du séparateur (tabulation ou virgule)
    try:
        df = pd.read_csv(csv_path, dtype=str, sep=None, engine='python')
    except Exception as e:
        logger.error(f"Erreur lors du chargement du CSV : {e}")
        raise
    logger.debug(f"Colonnes détectées : {list(df.columns)}")
    # Fusion <DATE> + <TIME> en datetime UTC
    if date_col not in df.columns or time_col not in df.columns:
        logger.error(f"Colonnes {date_col} et/ou {time_col} absentes du CSV.")
        raise ValueError(f"Colonnes {date_col} et/ou {time_col} absentes du CSV.")
    dt_str = df[date_col].astype(str) + ' ' + df[time_col].astype(str)
    # Inférence automatique du format datetime (supporte HH:MM ou HH:MM:SS)
    df['datetime'] = pd.to_datetime(dt_str, utc=True, errors='raise')
    df = df.set_index('datetime')
    df = df.sort_index()
    # Suppression des colonnes inutiles
    for col in [vol_col, spread_col]:
        if col in df.columns:
            df = df.drop(columns=col)
            logger.debug(f"Colonne supprimée : {col}")
    # Conversion des types
    for col in [open_col, high_col, low_col, close_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
    if tickvol_col in df.columns:
        df[tickvol_col] = pd.to_numeric(df[tickvol_col], errors='coerce').astype(np.float32)
    # Suppression des colonnes d'origine <DATE>/<TIME>
    df = df.drop(columns=[date_col, time_col])
    # Suppression des doublons
    n_before = len(df)
    df = df[~df.index.duplicated(keep='first')]
    n_after = len(df)
    logger.info(f"Suppression de {n_before - n_after} lignes (doublons)")
    # Remplissage des NaN si demandé
    if fill_method is not None:
        logger.info(f"Remplissage des NaN avec la méthode : {fill_method}")
        if fill_method == 'ffill':
            df = df.ffill()
        elif fill_method == 'bfill':
            df = df.bfill()
        elif fill_method == 'interpolate':
            df = df.interpolate(method='time', limit_direction='both')
        else:
            logger.error(f"Méthode de remplissage inconnue : {fill_method}")
            raise ValueError(f"Méthode de remplissage inconnue : {fill_method}")
    else:
        df = df.dropna()
    # Contrôle strict de la fréquence temporelle (1 min)
    freq = df.index.to_series().diff().dropna()
    trous = freq[freq != pd.Timedelta(minutes=1)]
    if report_missing is not None:
        logger.warning(f"Génération d'un rapport des trous temporels : {report_missing}")
        with open(report_missing, 'w') as f:
            f.write(f"Nombre de trous : {len(trous)}\n")
            f.write("Timestamps des débuts de trous :\n")
            for ts, delta in trous.items():
                f.write(f"{ts} (écart : {delta})\n")
    if not (freq == pd.Timedelta(minutes=1)).all():
        logger.error("Fréquence temporelle non conforme (doit être 1 minute fixe)")
    # Sélection des colonnes finales
    final_cols = [open_col, high_col, low_col, close_col, tickvol_col]
    df = df[final_cols]
    logger.info(f"DataFrame final : {df.shape[0]} lignes, colonnes : {final_cols}")
    return df 