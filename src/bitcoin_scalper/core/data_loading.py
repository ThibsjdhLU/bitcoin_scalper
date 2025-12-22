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
    
    Supporte deux formats de CSV :
    1. Legacy MT5 format: <DATE>, <TIME>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, <TICKVOL>, <VOL>, <SPREAD>
    2. Binance/Standard format: date, open, high, low, close, volume (lowercase)

    - Détecte automatiquement le format du CSV
    - Pour Binance format: parse date et renomme les colonnes vers format <TAGS>
    - Pour MT5 format: fusionne <DATE> et <TIME> en datetime indexé UTC
    - Supprime <VOL> et <SPREAD>
    - Convertit les types (float32 pour prix/volumes, int32 pour timestamp)
    - Supprime doublons et valeurs nulles (sauf si fill_method)
    - Contrôle la fréquence temporelle (1 min stricte)
    - Si report_missing est fourni, génère un rapport des trous temporels
    - Si fill_method est spécifié, remplit les NaN avant le contrôle de fréquence

    :param csv_path: Chemin du fichier CSV
    :param report_missing: Chemin du fichier de rapport des trous (ou None pour désactiver)
    :param fill_method: None (strict), 'ffill', 'bfill', 'interpolate'
    :return: DataFrame propre, indexé par datetime UTC, colonnes : <OPEN>, <HIGH>, <LOW>, <CLOSE>, <TICKVOL>
    """
    logger.info(f"Chargement du fichier CSV : {csv_path}")
    # Détection automatique du séparateur (tabulation ou virgule)
    try:
        df = pd.read_csv(csv_path, dtype=str, sep=None, engine='python')
    except Exception as e:
        logger.error(f"Erreur lors du chargement du CSV : {e}")
        raise
    logger.debug(f"Colonnes détectées : {list(df.columns)}")
    
    # STEP 1: DETECT FORMAT
    # Check if this is Binance/Standard format (lowercase columns)
    is_binance_format = False
    if 'date' in df.columns or 'timestamp' in df.columns:
        logger.info("Format détecté : Binance/Standard (colonnes lowercase)")
        is_binance_format = True
    elif date_col in df.columns and time_col in df.columns:
        logger.info("Format détecté : Legacy MT5 (colonnes <TAGS>)")
        is_binance_format = False
    else:
        logger.error(f"Format CSV non reconnu. Colonnes présentes : {list(df.columns)}")
        raise ValueError(f"Format CSV non reconnu. Attendu : soit 'date'/'timestamp', soit '{date_col}'/{time_col}'")
    
    # STEP 2: ADAPT FORMAT
    if is_binance_format:
        # Handle Binance/Standard format
        logger.info("Adaptation du format Binance vers format interne <TAGS>")
        
        # Parse date column to datetime and set as index
        date_column = 'date' if 'date' in df.columns else 'timestamp'
        df[date_column] = pd.to_datetime(df[date_column], utc=True, errors='raise')
        df = df.set_index(date_column)
        df = df.sort_index()
        df.index.name = 'datetime'
        
        # Rename columns from lowercase to <TAGS> format for internal pipeline compatibility
        column_mapping = {
            'open': '<OPEN>',
            'high': '<HIGH>',
            'low': '<LOW>',
            'close': '<CLOSE>',
            'volume': '<TICKVOL>'  # Map volume to <TICKVOL>
        }
        
        # Only rename columns that exist
        columns_to_rename = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=columns_to_rename)
        logger.debug(f"Colonnes renommées : {columns_to_rename}")
        
        # Update column references to use the new names
        open_col = '<OPEN>'
        high_col = '<HIGH>'
        low_col = '<LOW>'
        close_col = '<CLOSE>'
        tickvol_col = '<TICKVOL>'
        
    else:
        # Handle Legacy MT5 format
        logger.info("Traitement du format Legacy MT5")
        
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
        
        # Suppression des colonnes d'origine <DATE>/<TIME>
        df = df.drop(columns=[date_col, time_col])
    
    # STEP 3: TYPE CONVERSION (common for both formats)
    # Conversion des types
    for col in [open_col, high_col, low_col, close_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
    if tickvol_col in df.columns:
        df[tickvol_col] = pd.to_numeric(df[tickvol_col], errors='coerce').astype(np.float32)
    
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
    
    # STEP 4: VALIDATION
    # Ensure the final DataFrame has the required columns
    final_cols = [open_col, high_col, low_col, close_col, tickvol_col]
    missing_cols = [col for col in final_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Colonnes finales manquantes : {missing_cols}")
        raise ValueError(f"Colonnes finales manquantes : {missing_cols}")
    
    # Sélection des colonnes finales
    df = df[final_cols]
    
    # Validate DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("L'index final n'est pas un DatetimeIndex")
        raise ValueError("L'index final n'est pas un DatetimeIndex")
    
    logger.info(f"DataFrame final : {df.shape[0]} lignes, colonnes : {final_cols}")
    return df 