import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger("bitcoin_scalper.splitting")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

def split_dataset(
    df: pd.DataFrame,
    method: str = 'fixed',
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    purge_window: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Découpe un DataFrame chronologique en trois sous-ensembles (train, validation, test) sans fuite temporelle.

    - Découpage strictement chronologique (aucune intersection)
    - Proportions ou bornes explicites
    - Option purge_window pour exclure les points aux frontières (stratégie purged)
    - Logging structuré des bornes et tailles

    :param df: DataFrame équilibré, indexé par datetime UTC
    :param method: 'fixed' (par défaut) ou 'purged_kfold'
    :param train_frac: Fraction du train (par défaut 0.7)
    :param val_frac: Fraction de la validation (par défaut 0.15)
    :param test_frac: Fraction du test (par défaut 0.15)
    :param purge_window: Nombre de points à exclure aux frontières (None = pas de purge)
    :return: (train_df, val_df, test_df)
    """
    df = df.copy()
    df = df.sort_index()
    n = len(df)
    if method == 'fixed':
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        n_test = n - n_train - n_val
        train_end = n_train
        val_end = n_train + n_val
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        logger.info(f"Découpage FIXED : train={train_df.shape[0]}, val={val_df.shape[0]}, test={test_df.shape[0]}")
        # Purge optionnelle
        if purge_window:
            logger.info(f"Purge de {purge_window} points aux frontières train/val/test")
            if len(val_df) > 2*purge_window:
                val_df = val_df.iloc[purge_window:-purge_window]
            if len(test_df) > purge_window:
                test_df = test_df.iloc[purge_window:]
        # Exclusion explicite des points aux frontières (overlap de rolling)
        train_df = train_df[~train_df.index.isin(val_df.index)]
        val_df = val_df[~val_df.index.isin(test_df.index)]
        return train_df, val_df, test_df
    elif method == 'purged_kfold':
        # Découpage en 3 folds chronologiques avec purge_window entre chaque
        if purge_window is None:
            raise ValueError("purge_window doit être spécifié pour purged_kfold")
        fold_size = n // 3
        idx1 = fold_size
        idx2 = 2 * fold_size
        train_df = df.iloc[:idx1]
        val_df = df.iloc[idx1+purge_window:idx2]
        test_df = df.iloc[idx2+purge_window:]
        logger.info(f"Découpage PURGED_KFOLD : train={train_df.shape[0]}, val={val_df.shape[0]}, test={test_df.shape[0]}, purge_window={purge_window}")
        return train_df, val_df, test_df
    else:
        logger.error(f"Méthode de split inconnue : {method}")
        raise ValueError(f"Méthode de split inconnue : {method}") 