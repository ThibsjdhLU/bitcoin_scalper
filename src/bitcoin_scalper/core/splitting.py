import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
import json

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

def time_series_cv(
    df: pd.DataFrame,
    n_splits: int = 5
):
    """
    Génère des indices de validation croisée temporelle (TimeSeriesSplit) pour un DataFrame chronologique.
    :param df: DataFrame indexé par datetime
    :param n_splits: Nombre de splits (par défaut 5)
    :return: Générateur de tuples (train_idx, test_idx)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(df):
        yield train_idx, test_idx

def test_time_series_cv():
    """
    Test unitaire : vérifie que time_series_cv génère bien des splits chronologiques sans fuite.
    """
    import pandas as pd
    df = pd.DataFrame({'a': range(100)})
    splits = list(time_series_cv(df, n_splits=4))
    assert len(splits) == 4, f"Nombre de splits incorrect : {len(splits)}"
    for train_idx, test_idx in splits:
        assert max(train_idx) < min(test_idx), "Fuite temporelle détectée !"
    print("Test TimeSeriesSplit OK.")

def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
    train_frac: Optional[float] = None,
    val_frac: Optional[float] = None,
    test_frac: Optional[float] = None,
    horizon: int = 0,
    report_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporel robuste d'un DataFrame indexé par datetime :
    - Split par bornes explicites (dates) ou proportions
    - Vérifie l'absence de fuite sur les labels futurs (multi-horizon)
    - Génère un rapport JSON (dates, tailles, indices, sécurité)
    :param df: DataFrame indexé par datetime
    :param train_start/end, val_start/end, test_start/end: bornes explicites (str ou None)
    :param train_frac, val_frac, test_frac: proportions (si bornes non spécifiées)
    :param horizon: nombre de pas à exclure en fin de train/val pour éviter la fuite (ex : horizon max des labels)
    :param report_path: chemin du rapport JSON à générer (optionnel)
    :return: (train_df, val_df, test_df)
    """
    df = df.copy().sort_index()
    n = len(df)
    idx = df.index
    # Split par bornes explicites
    if train_start or train_end or val_start or val_end or test_start or test_end:
        train_mask = (idx >= train_start) if train_start else pd.Series(True, index=idx)
        train_mask &= (idx < train_end) if train_end else True
        val_mask = (idx >= val_start) if val_start else pd.Series(True, index=idx)
        val_mask &= (idx < val_end) if val_end else True
        test_mask = (idx >= test_start) if test_start else pd.Series(True, index=idx)
        test_mask &= (idx <= test_end) if test_end else True
        train_df = df[train_mask]
        val_df = df[val_mask]
        test_df = df[test_mask]
    # Split par proportions
    else:
        if not (train_frac and val_frac and test_frac):
            raise ValueError("Spécifier soit les bornes, soit les proportions train/val/test.")
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        n_test = n - n_train - n_val
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train+n_val]
        test_df = df.iloc[n_train+n_val:]
    # Exclusion des derniers points pour éviter la fuite (multi-horizon)
    if horizon > 0:
        if len(train_df) > horizon:
            train_df = train_df.iloc[:-horizon]
        if len(val_df) > horizon:
            val_df = val_df.iloc[:-horizon]
    # Rapport JSON
    report = {
        "train": {"start": str(train_df.index[0]) if not train_df.empty else None, "end": str(train_df.index[-1]) if not train_df.empty else None, "size": len(train_df)},
        "val": {"start": str(val_df.index[0]) if not val_df.empty else None, "end": str(val_df.index[-1]) if not val_df.empty else None, "size": len(val_df)},
        "test": {"start": str(test_df.index[0]) if not test_df.empty else None, "end": str(test_df.index[-1]) if not test_df.empty else None, "size": len(test_df)},
        "horizon_exclusion": horizon,
        "total": n,
        "overlap": bool(set(train_df.index) & set(val_df.index) or set(val_df.index) & set(test_df.index)),
        "leakage_risk": horizon > 0 and (len(train_df) < horizon or len(val_df) < horizon)
    }
    logger.info(f"Split temporel : train={report['train']}, val={report['val']}, test={report['test']}, horizon={horizon}")
    if report_path:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Rapport de split exporté : {report_path}")
    return train_df, val_df, test_df

def generate_time_series_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    report_path: Optional[str] = None
) -> list:
    """
    Génère les folds de validation croisée temporelle (TimeSeriesSplit).
    :param df: DataFrame indexé par datetime
    :param n_splits: nombre de splits
    :param report_path: chemin du rapport JSON à générer (optionnel)
    :return: liste de tuples (train_idx, val_idx)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    report = {"folds": []}
    for i, (train_idx, val_idx) in enumerate(tscv.split(df)):
        folds.append((train_idx, val_idx))
        report["folds"].append({
            "fold": i,
            "train_start": str(df.index[train_idx[0]]),
            "train_end": str(df.index[train_idx[-1]]),
            "val_start": str(df.index[val_idx[0]]),
            "val_end": str(df.index[val_idx[-1]]),
            "train_size": len(train_idx),
            "val_size": len(val_idx)
        })
    logger.info(f"Généré {n_splits} folds TimeSeriesSplit.")
    if report_path:
        import json
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Rapport de folds exporté : {report_path}")
    return folds

def generate_purged_kfold_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    purge_window: int = 5,
    report_path: Optional[str] = None
) -> list:
    """
    Génère les folds PurgedKFold (KFold chronologique avec fenêtre de purge anti-fuite).
    :param df: DataFrame indexé par datetime
    :param n_splits: nombre de splits
    :param purge_window: nombre de points à exclure entre train et val
    :param report_path: chemin du rapport JSON à générer (optionnel)
    :return: liste de tuples (train_idx, val_idx)
    """
    n = len(df)
    fold_sizes = [n // n_splits] * n_splits
    for i in range(n % n_splits):
        fold_sizes[i] += 1
    indices = np.arange(n)
    current = 0
    folds = []
    report = {"folds": []}
    for i, fold_size in enumerate(fold_sizes):
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        # Train = tout avant, tout après, avec purge_window
        train_idx = np.concatenate([
            indices[:max(start - purge_window, 0)],
            indices[min(stop + purge_window, n):]
        ])
        folds.append((train_idx, val_idx))
        report["folds"].append({
            "fold": i,
            "train_start": str(df.index[train_idx[0]]) if len(train_idx) else None,
            "train_end": str(df.index[train_idx[-1]]) if len(train_idx) else None,
            "val_start": str(df.index[val_idx[0]]),
            "val_end": str(df.index[val_idx[-1]]),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "purge_window": purge_window
        })
        current = stop
    logger.info(f"Généré {n_splits} folds PurgedKFold (purge_window={purge_window}).")
    if report_path:
        import json
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Rapport de folds exporté : {report_path}")
    return folds 