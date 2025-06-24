import pandas as pd
import numpy as np
from typing import Optional

def augment_rolling_jitter(
    df: pd.DataFrame,
    n_shifts: int = 3,
    jitter_seconds: int = 10,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Génère des versions augmentées du DataFrame en décalant les features (rolling) et en ajoutant un jitter aléatoire sur les timestamps.
    Chaque version augmentée reçoit un identifiant 'augmentation_id'.
    :param df: DataFrame d'origine indexé par datetime
    :param n_shifts: Nombre de décalages temporels à appliquer (1 à n_shifts)
    :param jitter_seconds: Amplitude maximale du jitter (en secondes)
    :param random_state: Graine aléatoire pour la reproductibilité
    :return: DataFrame concaténé (original + augmentations), avec colonne 'augmentation_id'
    """
    rng = np.random.default_rng(random_state)
    dfs = []
    # Version originale
    df_orig = df.copy()
    df_orig['augmentation_id'] = 0
    dfs.append(df_orig)
    for shift in range(1, n_shifts+1):
        df_shift = df.copy().shift(shift)
        df_shift = df_shift.iloc[shift:]  # On retire les premières lignes NaN
        # Jitter sur l'index
        jitter = rng.integers(-jitter_seconds, jitter_seconds+1, size=len(df_shift))
        new_index = df_shift.index + pd.to_timedelta(jitter, unit='s')
        df_shift.index = new_index
        df_shift['augmentation_id'] = shift
        dfs.append(df_shift)
    df_aug = pd.concat(dfs).sort_index()
    return df_aug 