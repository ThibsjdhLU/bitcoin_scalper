import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("bitcoin_scalper.labeling")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

def generate_labels(df: pd.DataFrame, horizon: int = 15, k: float = 0.5) -> pd.Series:
    """
    Génère une colonne de labels à 3 classes pour classification supervisée sur données financières minute.

    Le label est basé sur le mouvement futur du prix à un horizon donné, avec un seuil dynamique basé sur la volatilité locale.

    - 1 si future_return > +seuil
    - -1 si future_return < -seuil
    - 0 sinon

    :param df: DataFrame enrichi, indexé par datetime UTC, contenant au minimum la colonne <CLOSE> et log_return_1m
    :param horizon: Horizon de prédiction (en minutes, par défaut 15)
    :param k: Multiplicateur du rolling_std pour le seuil dynamique (par défaut 0.5)
    :return: pd.Series de labels indexée comme df, sans NaN
    :raises: ValueError si colonne <CLOSE> ou log_return_1m absente
    """
    if '<CLOSE>' not in df.columns or 'log_return_1m' not in df.columns:
        logger.error("Colonne <CLOSE> ou log_return_1m absente du DataFrame.")
        raise ValueError("Colonne <CLOSE> ou log_return_1m absente du DataFrame.")
    logger.info(f"Génération des labels à 3 classes (horizon={horizon}, k={k})")
    # Calcul du seuil dynamique (rolling std sur log_return_1m)
    rolling_std = df['log_return_1m'].rolling(window=30, min_periods=30).std()
    seuil = k * rolling_std
    # Calcul du future_return (log(CLOSE_{t+h} / CLOSE_t))
    future_close = df['<CLOSE>'].shift(-horizon)
    future_return = np.log(future_close / df['<CLOSE>'])
    # Attribution du label
    label = pd.Series(np.nan, index=df.index)
    label[future_return > seuil] = 1
    label[future_return < -seuil] = -1
    label[(future_return <= seuil) & (future_return >= -seuil)] = 0
    # Exclusion des lignes sans horizon suffisant ou sans rolling_std
    valid = (~seuil.isna()) & (~future_return.isna())
    label = label[valid]
    logger.info(f"Distribution des labels : {label.value_counts(dropna=True).to_dict()}")
    # Contrôle strict : aucune NaN, index aligné
    label = label.astype(int)
    return label 