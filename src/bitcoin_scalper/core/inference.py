import pandas as pd
import numpy as np
import logging
from typing import Optional, Union
from bitcoin_scalper.core.export import load_objects
from bitcoin_scalper.core.feature_engineering import add_features

logger = logging.getLogger("bitcoin_scalper.inference")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

def inference(
    df_minute: pd.DataFrame,
    path_prefix: str = "model"
) -> pd.Series:
    """
    Prédit le signal de trading (3 classes) à partir d'un DataFrame minute brut.

    - Charge le pipeline et le modèle LightGBM exportés
    - Applique le pipeline de features (causalité stricte)
    - Prédit le signal à chaque minute (labels -1, 0, 1)
    - Logging structuré, gestion robuste des erreurs

    :param df_minute: DataFrame minute brut (colonnes : <OPEN>, <HIGH>, <LOW>, <CLOSE>, <TICKVOL>, index datetime)
    :param path_prefix: Préfixe de chemin pour les objets exportés (par défaut 'model')
    :return: pd.Series des signaux prédits, indexé comme df_minute
    """
    try:
        logger.info(f"Chargement des objets d'inférence depuis {path_prefix}")
        objects = load_objects(path_prefix)
        model = objects['model']
        pipeline = objects['pipeline']
        scaler = objects['scaler']
        # 1. Feature engineering (causalité stricte)
        df_feat = add_features(df_minute)
        # 2. Application pipeline/scaler si existant
        X = df_feat.copy()
        if pipeline is not None:
            X = pipeline.transform(X)
        if scaler is not None:
            X = scaler.transform(X)
        # 3. Prédiction
        y_pred = model.predict(X)
        logger.info(f"Prédiction réalisée sur {len(X)} lignes.")
        return pd.Series(y_pred, index=df_feat.index, name='signal')
    except Exception as e:
        logger.error(f"Erreur d'inférence : {e}")
        raise 