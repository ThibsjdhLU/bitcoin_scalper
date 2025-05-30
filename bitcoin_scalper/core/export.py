import os
import pickle
import joblib
import logging
from typing import Any, Dict, Optional
from lightgbm import Booster, LGBMClassifier

logger = logging.getLogger("bitcoin_scalper.export")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

def save_objects(
    model: LGBMClassifier,
    pipeline: Optional[Any],
    encoders: Optional[Dict[str, Any]],
    scaler: Optional[Any],
    path_prefix: str
) -> None:
    """
    Sauvegarde tous les objets nécessaires à l'inférence (modèle, pipeline, encodages, scaler) avec validation post-sauvegarde.

    :param model: Modèle LightGBM entraîné
    :param pipeline: Pipeline de transformation (feature engineering, scaling, etc.)
    :param encoders: Dictionnaire d'encodages (labels, features)
    :param scaler: Scaler ou normalisateur (optionnel)
    :param path_prefix: Préfixe de chemin pour les fichiers sauvegardés
    """
    try:
        # Modèle LightGBM natif
        model_path = f"{path_prefix}_model.txt"
        model.booster_.save_model(model_path)
        logger.info(f"Modèle LightGBM sauvegardé : {model_path}")
        # Pipeline, encoders, scaler (pickle/joblib)
        for obj, name in zip([pipeline, encoders, scaler], ["pipeline", "encoders", "scaler"]):
            if obj is not None:
                obj_path = f"{path_prefix}_{name}.pkl"
                with open(obj_path, 'wb') as f:
                    pickle.dump(obj, f)
                logger.info(f"Objet {name} sauvegardé : {obj_path}")
        # Validation post-sauvegarde
        loaded = load_objects(path_prefix)
        assert loaded['model'] is not None, "Modèle non rechargé après sauvegarde"
        logger.info("Validation post-sauvegarde réussie.")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde : {e}")
        raise

def load_objects(path_prefix: str) -> Dict[str, Any]:
    """
    Charge tous les objets nécessaires à l'inférence à partir des fichiers sauvegardés.

    :param path_prefix: Préfixe de chemin utilisé lors de la sauvegarde
    :return: Dictionnaire {model, pipeline, encoders, scaler}
    """
    objects = {}
    try:
        # Modèle LightGBM natif
        model_path = f"{path_prefix}_model.txt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fichier modèle absent : {model_path}")
        model = LGBMClassifier()
        model._Booster = Booster(model_file=model_path)
        objects['model'] = model
        logger.info(f"Modèle LightGBM chargé : {model_path}")
        # Pipeline, encoders, scaler
        for name in ["pipeline", "encoders", "scaler"]:
            obj_path = f"{path_prefix}_{name}.pkl"
            if os.path.exists(obj_path):
                with open(obj_path, 'rb') as f:
                    objects[name] = pickle.load(f)
                logger.info(f"Objet {name} chargé : {obj_path}")
            else:
                objects[name] = None
        return objects
    except Exception as e:
        logger.error(f"Erreur lors du chargement : {e}")
        raise

def test_load_objects_model_absent():
    """
    Teste que FileNotFoundError est bien levée si le modèle est absent.
    """
    import pytest
    with pytest.raises(FileNotFoundError):
        load_objects("/tmp/model_inexistant")


def test_save_and_load_objects(tmp_path):
    """
    Teste la sauvegarde et le rechargement d'un modèle LightGBM minimal.
    """
    import numpy as np
    from lightgbm import LGBMClassifier
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    model = LGBMClassifier().fit(X, y)
    save_objects(model, None, None, None, str(tmp_path / "test_model"))
    loaded = load_objects(str(tmp_path / "test_model"))
    assert loaded["model"] is not None 