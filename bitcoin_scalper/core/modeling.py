import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
try:
    import optuna
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

# Importer la fonction early_stopping au lieu de la classe
from lightgbm import early_stopping

logger = logging.getLogger("bitcoin_scalper.modeling")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method: str = 'optuna',
    early_stopping_rounds: int = 20,
    random_state: int = 42
) -> LGBMClassifier:
    """
    Entraîne un modèle LightGBM multiclasse avec tuning d'hyperparamètres et early stopping.

    :param X_train: Features d'entraînement
    :param y_train: Labels d'entraînement
    :param X_val: Features de validation
    :param y_val: Labels de validation
    :param method: 'optuna' (par défaut) ou 'grid'
    :param early_stopping_rounds: Nombre de rounds pour l'early stopping
    :param random_state: Seed de reproductibilité
    :return: Modèle LGBMClassifier entraîné
    """
    if set(np.unique(y_train)) - {-1, 0, 1}:
        logger.error("Classes inattendues dans y_train. Attendu : -1, 0, 1")
        raise ValueError("Classes inattendues dans y_train")
    if X_train.shape[0] != y_train.shape[0] or X_val.shape[0] != y_val.shape[0]:
        logger.error("Shape incohérent entre X et y")
        raise ValueError("Shape incohérent entre X et y")
    logger.info(f"Démarrage de l'entraînement LightGBM ({method})")
    param_grid = {
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'feature_fraction': [0.7, 0.9, 1.0],
        'n_estimators': [100, 200]
    }
    best_params = None
    if method == 'grid':
        model = LGBMClassifier(objective='multiclass', random_state=random_state)
        gs = GridSearchCV(model, param_grid, scoring='f1_macro', cv=3, verbose=0)
        gs.fit(X_train, y_train)
        best_params = gs.best_params_
        logger.info(f"Meilleurs hyperparamètres (grid) : {best_params}")
    elif method == 'optuna':
        if not _HAS_OPTUNA:
            logger.error("Optuna n'est pas installé.")
            raise ImportError("Optuna n'est pas installé.")
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_categorical('num_leaves', [15, 31, 63]),
                'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1]),
                'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7]),
                'feature_fraction': trial.suggest_categorical('feature_fraction', [0.7, 0.9, 1.0]),
                'n_estimators': trial.suggest_categorical('n_estimators', [100, 200]),
                'objective': 'multiclass',
                'random_state': random_state
            }
            model = LGBMClassifier(**params)
            # Utiliser la fonction early_stopping pour créer le callback
            early_stopping_callback = early_stopping(early_stopping_rounds, verbose=False)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='multi_logloss',
                callbacks=[early_stopping_callback]
            )
            from sklearn.metrics import f1_score
            y_pred = model.predict(X_val)
            return f1_score(y_val, y_pred, average='macro')
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        best_params = study.best_params
        logger.info(f"Meilleurs hyperparamètres (optuna) : {best_params}")
    else:
        logger.error(f"Méthode de tuning inconnue : {method}")
        raise ValueError(f"Méthode de tuning inconnue : {method}")
    # Entraînement final sur train+val
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    model = LGBMClassifier(objective='multiclass', random_state=random_state, **(best_params or {}))
    # Utiliser la fonction early_stopping pour créer le callback pour l'entraînement final
    early_stopping_final_callback = early_stopping(early_stopping_rounds, verbose=False)
    model.fit(
        X_full, y_full,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[early_stopping_final_callback]
    )
    logger.info("Modèle entraîné avec succès.")
    return model

def predict(model: LGBMClassifier, X_test: pd.DataFrame) -> np.ndarray:
    """
    Prédit les classes sur X_test à partir d'un modèle LightGBM entraîné.

    :param model: Modèle LGBMClassifier entraîné
    :param X_test: Features de test
    :return: np.ndarray des classes prédites
    """
    if not hasattr(model, 'predict'):
        logger.error("L'objet passé n'est pas un modèle LightGBM valide.")
        raise ValueError("L'objet passé n'est pas un modèle LightGBM valide.")
    preds = model.predict(X_test)
    logger.info(f"Prédiction effectuée sur {len(X_test)} échantillons.")
    return preds 