import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GridSearchCV
try:
    import optuna
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

# Importer la fonction early_stopping au lieu de la classe
from lightgbm import early_stopping

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
try:
    from tensorflow import keras
    _HAS_KERAS = True
except ImportError:
    _HAS_KERAS = False

import os
import matplotlib.pyplot as plt
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

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
    random_state: int = 42,
    algo: str = 'catboost',
    cat_features: Optional[list] = None,
    **kwargs
) -> object:
    """
    Entraîne un modèle ML (CatBoost, XGBoost, DNN) avec tuning d'hyperparamètres et early stopping.
    :param X_train: Features d'entraînement
    :param y_train: Labels d'entraînement
    :param X_val: Features de validation
    :param y_val: Labels de validation
    :param method: 'optuna' (par défaut) ou 'grid'
    :param early_stopping_rounds: Nombre de rounds pour l'early stopping
    :param random_state: Seed de reproductibilité
    :param algo: 'catboost', 'xgboost', 'dnn_torch', 'dnn_keras'
    :param cat_features: Liste des colonnes catégorielles (indices ou noms)
    :return: Modèle entraîné
    """
    if set(np.unique(y_train)) - {-1, 0, 1}:
        logger.error("Classes inattendues dans y_train. Attendu : -1, 0, 1")
        raise ValueError("Classes inattendues dans y_train")
    if X_train.shape[0] != y_train.shape[0] or X_val.shape[0] != y_val.shape[0]:
        logger.error("Shape incohérent entre X et y")
        raise ValueError("Shape incohérent entre X et y")
    logger.info(f"Démarrage de l'entraînement {algo.upper()} ({method})")
    if algo == 'catboost':
        param_grid = {
            'depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [100, 200],
            'l2_leaf_reg': [1, 3, 5],
            'random_seed': [random_state],
            'loss_function': ['MultiClass']
        }
        best_params = None
        if method == 'grid':
            model = CatBoostClassifier(loss_function='MultiClass', random_seed=random_state, verbose=0)
            gs = GridSearchCV(model, param_grid, scoring='f1_macro', cv=3, verbose=0)
            gs.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), early_stopping_rounds=early_stopping_rounds)
            best_params = gs.best_params_
            logger.info(f"Meilleurs hyperparamètres (grid) : {best_params}")
        elif method == 'optuna':
            if not _HAS_OPTUNA:
                logger.error("Optuna n'est pas installé.")
                raise ImportError("Optuna n'est pas installé.")
            def objective(trial):
                params = {
                    'depth': trial.suggest_categorical('depth', [3, 5, 7]),
                    'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1]),
                    'iterations': trial.suggest_categorical('iterations', [100, 200]),
                    'l2_leaf_reg': trial.suggest_categorical('l2_leaf_reg', [1, 3, 5]),
                    'random_seed': random_state,
                    'loss_function': 'MultiClass',
                    'verbose': 0
                }
                model = CatBoostClassifier(**params)
                model.fit(
                    X_train, y_train,
                    cat_features=cat_features,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=early_stopping_rounds
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
        model = CatBoostClassifier(loss_function='MultiClass', random_seed=random_state, verbose=0, **(best_params or {}))
        model.fit(
            X_full, y_full,
            cat_features=cat_features,
            eval_set=(X_val, y_val),
            early_stopping_rounds=early_stopping_rounds
        )
        logger.info("Modèle CatBoost entraîné avec succès.")
        return model
    elif algo == 'xgboost':
        if not _HAS_XGBOOST:
            logger.error("xgboost n'est pas installé.")
            raise ImportError("xgboost n'est pas installé.")
        # Mapping des labels [-1, 0, 1] -> [0, 1, 2] si besoin
        label_map = None
        if set(np.unique(y_train)) == {-1, 0, 1}:
            label_map = {-1: 0, 0: 1, 1: 2}
            y_train = y_train.map(label_map)
            y_val = y_val.map(label_map)
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200],
            'subsample': [0.7, 0.9, 1.0],
        }
        best_params = None
        if method == 'grid':
            model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=random_state)
            gs = GridSearchCV(model, param_grid, scoring='f1_macro', cv=3, verbose=0)
            gs.fit(X_train, y_train)
            best_params = gs.best_params_
            logger.info(f"Meilleurs hyperparamètres (grid) : {best_params}")
        elif method == 'optuna':
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7]),
                    'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1]),
                    'n_estimators': trial.suggest_categorical('n_estimators', [100, 200]),
                    'subsample': trial.suggest_categorical('subsample', [0.7, 0.9, 1.0]),
                    'objective': 'multi:softmax',
                    'num_class': 3,
                    'random_state': random_state
                }
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=False)
                y_pred = model.predict(X_val)
                from sklearn.metrics import f1_score
                return f1_score(y_val, y_pred, average='macro')
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20, show_progress_bar=False)
            best_params = study.best_params
            logger.info(f"Meilleurs hyperparamètres (optuna) : {best_params}")
        else:
            logger.error(f"Méthode de tuning inconnue : {method}")
            raise ValueError(f"Méthode de tuning inconnue : {method}")
        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])
        model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=random_state, **(best_params or {}))
        model.fit(X_full, y_full, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=False)
        logger.info("Modèle XGBoost entraîné avec succès.")
        return model
    elif algo == 'dnn_torch':
        if not _HAS_TORCH:
            logger.error("PyTorch n'est pas installé.")
            raise ImportError("PyTorch n'est pas installé.")
        # Réseau simple pour classification 3 classes
        class SimpleDNN(nn.Module):
            def __init__(self, input_dim, hidden_dim=64, output_dim=3):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
            def forward(self, x):
                return self.net(x)
        Xtr = torch.tensor(X_train.values, dtype=torch.float32)
        ytr = torch.tensor(y_train.values, dtype=torch.long)
        Xv = torch.tensor(X_val.values, dtype=torch.float32)
        yv = torch.tensor(y_val.values, dtype=torch.long)
        model = SimpleDNN(Xtr.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        batch_size = 32
        for epoch in range(30):
            model.train()
            for i in range(0, len(Xtr), batch_size):
                xb = Xtr[i:i+batch_size]
                yb = ytr[i:i+batch_size]
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
            # Early stopping simple (non optimal)
            model.eval()
            with torch.no_grad():
                val_out = model(Xv)
                val_loss = criterion(val_out, yv)
            if val_loss.item() < 0.1:
                break
        logger.info("DNN PyTorch entraîné.")
        return model
    elif algo == 'dnn_keras':
        if not _HAS_KERAS:
            logger.error("Keras/TensorFlow n'est pas installé.")
            raise ImportError("Keras/TensorFlow n'est pas installé.")
        from tensorflow.keras import layers, models
        model = models.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), verbose=0)
        logger.info("DNN Keras entraîné.")
        return model
    else:
        logger.error(f"Algo ML inconnu : {algo}")
        raise ValueError(f"Algo ML inconnu : {algo}")

def predict(model: CatBoostClassifier, X_test: pd.DataFrame) -> np.ndarray:
    """
    Prédit les classes sur X_test à partir d'un modèle CatBoost entraîné.

    :param model: Modèle CatBoostClassifier entraîné
    :param X_test: Features de test
    :return: np.ndarray des classes prédites
    """
    if not hasattr(model, 'predict'):
        logger.error("L'objet passé n'est pas un modèle CatBoost valide.")
        raise ValueError("L'objet passé n'est pas un modèle CatBoost valide.")
    preds = model.predict(X_test)
    logger.info(f"Prédiction effectuée sur {len(X_test)} échantillons.")
    return preds

# Test unitaire minimal pour chaque algo

def test_train_model_xgboost():
    if not _HAS_XGBOOST:
        print("xgboost non installé : test ignoré.")
        return
    X = pd.DataFrame(np.random.randn(100, 5))
    y = pd.Series(np.random.choice([0, 1, 2], 100))
    Xtr, Xv = X.iloc[:80], X.iloc[80:]
    ytr, yv = y.iloc[:80], y.iloc[80:]
    model = train_model(Xtr, ytr, Xv, yv, algo='xgboost', method='grid')
    assert hasattr(model, 'predict'), "XGBoost : modèle non entraîné."
    print("Test XGBoost OK.")

def test_train_model_dnn_torch():
    if not _HAS_TORCH:
        print("PyTorch non installé : test ignoré.")
        return
    X = pd.DataFrame(np.random.randn(100, 5))
    y = pd.Series(np.random.choice([-1, 0, 1], 100))
    Xtr, Xv = X.iloc[:80], X.iloc[80:]
    ytr, yv = y.iloc[:80], y.iloc[80:]
    model = train_model(Xtr, ytr, Xv, yv, algo='dnn_torch')
    assert hasattr(model, 'forward'), "DNN Torch : modèle non entraîné."
    print("Test DNN Torch OK.")

def test_train_model_dnn_keras():
    if not _HAS_KERAS:
        print("Keras non installé : test ignoré.")
        return
    X = pd.DataFrame(np.random.randn(100, 5))
    y = pd.Series(np.random.choice([0, 1, 2], 100))
    Xtr, Xv = X.iloc[:80], X.iloc[80:]
    ytr, yv = y.iloc[:80], y.iloc[80:]
    model = train_model(Xtr, ytr, Xv, yv, algo='dnn_keras')
    assert hasattr(model, 'predict'), "DNN Keras : modèle non entraîné."
    print("Test DNN Keras OK.")

def analyze_feature_importance(model, X: pd.DataFrame, out_dir: str = "data/features", prefix: str = "") -> None:
    """
    Analyse et exporte l'importance des features d'un modèle ML (CatBoost/XGBoost) :
    - Importance classique (gain, split, etc.)
    - Valeurs SHAP si possible
    - Génère des rapports PNG/HTML dans out_dir
    :param model: Modèle ML entraîné (CatBoost ou XGBoost)
    :param X: DataFrame des features utilisées pour l'entraînement
    :param out_dir: Dossier de sortie pour les rapports
    :param prefix: Préfixe pour les fichiers exportés
    """
    os.makedirs(out_dir, exist_ok=True)
    # Importance classique
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X.columns
        plt.figure(figsize=(10, 6))
        idx = np.argsort(importances)[::-1]
        plt.bar(range(len(importances)), importances[idx])
        plt.xticks(range(len(importances)), features[idx], rotation=90)
        plt.title("Feature Importance (gain/split)")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{prefix}feature_importance.png")
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Feature importance PNG exporté : {out_path}")
    # SHAP
    if _HAS_SHAP:
        try:
            explainer = None
            if hasattr(model, 'booster_'):
                explainer = shap.TreeExplainer(model)
            elif hasattr(model, 'get_booster'):
                explainer = shap.TreeExplainer(model.get_booster())
            if explainer is not None:
                shap_values = explainer.shap_values(X)
                plt.figure()
                shap.summary_plot(shap_values, X, show=False)
                out_path = os.path.join(out_dir, f"{prefix}shap_summary.png")
                plt.savefig(out_path, bbox_inches='tight')
                plt.close()
                logger.info(f"SHAP summary plot exporté : {out_path}")
                # Bar plot
                plt.figure()
                shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                out_path = os.path.join(out_dir, f"{prefix}shap_bar.png")
                plt.savefig(out_path, bbox_inches='tight')
                plt.close()
                logger.info(f"SHAP bar plot exporté : {out_path}")
        except Exception as e:
            logger.warning(f"Erreur lors de l'analyse SHAP : {e}")
    else:
        logger.info("SHAP non installé : analyse SHAP ignorée.")

def select_features_by_importance(model, X: pd.DataFrame, top_n: int = 20, use_shap: bool = False) -> list:
    """
    Sélectionne les features les plus importantes selon le modèle (gain/split ou SHAP).
    :param model: Modèle ML entraîné (CatBoost/XGBoost)
    :param X: DataFrame des features
    :param top_n: Nombre de features à conserver
    :param use_shap: Si True, utilise SHAP si disponible, sinon importance classique
    :return: Liste des noms de features à conserver
    """
    importances = None
    if use_shap and _HAS_SHAP:
        try:
            explainer = None
            if hasattr(model, 'booster_'):
                explainer = shap.TreeExplainer(model)
            elif hasattr(model, 'get_booster'):
                explainer = shap.TreeExplainer(model.get_booster())
            if explainer is not None:
                shap_values = explainer.shap_values(X)
                # Moyenne absolue des valeurs SHAP sur toutes les classes/features
                if isinstance(shap_values, list):
                    shap_vals = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                else:
                    shap_vals = np.abs(shap_values).mean(axis=0)
                importances = shap_vals
        except Exception as e:
            logger.warning(f"Erreur SHAP pour la sélection de features : {e}")
    if importances is None and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    if importances is None:
        logger.error("Impossible de récupérer l'importance des features.")
        return list(X.columns)
    idx = np.argsort(importances)[::-1][:top_n]
    selected = list(X.columns[idx])
    logger.info(f"Features sélectionnées (top {top_n}) : {selected}")
    return selected

from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor

def train_qvalue_model(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    Y_val: pd.DataFrame,
    method: str = 'optuna',
    early_stopping_rounds: int = 20,
    random_state: int = 42,
    algo: str = 'catboost',
    cat_features: Optional[list] = None,
    **kwargs
) -> object:
    """
    Entraîne un modèle de régression multi-sortie pour Q-values (CatBoost, XGBoost, MLP).
    :param X_train: Features d'entraînement
    :param Y_train: Q-values d'entraînement (DataFrame, colonnes = actions)
    :param X_val: Features de validation
    :param Y_val: Q-values de validation
    :param method: 'optuna' ou 'grid'
    :param early_stopping_rounds: Early stopping
    :param random_state: Seed
    :param algo: 'catboost', 'xgboost', 'mlp'
    :param cat_features: Colonnes catégorielles
    :return: Modèle entraîné (MultiOutputRegressor)
    """
    if algo == 'catboost':
        base = CatBoostRegressor(loss_function='RMSE', random_seed=random_state, verbose=0)
    elif algo == 'xgboost':
        import xgboost as xgb
        base = xgb.XGBRegressor(random_state=random_state, verbosity=0)
    elif algo == 'mlp':
        from sklearn.neural_network import MLPRegressor
        base = MLPRegressor(hidden_layer_sizes=(64, 32), random_state=random_state, max_iter=200)
    else:
        raise ValueError(f"Algo de régression Q-value non supporté : {algo}")
    model = MultiOutputRegressor(base)
    model.fit(X_train, Y_train)
    return model 