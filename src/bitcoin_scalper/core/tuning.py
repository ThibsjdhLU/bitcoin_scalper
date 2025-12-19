import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, Any, Optional
from sklearn.model_selection import ParameterGrid, ParameterSampler, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

logger = logging.getLogger("bitcoin_scalper.tuning")
logger.setLevel(logging.INFO)

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import optuna
except ImportError:
    optuna = None

def tune_model_hyperparams(
    df: pd.DataFrame,
    label_col: str,
    model_type: str = "catboost",
    param_grid: Optional[Dict[str, Any]] = None,
    method: str = "grid",
    n_iter: int = 20,
    cv_params: Optional[Dict[str, Any]] = None,
    out_dir: str = "tuning_reports",
    random_state: int = 42,
    cat_features: Optional[list] = None
) -> Dict[str, Any]:
    """
    Tuning avancé des hyperparamètres (grid/random/optuna) avec TimeSeriesSplit.
    :param df: DataFrame features+labels indexé par datetime
    :param label_col: colonne cible à prédire
    :param model_type: "catboost" (par défaut) ou "xgboost"
    :param param_grid: dict d'espace de recherche (ex : {"depth": [3,5], ...})
    :param method: "grid", "random", "optuna"
    :param n_iter: nombre d'itérations (random/optuna)
    :param cv_params: dict pour TimeSeriesSplit
    :param out_dir: dossier de sortie des rapports
    :param random_state: seed
    :param cat_features: Liste des colonnes catégorielles (indices ou noms)
    :return: dict rapport global
    """
    os.makedirs(out_dir, exist_ok=True)
    cv_params = cv_params or {"n_splits": 3}
    tscv = TimeSeriesSplit(**cv_params)
    features = [c for c in df.columns if c != label_col]
    X, y = df[features], df[label_col]
    results = []
    best_score = -np.inf
    best_params = None
    best_model = None
    def evaluate(params):
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            if model_type == "catboost":
                model = CatBoostClassifier(random_seed=random_state, loss_function='MultiClass', verbose=0, **params)
                model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), early_stopping_rounds=10)
            elif model_type == "xgboost":
                model = xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric="logloss", **params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
            else:
                raise ValueError(f"Modèle non supporté : {model_type}")
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average="macro")
            scores.append(score)
        return np.mean(scores)
    # Méthode grid/random
    if method in ["grid", "random"]:
        if not param_grid:
            raise ValueError("param_grid doit être spécifié pour grid/random search")
        if method == "grid":
            param_list = list(ParameterGrid(param_grid))
        else:
            param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state))
        for i, params in enumerate(param_list):
            score = evaluate(params)
            results.append({"params": params, "score": score})
            if score > best_score:
                best_score = score
                best_params = params
        logger.info(f"Tuning {method} : {len(param_list)} essais, meilleur score={best_score:.4f}")
    # Méthode optuna
    elif method == "optuna":
        if optuna is None:
            raise ImportError("optuna n'est pas installé")
        def objective(trial):
            params = {k: trial.suggest_categorical(k, v) if isinstance(v, list) else trial.suggest_float(k, v[0], v[1]) for k, v in param_grid.items()}
            return evaluate(params)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_iter)
        best_params = study.best_params
        best_score = study.best_value
        logger.info(f"Tuning optuna : {n_iter} essais, meilleur score={best_score:.4f}")
    else:
        raise ValueError(f"Méthode de tuning inconnue : {method}")
    # Réentraînement sur tout le train avec les meilleurs params
    if model_type == "catboost":
        best_model = CatBoostClassifier(random_seed=random_state, loss_function='MultiClass', verbose=0, **best_params)
        best_model.fit(X, y, cat_features=cat_features)
    elif model_type == "xgboost":
        best_model = xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric="logloss", **best_params)
        best_model.fit(X, y)
    # Export des résultats
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, "tuning_results.csv"), index=False)
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump({"best_params": best_params, "best_score": best_score}, f, indent=2)
    # Courbe des scores
    plt.figure(figsize=(6,4))
    plt.plot([r["score"] for r in results], marker="o")
    plt.title("Score F1 macro par essai")
    plt.xlabel("Essai")
    plt.ylabel("F1 macro")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tuning_scores.png"))
    plt.close()
    # Importance des features
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        imp_df = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
        imp_df.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
        plt.figure(figsize=(8,4))
        plt.bar(imp_df["feature"], imp_df["importance"])
        plt.title("Feature importance (tuning)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "feature_importance.png"))
        plt.close()
    # Rapport global
    report = {
        "tuning_results": os.path.join(out_dir, "tuning_results.csv"),
        "best_params": os.path.join(out_dir, "best_params.json"),
        "tuning_scores": os.path.join(out_dir, "tuning_scores.png"),
        "feature_importance": os.path.join(out_dir, "feature_importance.csv"),
        "feature_importance_png": os.path.join(out_dir, "feature_importance.png")
    }
    with open(os.path.join(out_dir, "tuning_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Rapport tuning exporté : {os.path.join(out_dir, 'tuning_report.json')}")
    return report 