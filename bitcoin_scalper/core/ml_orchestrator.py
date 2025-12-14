import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, Any, Optional
from bitcoin_scalper.core.splitting import temporal_train_val_test_split, generate_time_series_folds
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

logger = logging.getLogger("bitcoin_scalper.ml_orchestrator")
logger.setLevel(logging.INFO)

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None

def run_ml_pipeline(
    df: pd.DataFrame,
    label_col: str,
    model_type: str = "catboost",
    split_params: Optional[Dict[str, Any]] = None,
    cv_params: Optional[Dict[str, Any]] = None,
    out_dir: str = "ml_reports",
    random_state: int = 42,
    cat_features: Optional[list] = None
) -> Dict[str, Any]:
    """
    Orchestration complète du pipeline ML : split, folds, entraînement, validation, test, reporting.
    :param df: DataFrame features+labels indexé par datetime
    :param label_col: colonne cible à prédire
    :param model_type: "catboost" (par défaut) ou "xgboost"
    :param split_params: dict pour temporal_train_val_test_split
    :param cv_params: dict pour generate_time_series_folds
    :param out_dir: dossier de sortie des rapports
    :param random_state: seed pour la reproductibilité
    :param cat_features: Liste des colonnes catégorielles (indices ou noms)
    :return: dict rapport global
    """
    os.makedirs(out_dir, exist_ok=True)
    # 1. Split train/val/test
    split_params = split_params or {"train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15, "horizon": 0}
    train, val, test = temporal_train_val_test_split(df, **split_params, report_path=os.path.join(out_dir, "split_report.json"))
    # 2. Folds CV sur train+val
    cv_params = cv_params or {"n_splits": 5}
    folds = generate_time_series_folds(pd.concat([train, val]), **cv_params, report_path=os.path.join(out_dir, "cv_folds.json"))
    # 3. Préparation des features/labels
    features = [c for c in df.columns if c != label_col]
    X_train, y_train = train[features], train[label_col]
    X_val, y_val = val[features], val[label_col]
    X_test, y_test = test[features], test[label_col]
    # Diagnostic maximal et filtrage des colonnes non numériques
    for split_name, X in zip(['train', 'val', 'test'], [X_train, X_val, X_test]):
        logger.info(f"dtypes {split_name}:\n{X.dtypes}")
        non_num_cols = [col for col in X.columns if not (np.issubdtype(X[col].dtype, np.number))]
        if non_num_cols:
            logger.warning(f"Colonnes non numériques dans {split_name}: {non_num_cols}")
            for col in non_num_cols:
                v = X[col].iloc[0]
                logger.warning(f"  {col}: type={type(v)}, shape={getattr(v, 'shape', None)}, exemple={v}")
    # Filtrage automatique des colonnes non numériques (hors cat_features)
    allowed_cols = [col for col in features if np.issubdtype(df[col].dtype, np.number) or (cat_features and col in cat_features)]
    if set(allowed_cols) != set(features):
        logger.warning(f"Features non numériques supprimées du modèle: {set(features) - set(allowed_cols)}")
    features = allowed_cols
    X_train, X_val, X_test = train[features], val[features], test[features]
    logger.info(f"Features finales utilisées pour l'entraînement: {features}")
    # Suppression automatique des colonnes avec >10% de NaN dans le train
    nan_ratio_train = X_train.isna().mean()
    cols_to_drop = nan_ratio_train[nan_ratio_train > 0.1].index.tolist()
    if cols_to_drop:
        logger.warning(f"Colonnes supprimées du pipeline ML (>10% de NaN dans le train) : {cols_to_drop}")
        X_train = X_train.drop(columns=cols_to_drop)
        X_val = X_val.drop(columns=cols_to_drop, errors='ignore')
        X_test = X_test.drop(columns=cols_to_drop, errors='ignore')
        features = [col for col in features if col not in cols_to_drop]
        logger.info(f"Features finales après suppression des colonnes incomplètes : {features}")
    # Nettoyage final : conversion forcée en float
    for split_name, X in zip(['train', 'val', 'test'], [X_train, X_val, X_test]):
        X_converted = X.apply(pd.to_numeric, errors='coerce')
        nan_ratio = X_converted.isna().mean()
        bad_cols = nan_ratio[nan_ratio > 0.1].index.tolist()
        if bad_cols:
            for col in bad_cols:
                logger.error(f"Colonne {col} dans {split_name} : {nan_ratio[col]*100:.1f}% de NaN après conversion. Exemple avant conversion : {X[col].iloc[0]}")
            raise ValueError(f"Colonnes avec >10% de NaN après conversion dans {split_name} : {bad_cols}")
        if nan_ratio.any():
            logger.warning(f"Colonnes avec NaN après conversion dans {split_name} : {nan_ratio[nan_ratio>0].to_dict()}")
        if split_name == 'train':
            X_train = X_converted
        elif split_name == 'val':
            X_val = X_converted
        else:
            X_test = X_converted
    # Décalage du début du train à la première date où toutes les features sont valides
    first_valid_idx = X_train.dropna().index.min()
    if first_valid_idx is not None and X_train.index[0] != first_valid_idx:
        n_ignored = X_train.index.get_loc(first_valid_idx)
        logger.warning(f"Décalage du début du train à {first_valid_idx} (ignoration de {n_ignored} lignes initiales avec NaN sur les features)")
        X_train = X_train.loc[first_valid_idx:]
        y_train = y_train.loc[first_valid_idx:]
    else:
        logger.info(f"Aucun décalage du début du train nécessaire : toutes les features sont valides dès la première ligne.")
    # 4. Entraînement modèle
    if model_type == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost n'est pas installé")
        model = CatBoostClassifier(loss_function='MultiClass', random_seed=random_state, verbose=0)
        model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), early_stopping_rounds=20)
    elif model_type == "xgboost":
        if xgb is None:
            raise ImportError("xgboost n'est pas installé")
        model = xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Modèle non supporté : {model_type}")
    # 5. Prédiction/évaluation
    y_val_pred = np.array(model.predict(X_val)).ravel()
    y_test_pred = np.array(model.predict(X_test)).ravel()
    y_val_proba = model.predict_proba(X_val)[:,1] if hasattr(model, "predict_proba") else None
    y_test_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    metrics = {
        "val": {
            "accuracy": accuracy_score(y_val, y_val_pred),
            "f1": f1_score(y_val, y_val_pred, average="macro"),
            "roc_auc": roc_auc_score(y_val, y_val_proba) if y_val_proba is not None and len(np.unique(y_val)) == 2 else None,
            "confusion": confusion_matrix(y_val, y_val_pred).tolist()
        },
        "test": {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "f1": f1_score(y_test, y_test_pred, average="macro"),
            "roc_auc": roc_auc_score(y_test, y_test_proba) if y_test_proba is not None and len(np.unique(y_test)) == 2 else None,
            "confusion": confusion_matrix(y_test, y_test_pred).tolist()
        }
    }
    # 6. Reporting
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"y_val": y_val, "y_val_pred": y_val_pred}).to_csv(os.path.join(out_dir, "val_predictions.csv"))
    pd.DataFrame({"y_test": y_test, "y_test_pred": y_test_pred}).to_csv(os.path.join(out_dir, "test_predictions.csv"))
    # Courbe de confusion
    for split, y_true, y_pred in [("val", y_val, y_val_pred), ("test", y_test, y_test_pred)]:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4,4))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"Confusion matrix {split}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"confusion_{split}.png"))
        plt.close()
    # Importance des features (CatBoost/XGBoost)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
        imp_df.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
        plt.figure(figsize=(8,4))
        plt.bar(imp_df["feature"], imp_df["importance"])
        plt.title("Feature importance")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "feature_importance.png"))
        plt.close()
    # Rapport global
    report = {
        "split": os.path.join(out_dir, "split_report.json"),
        "cv_folds": os.path.join(out_dir, "cv_folds.json"),
        "metrics": os.path.join(out_dir, "metrics.json"),
        "val_predictions": os.path.join(out_dir, "val_predictions.csv"),
        "test_predictions": os.path.join(out_dir, "test_predictions.csv"),
        "confusion_val": os.path.join(out_dir, "confusion_val.png"),
        "confusion_test": os.path.join(out_dir, "confusion_test.png"),
        "feature_importance": os.path.join(out_dir, "feature_importance.csv"),
        "feature_importance_png": os.path.join(out_dir, "feature_importance.png")
    }
    with open(os.path.join(out_dir, "global_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Rapport global ML exporté : {os.path.join(out_dir, 'global_report.json')}")
    return report 

def run_tuning_pipeline(
    df: pd.DataFrame,
    label_col: str,
    model_type: str = "catboost",
    param_grid: Optional[Dict[str, Any]] = None,
    method: str = "optuna",
    n_iter: int = 20,
    cv_params: Optional[Dict[str, Any]] = None,
    out_dir: str = "tuning_reports",
    random_state: int = 42,
    cat_features: Optional[list] = None
) -> Dict[str, Any]:
    """
    Orchestration du tuning avancé des hyperparamètres (grid/random/optuna).
    """
    from bitcoin_scalper.core.tuning import tune_model_hyperparams
    report = tune_model_hyperparams(
        df=df,
        label_col=label_col,
        model_type=model_type,
        param_grid=param_grid,
        method=method,
        n_iter=n_iter,
        cv_params=cv_params,
        out_dir=out_dir,
        random_state=random_state,
        cat_features=cat_features
    )
    logger.info(f"Rapport tuning exporté : {report}")
    return report


def run_backtest_pipeline(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "<CLOSE>",
    label_col: Optional[str] = None,
    model: Optional[Any] = None,
    initial_capital: float = 10000.0,
    fee: float = 0.0005,
    slippage: float = 0.0002,
    out_dir: str = "backtest_reports",
    benchmarks: Optional[list] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Orchestration du backtest réaliste (PnL, Sharpe, drawdown, benchmarks).
    """
    from bitcoin_scalper.core.backtesting import Backtester
    backtester = Backtester(
        df=df,
        signal_col=signal_col,
        price_col=price_col,
        label_col=label_col,
        model=model,
        initial_capital=initial_capital,
        fee=fee,
        slippage=slippage,
        out_dir=out_dir,
        benchmarks=benchmarks,
        **kwargs
    )
    out_df, trades, kpis, benchmarks_results = backtester.run()
    out_df.to_csv(os.path.join(out_dir, "backtest_enriched.csv"))
    with open(os.path.join(out_dir, "backtest_kpis.json"), "w") as f:
        json.dump(kpis, f, indent=2)
    logger.info(f"Backtest terminé. KPIs : {kpis}")
    return {"out_df": os.path.join(out_dir, "backtest_enriched.csv"), "kpis": os.path.join(out_dir, "backtest_kpis.json"), "benchmarks": benchmarks_results}


def run_rl_pipeline(
    df: pd.DataFrame,
    window_size: int = 30,
    fee: float = 0.0005,
    spread: float = 0.0002,
    initial_balance: float = 10000.0,
    algo: str = "dqn",
    n_episodes: int = 100,
    out_dir: str = "rl_reports",
    **kwargs
) -> Dict[str, Any]:
    """
    Orchestration d'un pipeline RL (DQN, PPO, etc.) sur l'environnement BitcoinScalperEnv.
    """
    from bitcoin_scalper.core.rl_env import BitcoinScalperEnv
    os.makedirs(out_dir, exist_ok=True)
    arr = df.values.astype(np.float32)
    env = BitcoinScalperEnv(arr, fee=fee, spread=spread, window_size=window_size, initial_balance=initial_balance)
    # Placeholder: à remplacer par un vrai agent RL (DQN, PPO, etc.)
    rewards = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()  # Random policy (à remplacer)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    np.save(os.path.join(out_dir, "rl_rewards.npy"), np.array(rewards))
    logger.info(f"RL pipeline terminé. Moyenne reward : {np.mean(rewards):.2f}")
    return {"rewards": os.path.join(out_dir, "rl_rewards.npy")}


def run_stacking_pipeline(
    df: pd.DataFrame,
    label_col: str,
    base_models: Optional[list] = None,
    meta_model: Optional[Any] = None,
    split_params: Optional[Dict[str, Any]] = None,
    out_dir: str = "stacking_reports",
    random_state: int = 42,
    cat_features: Optional[list] = None
) -> Dict[str, Any]:
    """
    Orchestration d'un pipeline stacking (métamodèle).
    """
    # Placeholder: à implémenter selon les besoins (empilement de modèles)
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Pipeline stacking non encore implémenté (placeholder)")
    return {"status": "not_implemented", "out_dir": out_dir}


def run_hybrid_strategy_pipeline(
    df: pd.DataFrame,
    strategies: Optional[list] = None,
    out_dir: str = "hybrid_reports"
) -> Dict[str, Any]:
    """
    Orchestration d'un pipeline de stratégies hybrides (rule-based + ML).
    """
    from bitcoin_scalper.core.strategies_hybrid import HybridStrategyEngine
    os.makedirs(out_dir, exist_ok=True)
    engine = HybridStrategyEngine(strategies or [])
    engine.fit(df)
    preds = engine.predict(df)
    pd.DataFrame({"hybrid_signal": preds}, index=df.index).to_csv(os.path.join(out_dir, "hybrid_signals.csv"))
    logger.info("Pipeline hybrid terminé.")
    return {"hybrid_signals": os.path.join(out_dir, "hybrid_signals.csv")} 