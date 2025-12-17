import pandas as pd
import numpy as np
import os
import json
import logging
import joblib
from typing import Dict, Any, Optional
from bitcoin_scalper.core.splitting import temporal_train_val_test_split, generate_time_series_folds
from bitcoin_scalper.core.modeling import ModelTrainer
from bitcoin_scalper.core.backtesting import Backtester
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

logger = logging.getLogger("bitcoin_scalper.ml_orchestrator")
logger.setLevel(logging.INFO)

def run_ml_pipeline(
    df: pd.DataFrame,
    label_col: str,
    model_type: str = "catboost",
    split_params: Optional[Dict[str, Any]] = None,
    cv_params: Optional[Dict[str, Any]] = None,
    out_dir: str = "ml_reports",
    random_state: int = 42,
    cat_features: Optional[list] = None,
    tuning_method: str = "optuna",
    n_trials: int = 20,
    timeout: Optional[int] = 600,
    early_stopping_rounds: int = 20
) -> Dict[str, Any]:
    """
    Orchestration complète du pipeline ML "Master Edition":
    - Strict Feature Selection (Stationarity Only)
    - Expert Training via ModelTrainer (RobustScaler + Optuna)
    - Double Artifact Saving (Archive + Production Pointers)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Create Production Models Directory
    PROD_DIR = "models"
    os.makedirs(PROD_DIR, exist_ok=True)

    # 1. Strict Feature Selection (Golden Rule + Kill List)
    features = [c for c in df.columns if c != label_col]

    # KILL LIST (Case insensitive)
    KILL_LIST = ['open', 'high', 'low', 'close', 'volume', 'tick_volume', 'vwap', 'sma', 'ema', 'wma']

    # ALLOWED TRANSFORMS (Prefixes/Suffixes)
    ALLOWED_TRANSFORMS = ['ret_', 'log_', 'dist_', 'ratio_', 'osc_', '_rsi', '_adx', '_cci', '_mfi', '_roc', 'diff_']

    def is_safe_feature(col_name):
        col_lower = col_name.lower()

        # 1. Kill List Check
        for forbidden in KILL_LIST:
            # Check if forbidden word is present WITHOUT a transforming prefix/context
            # e.g. "close" is bad, "dist_close" is good.
            # But "sma_200" is bad. "dist_sma_200" is good.
            # Strategy: If the column name CONTAINS a forbidden word, it is SUSPECT.
            if forbidden in col_lower:
                # It is only allowed if it passes the Golden Rule
                pass

            # Additional explicit check for raw columns like "<CLOSE>", "1min_<CLOSE>"
            if col_lower == forbidden or f"<{forbidden}>" in col_lower or f"_{forbidden}" in col_lower or f"{forbidden}_" in col_lower:
                 # It's likely raw unless transformed.
                 pass

        # 2. Golden Rule: Must have a transform prefix/suffix
        is_transformed = any(t in col_lower for t in ALLOWED_TRANSFORMS)

        # 3. Explicit check: If it's a raw price/volume, kill it regardless of transform check (sanity check)
        # e.g. "log_close" is technically transformed but maybe we want "log_return".
        # But "log_close" is not stationary if price trends. "log_return" is diff.
        # "dist_sma" is stationary.

        # Strict Kill: If it matches exactly forbidden patterns without "dist", "ratio", "ret", "diff"
        # Using the blacklist provided in instructions.
        for ban in KILL_LIST:
            if ban in col_lower:
                # Exception: it is allowed ONLY if it contains specific stationarity keywords
                if any(x in col_lower for x in ['dist_', 'ratio_', 'ret_', 'diff_', 'osc_', 'rsi', 'adx']):
                    continue
                return False # Banned

        # If it survived the kill list, is it explicitly allowed or generally safe?
        # The prompt says: "Si une colonne ne contient pas un préfixe/suffixe... elle est suspecte et doit être dropée."
        if not is_transformed:
             # Last chance: maybe it is a known indicator like "rsi", "atr" (often raw but bounded)
             if any(x in col_lower for x in ['rsi', 'adx', 'cci', 'mfi', 'atr']): # ATR is not stationary usually (price dep), but often used.
                 # Wait, ATR is price dependent. ATR% or ATR/Close is stationary.
                 # Let's trust the "Golden Rule" strictly as requested.
                 return False

        return True

    final_features = [f for f in features if is_safe_feature(f)]
    dropped_features = list(set(features) - set(final_features))

    logger.info(f"🚫 Dropped {len(dropped_features)} non-stationary/raw features.")
    if dropped_features:
        logger.info(f"Examples dropped: {dropped_features[:10]}...")
    logger.info(f"✅ Training with {len(final_features)} stationary features.")

    # Persist the FINAL feature list (Double Save)
    # 1. Archive
    joblib.dump(final_features, os.path.join(out_dir, "features_list.pkl"))
    # 2. Production
    joblib.dump(final_features, os.path.join(PROD_DIR, "latest_features_list.pkl"))
    logger.info(f"💾 Feature list saved to {os.path.join(PROD_DIR, 'latest_features_list.pkl')}")

    # 2. Split Data
    split_params = split_params or {"train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15, "horizon": 0}
    train, val, test = temporal_train_val_test_split(df, **split_params, report_path=os.path.join(out_dir, "split_report.json"))

    X_train, y_train = train[final_features], train[label_col]
    X_val, y_val = val[final_features], val[label_col]
    X_test, y_test = test[final_features], test[label_col]

    # Clean numeric features
    numeric_features = [f for f in final_features if not (cat_features and f in cat_features)]
    for split_name, X in zip(['train', 'val', 'test'], [X_train, X_val, X_test]):
        if numeric_features:
            X_converted = X.copy()
            X_converted[numeric_features] = X_converted[numeric_features].apply(pd.to_numeric, errors='coerce')
            if split_name == 'train': X_train = X_converted
            elif split_name == 'val': X_val = X_converted
            else: X_test = X_converted

    # Remove NaNs in Train
    if X_train.isna().any().any():
         logger.warning("NaNs found in X_train after numeric conversion. Filling with 0 (safe for stationary features like returns).")
         X_train = X_train.fillna(0)
         # Val/Test handled by scaler usually, but good to be clean
         X_val = X_val.fillna(0)
         X_test = X_test.fillna(0)

    # ✅ PHASE 3: Gestion du Déséquilibre (SMOTE)
    # Si le marché a 90% de "Hold" et 10% de "Trade", SMOTE est activé pour forcer le modèle à apprendre les signaux rares
    class_distribution = y_train.value_counts(normalize=True)
    logger.info(f"📊 Class distribution before SMOTE: {class_distribution.to_dict()}")
    
    # Check for severe imbalance (any class < 15% and not all classes are very balanced)
    min_class_ratio = class_distribution.min()
    max_class_ratio = class_distribution.max()
    imbalance_ratio = max_class_ratio / min_class_ratio if min_class_ratio > 0 else float('inf')
    
    if imbalance_ratio > 3.0:  # Severe imbalance threshold
        logger.warning(f"⚠️ Severe class imbalance detected (ratio: {imbalance_ratio:.2f}). Applying SMOTE...")
        try:
            from bitcoin_scalper.core.balancing import balance_with_smote
            smote_result = balance_with_smote(X_train, y_train, random_state=random_state)
            if smote_result is not None:
                X_train, y_train = smote_result
                # Convert back to DataFrame/Series with proper indexing
                if not isinstance(X_train, pd.DataFrame):
                    X_train = pd.DataFrame(X_train, columns=final_features)
                if not isinstance(y_train, pd.Series):
                    y_train = pd.Series(y_train, name=label_col)
                logger.info(f"✅ SMOTE applied: New distribution: {y_train.value_counts(normalize=True).to_dict()}")
            else:
                logger.warning("⚠️ SMOTE not available (imblearn not installed). Proceeding without balancing.")
        except Exception as e:
            logger.warning(f"⚠️ SMOTE failed: {e}. Proceeding without balancing.")
    else:
        logger.info(f"✅ Class balance acceptable (ratio: {imbalance_ratio:.2f}). No SMOTE needed.")

    # 3. Train with ModelTrainer (Pipeline + Optuna)
    trainer = ModelTrainer(algo=model_type, random_state=random_state, use_scaler=True)
    pipeline = trainer.fit(
        X_train, y_train,
        X_val, y_val,
        tuning_method=tuning_method,
        n_trials=n_trials,
        timeout=timeout,
        early_stopping_rounds=early_stopping_rounds,
        cat_features=cat_features
    )

    # 4. Save Model (Double Save)
    # 1. Archive
    joblib.dump(pipeline, os.path.join(out_dir, "model_pipeline.pkl"))
    # 2. Production
    joblib.dump(pipeline, os.path.join(PROD_DIR, "latest_model.pkl"))
    logger.info(f"💾 Model Pipeline saved to {os.path.join(PROD_DIR, 'latest_model.pkl')}")
    
    # ✅ PHASE 5: Save training reference data for Drift Monitor
    # Save a sample of training data for KS-Test comparison in production
    train_reference = train[final_features].sample(n=min(1000, len(train)), random_state=random_state)
    joblib.dump(train_reference, os.path.join(PROD_DIR, "train_reference.pkl"))
    logger.info(f"💾 Training reference saved for drift monitoring ({len(train_reference)} samples)")

    # 5. Evaluation
    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)

    # Proba
    y_val_proba = None
    y_test_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_val_proba = pipeline.predict_proba(X_val)
            y_test_proba = pipeline.predict_proba(X_test)
            if y_val_proba.shape[1] == 2:
                y_val_proba = y_val_proba[:, 1]
                y_test_proba = y_test_proba[:, 1]
        except Exception: pass

    metrics = {
        "val": {
            "accuracy": accuracy_score(y_val, y_val_pred),
            "f1": f1_score(y_val, y_val_pred, average="macro"),
            "confusion": confusion_matrix(y_val, y_val_pred).tolist()
        },
        "test": {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "f1": f1_score(y_test, y_test_pred, average="macro"),
            "confusion": confusion_matrix(y_test, y_test_pred).tolist()
        }
    }

    # Financial Verification (Test Set)
    price_col = next((c for c in df.columns if 'CLOSE' in c.upper() and ('1MIN' in c.upper() or c == '<CLOSE>')), None)
    if price_col:
        test_bt_df = test.copy() # Contains all original columns including price
        test_bt_df['signal'] = y_test_pred
        try:
            backtester = Backtester(
                df=test_bt_df,
                signal_col='signal',
                price_col=price_col,
                out_dir=os.path.join(out_dir, "test_backtest")
            )
            _, _, kpis, _ = backtester.run()
            metrics["test"]["financial"] = kpis

            p_color = "\033[92m" if kpis['final_return'] > 0 else "\033[91m"
            logger.info("="*60)
            logger.info("💰 FINANCIAL VERIFICATION REPORT (TEST SET) 💰")
            logger.info(f"Final Capital:   ${kpis['final_capital']:.2f} ({p_color}{kpis['final_return']*100:+.2f}%\033[0m)")
            logger.info("="*60)
        except Exception as e:
            logger.error(f"Financial verification failed: {e}")

    # Export Report
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    report = {
        "metrics": os.path.join(out_dir, "metrics.json"),
        "model_object": pipeline # Return the pipeline object
    }
    return report

def run_tuning_pipeline(*args, **kwargs):
    from bitcoin_scalper.core.tuning import tune_model_hyperparams
    return tune_model_hyperparams(*args, **kwargs)

def run_backtest_pipeline(*args, **kwargs):
    from bitcoin_scalper.core.backtesting import Backtester
    backtester = Backtester(*args, **kwargs)
    out_df, _, kpis, benchmarks = backtester.run()
    return {"kpis": kpis}

def run_rl_pipeline(*args, **kwargs):
    return {"status": "rl_placeholder"}

def run_stacking_pipeline(*args, **kwargs):
    return {"status": "stacking_placeholder"}

def run_hybrid_strategy_pipeline(*args, **kwargs):
    return {"status": "hybrid_placeholder"}
