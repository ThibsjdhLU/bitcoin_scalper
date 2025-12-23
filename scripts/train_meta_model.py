#!/usr/bin/env python3
"""
MetaModel Training Script - FIXED with Cross-Validation + Optuna Optimization
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime
import json

# Add src/ to PYTHONPATH
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit, cross_val_score
import joblib
import optuna

# Import project modules
from bitcoin_scalper.core.data_loading import load_minute_csv
from bitcoin_scalper.core.feature_engineering import FeatureEngineering
from bitcoin_scalper.core.labeling import generate_labels
from bitcoin_scalper.models.meta_model import MetaModel

# Optional CatBoost import
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    print("⚠️  CatBoost not installed. Please install: pip install catboost")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs' / f'train_meta_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MetaModel for Bitcoin trading with Optuna optimization')
    parser.add_argument('--csv', type=str, 
                       default=str(project_root / 'data/raw/BTCUSD_M1_202301010000_202512011647.csv'),
                       help='Path to OHLCV CSV file')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--threshold', type=float, default=0.6, 
                       help='Meta model confidence threshold for signal filtering')
    parser.add_argument('--horizon', type=int, default=15, 
                       help='Prediction horizon for label generation')
    parser.add_argument('--label_k', type=float, default=0.5, 
                       help='Standard deviation multiplier for label thresholds')
    parser.add_argument('--output', type=str,
                       default=str(project_root / 'models/meta_model_production.pkl'),
                       help='Output path for trained model')
    return parser.parse_args()


def load_and_prepare_data(csv_path: str):
    logger.info("=" * 70)
    logger.info("📊 STEP 1: DATA LOADING")
    logger.info("=" * 70)
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = load_minute_csv(csv_path, fill_method='ffill')
    
    # Convert all numeric columns to float32 to reduce RAM usage
    numeric_cols = df.select_dtypes(include=[np.float64, np.int64, np.int32]).columns
    for col in numeric_cols:
        df[col] = df[col].astype(np.float32)
    
    logger.info(f"✅ Loaded {len(df)} rows (converted to float32)")
    return df


def generate_features(df: pd.DataFrame):
    logger.info("\n" + "=" * 70)
    logger.info("🔧 STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 70)
    feature_eng = FeatureEngineering()
    df['log_return_1m'] = np.log(df['<CLOSE>'] / df['<CLOSE>'].shift(1))
    df = feature_eng.add_indicators(df, price_col='<CLOSE>', high_col='<HIGH>', low_col='<LOW>', volume_col='<TICKVOL>', prefix='1min_')
    df = feature_eng.add_features(df, price_col='<CLOSE>', volume_col='<TICKVOL>', prefix='1min_')
    initial_len = len(df)
    df = df.dropna()
    
    # Convert all numeric columns to float32 to reduce RAM usage
    numeric_cols = df.select_dtypes(include=[np.float64, np.int64, np.int32]).columns
    for col in numeric_cols:
        df[col] = df[col].astype(np.float32)
    
    logger.info(f"✅ Generated features (dropped {initial_len - len(df)} NaN rows, converted to float32)")
    return df


def generate_labels_primary(df: pd.DataFrame, horizon: int = 15, k: float = 0.5):
    """Generate GROUND TRUTH labels for Direction."""
    logger.info("\n" + "=" * 70)
    logger.info("🏷️  STEP 3: LABEL GENERATION (PRIMARY)")
    logger.info("=" * 70)
    
    y_direction = generate_labels(
        df, horizon=horizon, k=k, threshold_type='std',
        n_classes=3, neutral_policy='keep'
    )
    # Align
    df = df.loc[y_direction.index]
    y_direction = y_direction.astype(int)
    
    counts = y_direction.value_counts().sort_index()
    logger.info(f"Labels: {counts.to_dict()}")
    return df, y_direction


def generate_meta_labels_cv(X, y_true, primary_params, cv_folds=3):
    """
    GENERATE REALISTIC META-LABELS VIA CROSS-VALIDATION.
    This forces the primary model to make errors, creating a valid target for the Meta-Model.
    """
    logger.info("\n" + "=" * 70)
    logger.info("🧠 STEP 3.5: GENERATING META-LABELS (CV)")
    logger.info("=" * 70)
    logger.info(f"Running {cv_folds}-fold Cross-Validation to generate out-of-sample predictions...")
    
    # Initialize a temporary model for CV
    temp_model = CatBoostClassifier(
        iterations=min(primary_params['iterations'], 100), # Faster for CV
        depth=primary_params['depth'],
        learning_rate=primary_params['learning_rate'],
        loss_function='MultiClass',
        verbose=False,
        random_state=42,
        task_type='CPU',
        allow_writing_files=False
    )
    
    # TimeSeriesSplit to prevent leakage
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Generate predictions on the training set
    # Note: cross_val_predict returns predictions for elements that were in the test set of the CV
    # For TimeSeriesSplit, this means the first chunk of data won't have predictions.
    # We handle this by aligning indices.
    
    try:
        y_pred_cv = cross_val_predict(temp_model, X, y_true, cv=tscv, n_jobs=-1)
    except Exception as e:
        logger.warning(f"CV failed ({e}), falling back to simple split...")
        # Fallback: simple fit on first 50%, predict on rest (simpler leakage avoidance)
        split = int(len(X) * 0.5)
        temp_model.fit(X.iloc[:split], y_true.iloc[:split])
        y_pred_rest = temp_model.predict(X.iloc[split:]).flatten()
        # Pad beginning with 0 (neutral) or ignore
        y_pred_cv = np.concatenate([np.zeros(split), y_pred_rest])

    # Align lengths (Time Series Split might return smaller array if not handled carefully, 
    # but sklearn cross_val_predict usually pads or returns aligned if method is clear.
    # Actually TimeSeriesSplit with cross_val_predict is tricky.
    # Simpler approach for robustness: Use a single hold-out for meta-label generation? 
    # No, let's use the predictions we have.
    
    # Ensure y_pred_cv is same length, if not pad with 0 at start
    if len(y_pred_cv) < len(y_true):
        pad = len(y_true) - len(y_pred_cv)
        y_pred_cv = np.concatenate([np.zeros(pad), y_pred_cv])
        
    # === META LABEL DEFINITION ===
    # 1 = Success (Model traded AND was correct)
    # 0 = Failure (Model traded AND was wrong) OR Model didn't trade (Neutral)
    
    # We only care about filtering ACTIVE trades.
    # If Primary says 0 (Neutral), Meta is irrelevant (we don't trade anyway).
    # If Primary says 1/-1, Meta checks if it matches y_true.
    
    y_meta = pd.Series(0, index=y_true.index)
    
    # Indices where model took a trade
    active_indices = (y_pred_cv != 0)
    
    # Success = Active Trade AND Correct Prediction
    success_mask = active_indices & (y_pred_cv == y_true)
    
    y_meta[success_mask] = 1
    
    # Log stats
    n_trades = active_indices.sum()
    n_success = success_mask.sum()
    n_fail = n_trades - n_success
    
    logger.info(f"CV Predictions Generated:")
    logger.info(f"   Total Signals: {n_trades} (simulated)")
    logger.info(f"   Correct:       {n_success} (Meta-Label=1)")
    logger.info(f"   Wrong:         {n_fail} (Meta-Label=0)")
    
    if n_fail == 0:
        logger.warning("⚠️ WARNING: Primary model is still perfect on CV? This is highly suspicious.")
        # Force some errors for stability if needed, but CV should handle it.
    else:
        logger.info(f"   Meta-Model Data: {n_success} Positives / {n_fail} Negatives")
        
    return y_meta


def optimize_primary_params(X, y):
    """
    Optimize CatBoost hyperparameters for the primary model using Optuna.
    Runs 50 trials to maximize Accuracy.
    
    Args:
        X: Feature matrix (DataFrame or ndarray)
        y: Target labels (Series or ndarray)
    
    Returns:
        dict: Best hyperparameters found
    """
    logger.info("\n" + "=" * 70)
    logger.info("🔍 OPTIMIZING PRIMARY MODEL HYPERPARAMETERS")
    logger.info("=" * 70)
    
    # Convert to float32 if needed
    if isinstance(X, pd.DataFrame):
        X_opt = X.copy()
        numeric_cols = X_opt.select_dtypes(include=[np.float64, np.int64, np.int32]).columns
        for col in numeric_cols:
            X_opt[col] = X_opt[col].astype(np.float32)
    else:
        X_opt = X.astype(np.float32)
    
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'loss_function': 'MultiClass',
            'verbose': False,
            'random_state': 42,
            'task_type': 'CPU',
            'allow_writing_files': False
        }
        
        # Create model
        model = CatBoostClassifier(**params)
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Calculate cross-validated accuracy
        try:
            scores = cross_val_score(model, X_opt, y, cv=tscv, scoring='accuracy', n_jobs=1)
            return scores.mean()
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0
    
    # Create and run study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    logger.info("Starting 50 trials for primary model optimization...")
    study.optimize(objective, n_trials=50, show_progress_bar=False, n_jobs=1)
    
    best_params = study.best_params
    logger.info(f"✅ Best Accuracy: {study.best_value:.4f}")
    logger.info(f"✅ Best Parameters: {best_params}")
    
    return best_params


def optimize_meta_params(X, y):
    """
    Optimize CatBoost hyperparameters for the meta model using Optuna.
    
    Args:
        X: Feature matrix (DataFrame or ndarray)
        y: Target labels (Series or ndarray)
    
    Returns:
        dict: Best hyperparameters found
    """
    logger.info("\n" + "=" * 70)
    logger.info("🔍 OPTIMIZING META MODEL HYPERPARAMETERS")
    logger.info("=" * 70)
    
    # Convert to float32 if needed
    if isinstance(X, pd.DataFrame):
        X_opt = X.copy()
        numeric_cols = X_opt.select_dtypes(include=[np.float64, np.int64, np.int32]).columns
        for col in numeric_cols:
            X_opt[col] = X_opt[col].astype(np.float32)
    else:
        X_opt = X.astype(np.float32)
    
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'loss_function': 'Logloss',
            'verbose': False,
            'random_state': 43,
            'task_type': 'CPU',
            'allow_writing_files': False
        }
        
        # Create model
        model = CatBoostClassifier(**params)
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Calculate cross-validated accuracy
        try:
            scores = cross_val_score(model, X_opt, y, cv=tscv, scoring='accuracy', n_jobs=1)
            return scores.mean()
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0
    
    # Create and run study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=43)
    )
    
    logger.info("Starting optimization for meta model...")
    study.optimize(objective, n_trials=50, show_progress_bar=False, n_jobs=1)
    
    best_params = study.best_params
    logger.info(f"✅ Best Accuracy: {study.best_value:.4f}")
    logger.info(f"✅ Best Parameters: {best_params}")
    
    return best_params


def evaluate_model(meta_model, X_test, y_dir_test):
    """
    Evaluate the meta model on test data using flat numpy arrays.
    
    Args:
        meta_model: Trained MetaModel instance
        X_test: Test features
        y_dir_test: Test direction labels
    """
    logger.info("\n📊 STEP 6: EVALUATION (Test Set)")
    
    # Predict on test
    res = meta_model.predict_meta(X_test, return_all=True)
    
    # Convert ALL to flat numpy arrays to avoid indexing errors
    # flatten() ensures we have 1D arrays (N,)
    raw = np.array(res['raw_signal']).flatten()
    final = np.array(res['final_signal']).flatten()
    y_true = np.array(y_dir_test).flatten()
    
    # Check shapes
    if len(raw) != len(y_true):
        logger.warning(f"Shape mismatch: raw={len(raw)}, true={len(y_true)}. Adjusting...")
        min_len = min(len(raw), len(y_true))
        raw = raw[:min_len]
        final = final[:min_len]
        y_true = y_true[:min_len]

    # Metrics 1: Primary (Raw)
    # Mask where signal is NOT 0 (Neutral)
    mask_raw = (raw != 0) 
    
    if mask_raw.sum() > 0:
        # We only evaluate accuracy on active trades
        raw_acc = accuracy_score(y_true[mask_raw], raw[mask_raw])
    else:
        raw_acc = 0.0
        
    # Metrics 2: Final (Meta Filtered)
    mask_final = (final != 0)
    
    if mask_final.sum() > 0:
        final_acc = accuracy_score(y_true[mask_final], final[mask_final])
        # Filter rate = (Trades Removed / Original Trades)
        filter_rate = (1 - mask_final.sum() / max(mask_raw.sum(), 1)) * 100
    else:
        final_acc = 0.0
        filter_rate = 100.0
        
    logger.info(f"Primary Accuracy (Raw):    {raw_acc:.4f}")
    logger.info(f"Final Accuracy (Filtered): {final_acc:.4f}")
    logger.info(f"Trades Ignored: {filter_rate:.1f}%")
    
    if final_acc > raw_acc:
        logger.info(f"🚀 PERFORMANCE BOOST: +{(final_acc-raw_acc)*100:.2f}%")
    else:
        logger.info(f"⚠️ No boost detected")
    
    return raw_acc, final_acc, filter_rate


def main():
    args = parse_args()
    print("="*70 + "\n🚀 MetaModel Training Pipeline with Optuna Optimization\n" + "="*70)
    
    # 1. Load & Feat
    df = load_and_prepare_data(args.csv)
    df = generate_features(df)
    
    # 2. Labels (Primary Ground Truth)
    df, y_direction = generate_labels_primary(df, args.horizon, args.label_k)
    
    # 3. Split
    logger.info("\n✂️  STEP 4: TRAIN/TEST SPLIT")
    exclude = ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']
    X = df[[c for c in df.columns if c not in exclude]]
    
    # Convert to float32 for all numeric columns
    numeric_cols = X.select_dtypes(include=[np.float64, np.int64, np.int32]).columns
    for col in numeric_cols:
        X[col] = X[col].astype(np.float32)
    
    split_idx = int(len(X) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_dir_train, y_dir_test = y_direction.iloc[:split_idx], y_direction.iloc[split_idx:]
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 4. Optimize Primary Model Parameters
    logger.info("\n" + "=" * 70)
    logger.info("🔧 STEP 4.5: HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 70)
    
    best_primary_params = optimize_primary_params(X_train, y_dir_train)
    
    # Build full primary params dict
    primary_params = {
        'iterations': best_primary_params['iterations'],
        'depth': best_primary_params['depth'],
        'learning_rate': best_primary_params['learning_rate'],
        'l2_leaf_reg': best_primary_params['l2_leaf_reg'],
        'border_count': best_primary_params['border_count']
    }
    
    # 5. Generate Meta-Labels via CV (The Fix)
    # This ensures y_meta_train contains mistakes!
    y_meta_train = generate_meta_labels_cv(X_train, y_dir_train, primary_params)
    
    # 6. Optimize Meta Model Parameters
    # For meta optimization, we need augmented features (original + primary probabilities)
    # We'll train a temporary primary model first
    logger.info("\nTraining temporary primary model for meta optimization...")
    temp_primary = CatBoostClassifier(
        **primary_params,
        loss_function='MultiClass',
        verbose=False,
        random_state=42,
        task_type='CPU',
        allow_writing_files=False
    )
    temp_primary.fit(X_train, y_dir_train)
    
    # Generate augmented features for meta training
    primary_proba_train = temp_primary.predict_proba(X_train)
    
    if isinstance(X_train, pd.DataFrame):
        proba_cols = [f'primary_proba_{i}' for i in range(primary_proba_train.shape[1])]
        proba_df = pd.DataFrame(
            primary_proba_train,
            index=X_train.index,
            columns=proba_cols
        )
        X_meta_train = pd.concat([X_train, proba_df], axis=1)
    else:
        X_meta_train = np.hstack([X_train, primary_proba_train])
    
    # Optimize meta parameters
    best_meta_params = optimize_meta_params(X_meta_train, y_meta_train)
    
    # 7. Train Final MetaModel with optimized parameters
    logger.info("\n🤖 STEP 5: FINAL TRAINING WITH OPTIMIZED PARAMETERS")
    
    primary_model = CatBoostClassifier(
        iterations=best_primary_params['iterations'],
        depth=best_primary_params['depth'],
        learning_rate=best_primary_params['learning_rate'],
        l2_leaf_reg=best_primary_params['l2_leaf_reg'],
        border_count=best_primary_params['border_count'],
        loss_function='MultiClass',
        verbose=False,
        random_state=42,
        task_type='CPU',
        allow_writing_files=False
    )
    
    meta_model_classifier = CatBoostClassifier(
        iterations=best_meta_params['iterations'],
        depth=best_meta_params['depth'],
        learning_rate=best_meta_params['learning_rate'],
        l2_leaf_reg=best_meta_params['l2_leaf_reg'],
        border_count=best_meta_params['border_count'],
        loss_function='Logloss',
        verbose=False,
        random_state=43,
        task_type='CPU',
        allow_writing_files=False
    )
    
    meta_model = MetaModel(primary_model, meta_model_classifier, args.threshold)
    
    # Train the CLASS wrapper
    # Note: We pass y_meta_train as y_success. The MetaModel class will train the secondary model on this.
    meta_model.train(X_train, y_dir_train, y_meta_train)
    
    # 8. Evaluate
    raw_acc, final_acc, filter_rate = evaluate_model(meta_model, X_test, y_dir_test)

    # 9. Save
    save_model(meta_model, args.output, best_primary_params, best_meta_params)


def save_model(model, path, primary_params, meta_params):
    logger.info("\n💾 STEP 7: SAVING")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    joblib.dump(model, p)
    logger.info(f"Saved model to {p}")
    
    # Save hyperparameters as JSON
    params_path = p.parent / f"{p.stem}_params.json"
    params_data = {
        'primary_params': primary_params,
        'meta_params': meta_params,
        'meta_threshold': model.meta_threshold
    }
    with open(params_path, 'w') as f:
        json.dump(params_data, f, indent=2)
    logger.info(f"Saved hyperparameters to {params_path}")

if __name__ == '__main__':
    main()
