#!/usr/bin/env python3
"""
MetaModel Training Script - FIXED with Cross-Validation
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
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit
import joblib

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
    parser = argparse.ArgumentParser(description='Train MetaModel for Bitcoin trading')
    parser.add_argument('--csv', type=str, 
                       default=str(project_root / 'data/raw/BTCUSD_M1_202301010000_202512011647.csv'),
                       help='Path to OHLCV CSV file')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--primary_iterations', type=int, default=200)
    parser.add_argument('--primary_depth', type=int, default=6)
    parser.add_argument('--primary_lr', type=float, default=0.05)
    parser.add_argument('--meta_iterations', type=int, default=100)
    parser.add_argument('--meta_depth', type=int, default=4)
    parser.add_argument('--meta_lr', type=float, default=0.05)
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--label_k', type=float, default=0.5)
    parser.add_argument('--output', type=str,
                       default=str(project_root / 'models/meta_model_production.pkl'))
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def load_and_prepare_data(csv_path: str):
    logger.info("=" * 70)
    logger.info("📊 STEP 1: DATA LOADING")
    logger.info("=" * 70)
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = load_minute_csv(csv_path, fill_method='ffill')
    logger.info(f"✅ Loaded {len(df)} rows")
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
    logger.info(f"✅ Generated features (dropped {initial_len - len(df)} NaN rows)")
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


def main():
    args = parse_args()
    print("="*70 + "\n🚀 MetaModel Training Pipeline (Fixed)\n" + "="*70)
    
    # 1. Load & Feat
    df = load_and_prepare_data(args.csv)
    df = generate_features(df)
    
    # 2. Labels (Primary Ground Truth)
    df, y_direction = generate_labels_primary(df, args.horizon, args.label_k)
    
    # 3. Split
    logger.info("\n✂️  STEP 4: TRAIN/TEST SPLIT")
    exclude = ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']
    X = df[[c for c in df.columns if c not in exclude]]
    
    split_idx = int(len(X) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_dir_train, y_dir_test = y_direction.iloc[:split_idx], y_direction.iloc[split_idx:]
    
    # 4. Generate Meta-Labels via CV (The Fix)
    # This ensures y_meta_train contains mistakes!
    primary_params = {
        'iterations': args.primary_iterations, 'depth': args.primary_depth, 'learning_rate': args.primary_lr
    }
    y_meta_train = generate_meta_labels_cv(X_train, y_dir_train, primary_params)
    
    # For Test set, we don't need to generate meta-labels for training, 
    # but we need ground truth to evaluate.
    # Ground truth for test = (Perfect Prediction == True Prediction)
    # BUT we can't know Perfect Prediction without the model. 
    # We will let the evaluate function handle this naturally.
    # We just create a dummy y_meta_test for the API consistency or ignore it.
    y_meta_test = pd.Series(0, index=y_dir_test.index) # Dummy, unused for eval logic
    
    # 5. Train Final MetaModel
    logger.info("\n🤖 STEP 5: FINAL TRAINING")
    
    primary_model = CatBoostClassifier(
        **primary_params, loss_function='MultiClass', verbose=False, random_state=42, allow_writing_files=False
    )
    
    meta_model_classifier = CatBoostClassifier(
        iterations=args.meta_iterations, depth=args.meta_depth, learning_rate=args.meta_lr,
        loss_function='Logloss', verbose=False, random_state=43, allow_writing_files=False
    )
    
    meta_model = MetaModel(primary_model, meta_model_classifier, args.threshold)
    
    # Train the CLASS wrapper
    # Note: We pass y_meta_train as y_success. The MetaModel class will train the secondary model on this.
    meta_model.train(X_train, y_dir_train, y_meta_train)
    
    # 6. Evaluate
    logger.info("\n📊 STEP 6: EVALUATION (Test Set)")
    
    # Predict on test
    # predict_meta returns numpy arrays or simple types, not Series
    res = meta_model.predict_meta(X_test, return_all=True)
    raw = res['raw_signal']
    final = res['final_signal']
    conf = res['meta_conf']
    
    # Convert y_dir_test to numpy for safe comparison
    y_test_np = y_dir_test.values
    
    # Metrics calculation
    # 1. Primary Model Performance (Raw)
    active_mask_raw = (raw != 0)
    if active_mask_raw.sum() > 0:
        raw_acc = accuracy_score(y_test_np[active_mask_raw], raw[active_mask_raw])
    else:
        raw_acc = 0.0
        
    # 2. Meta Model Performance (Filtered)
    active_mask_final = (final != 0)
    if active_mask_final.sum() > 0:
        final_acc = accuracy_score(y_test_np[active_mask_final], final[active_mask_final])
        filter_rate = (1 - active_mask_final.sum() / active_mask_raw.sum()) * 100
    else:
        final_acc = 0.0
        filter_rate = 100.0
        
    logger.info(f"Primary Accuracy (Raw):    {raw_acc:.4f}")
    logger.info(f"Final Accuracy (Filtered): {final_acc:.4f}")
    logger.info(f"Trades Ignored: {filter_rate:.1f}%")
    
    if final_acc > raw_acc:
        logger.info(f"🚀 PERFORMANCE BOOST: +{(final_acc-raw_acc)*100:.2f}%")
    else:
        logger.info(f"⚠️ No boost detected (Threshold {args.threshold} might be too high/low)")

    # 7. Save
    save_model(meta_model, args.output)

def save_model(model, path):
    logger.info("\n💾 STEP 7: SAVING")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    logger.info(f"Saved to {p}")

if __name__ == '__main__':
    main()
