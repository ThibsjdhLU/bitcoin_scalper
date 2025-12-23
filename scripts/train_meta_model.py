#!/usr/bin/env python3
"""
MetaModel Training Script - Complete Meta-Labeling Pipeline

This script trains a production-ready MetaModel with two-stage architecture:
- Stage 1: Primary model predicts direction (Buy/Sell/Neutral)
- Stage 2: Meta model predicts success probability (Take/Pass)

The meta model learns to filter false positives from the primary model,
significantly improving Sharpe ratio and trading precision.

Usage:
    python scripts/train_meta_model.py
    python scripts/train_meta_model.py --csv data/raw/custom_data.csv
    python scripts/train_meta_model.py --threshold 0.7
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
from sklearn.metrics import accuracy_score, classification_report
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MetaModel for Bitcoin trading')
    
    # Data arguments
    parser.add_argument('--csv', type=str, 
                       default=str(project_root / 'data/raw/BTCUSD_M1_202301010000_202512011647.csv'),
                       help='Path to OHLCV CSV file')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    
    # Model hyperparameters
    parser.add_argument('--primary_iterations', type=int, default=200,
                       help='Primary model iterations (default: 200)')
    parser.add_argument('--primary_depth', type=int, default=6,
                       help='Primary model depth (default: 6)')
    parser.add_argument('--primary_lr', type=float, default=0.05,
                       help='Primary model learning rate (default: 0.05)')
    
    parser.add_argument('--meta_iterations', type=int, default=100,
                       help='Meta model iterations (default: 100)')
    parser.add_argument('--meta_depth', type=int, default=4,
                       help='Meta model depth (default: 4)')
    parser.add_argument('--meta_lr', type=float, default=0.05,
                       help='Meta model learning rate (default: 0.05)')
    
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Meta confidence threshold (default: 0.6)')
    
    # Labeling arguments
    parser.add_argument('--horizon', type=int, default=15,
                       help='Prediction horizon in minutes (default: 15)')
    parser.add_argument('--label_k', type=float, default=0.5,
                       help='Label threshold multiplier (default: 0.5)')
    
    # Output arguments
    parser.add_argument('--output', type=str,
                       default=str(project_root / 'models/meta_model_production.pkl'),
                       help='Output path for trained model')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def load_and_prepare_data(csv_path: str, verbose: bool = False):
    """
    Load OHLCV data and prepare features.
    
    Returns:
        df: DataFrame with features
    """
    logger.info("=" * 70)
    logger.info("📊 STEP 1: DATA LOADING")
    logger.info("=" * 70)
    
    logger.info(f"Loading data from: {csv_path}")
    
    # Check if file exists
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load data
    df = load_minute_csv(
        csv_path,
        fill_method='ffill'  # Fill missing values
    )
    
    logger.info(f"✅ Loaded {len(df)} rows")
    logger.info(f"   Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"   Columns: {list(df.columns)}")
    
    return df


def generate_features(df: pd.DataFrame, verbose: bool = False):
    """
    Generate technical indicators and features.
    
    Returns:
        df: DataFrame with added features
    """
    logger.info("\n" + "=" * 70)
    logger.info("🔧 STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 70)
    
    feature_eng = FeatureEngineering()
    
    # Add log returns
    logger.info("Computing log returns...")
    df['log_return_1m'] = np.log(df['<CLOSE>'] / df['<CLOSE>'].shift(1))
    
    # Add technical indicators
    logger.info("Computing technical indicators...")
    df = feature_eng.add_indicators(
        df,
        price_col='<CLOSE>',
        high_col='<HIGH>',
        low_col='<LOW>',
        volume_col='<TICKVOL>',
        prefix='1min_'
    )
    
    # Add derived features
    logger.info("Computing derived features...")
    df = feature_eng.add_features(
        df,
        price_col='<CLOSE>',
        volume_col='<TICKVOL>',
        prefix='1min_'
    )
    
    # Drop NaN values introduced by indicators
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    
    logger.info(f"✅ Generated features for {len(df)} rows (dropped {dropped} NaN rows)")
    logger.info(f"   Total features: {len(df.columns)}")
    
    return df


def generate_labels_for_meta(df: pd.DataFrame, horizon: int = 15, k: float = 0.5, verbose: bool = False):
    """
    Generate labels for meta-labeling.
    
    Returns:
        df: DataFrame with labels
        y_direction: Direction labels (-1, 0, 1)
        y_success: Success labels (0, 1)
    """
    logger.info("\n" + "=" * 70)
    logger.info("🏷️  STEP 3: LABEL GENERATION")
    logger.info("=" * 70)
    
    # Generate direction labels (primary labels)
    logger.info(f"Generating direction labels (horizon={horizon}, k={k})...")
    y_direction = generate_labels(
        df,
        horizon=horizon,
        k=k,
        threshold_type='std',
        n_classes=3,
        neutral_policy='keep'
    )
    
    # Align labels with dataframe
    df = df.loc[y_direction.index]
    y_direction = y_direction.astype(int)
    
    logger.info(f"Direction label distribution:")
    for label, count in y_direction.value_counts().sort_index().items():
        pct = count / len(y_direction) * 100
        label_name = {-1: 'SELL', 0: 'NEUTRAL', 1: 'BUY'}[label]
        logger.info(f"   {label_name:8s} ({label:2d}): {count:6d} ({pct:5.1f}%)")
    
    # Generate success labels (meta labels)
    # Success = trade was profitable
    logger.info(f"\nGenerating success labels...")
    
    # Calculate future returns for non-neutral signals
    future_close = df['<CLOSE>'].shift(-horizon)
    future_return = (future_close - df['<CLOSE>']) / df['<CLOSE>']
    
    # Success criteria:
    # - For BUY (1): success if future_return > 0
    # - For SELL (-1): success if future_return < 0
    # - For NEUTRAL (0): always success (no trade taken)
    y_success = pd.Series(0, index=df.index, dtype=int)
    
    # Buy signals
    buy_mask = (y_direction == 1)
    y_success[buy_mask & (future_return > 0)] = 1
    
    # Sell signals
    sell_mask = (y_direction == -1)
    y_success[sell_mask & (future_return < 0)] = 1
    
    # Neutral signals (always "success" since no trade)
    neutral_mask = (y_direction == 0)
    y_success[neutral_mask] = 1
    
    # Drop rows with NaN in success labels
    valid_mask = ~future_return.isna()
    df = df[valid_mask]
    y_direction = y_direction[valid_mask]
    y_success = y_success[valid_mask]
    
    logger.info(f"Success label distribution:")
    for label, count in y_success.value_counts().sort_index().items():
        pct = count / len(y_success) * 100
        label_name = {0: 'FAILED', 1: 'SUCCESS'}[label]
        logger.info(f"   {label_name:8s} ({label}): {count:6d} ({pct:5.1f}%)")
    
    # Show success rate by signal type
    logger.info(f"\nSuccess rate by signal type:")
    for signal in [-1, 0, 1]:
        mask = (y_direction == signal)
        if mask.sum() > 0:
            success_rate = y_success[mask].mean() * 100
            signal_name = {-1: 'SELL', 0: 'NEUTRAL', 1: 'BUY'}[signal]
            logger.info(f"   {signal_name:8s}: {success_rate:5.1f}%")
    
    logger.info(f"\n✅ Generated {len(y_direction)} labeled samples")
    
    return df, y_direction, y_success


def prepare_train_test_split(df: pd.DataFrame, y_direction: pd.Series, y_success: pd.Series, 
                             test_size: float = 0.2, verbose: bool = False):
    """
    Split data into train/test sets respecting temporal order.
    
    Returns:
        X_train, X_test, y_dir_train, y_dir_test, y_succ_train, y_succ_test
    """
    logger.info("\n" + "=" * 70)
    logger.info("✂️  STEP 4: TRAIN/TEST SPLIT")
    logger.info("=" * 70)
    
    # Select feature columns (exclude price columns and labels)
    exclude_cols = ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    
    # Temporal split (respect chronological order)
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    
    y_dir_train = y_direction.iloc[:split_idx]
    y_dir_test = y_direction.iloc[split_idx:]
    
    y_succ_train = y_success.iloc[:split_idx]
    y_succ_test = y_success.iloc[split_idx:]
    
    logger.info(f"Train set: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    logger.info(f"   Date range: {X_train.index.min()} to {X_train.index.max()}")
    logger.info(f"Test set:  {len(X_test)} samples ({test_size*100:.0f}%)")
    logger.info(f"   Date range: {X_test.index.min()} to {X_test.index.max()}")
    logger.info(f"Features: {len(feature_cols)}")
    
    return X_train, X_test, y_dir_train, y_dir_test, y_succ_train, y_succ_test


def train_metamodel(X_train, X_test, y_dir_train, y_dir_test, y_succ_train, y_succ_test,
                   primary_params, meta_params, meta_threshold, verbose=False):
    """
    Train the MetaModel with two-stage architecture.
    
    Returns:
        meta_model: Trained MetaModel instance
    """
    logger.info("\n" + "=" * 70)
    logger.info("🤖 STEP 5: MODEL TRAINING")
    logger.info("=" * 70)
    
    # Create primary model
    logger.info("Creating Primary Model (Direction Prediction)...")
    logger.info(f"   Iterations: {primary_params['iterations']}")
    logger.info(f"   Depth: {primary_params['depth']}")
    logger.info(f"   Learning rate: {primary_params['learning_rate']}")
    
    primary_model = CatBoostClassifier(
        iterations=primary_params['iterations'],
        depth=primary_params['depth'],
        learning_rate=primary_params['learning_rate'],
        loss_function='MultiClass',
        verbose=False,
        random_state=42,
        task_type='CPU',
        early_stopping_rounds=20
    )
    
    # Create meta model
    logger.info("\nCreating Meta Model (Success Prediction)...")
    logger.info(f"   Iterations: {meta_params['iterations']}")
    logger.info(f"   Depth: {meta_params['depth']}")
    logger.info(f"   Learning rate: {meta_params['learning_rate']}")
    
    meta_model_classifier = CatBoostClassifier(
        iterations=meta_params['iterations'],
        depth=meta_params['depth'],
        learning_rate=meta_params['learning_rate'],
        loss_function='Logloss',
        verbose=False,
        random_state=43,
        task_type='CPU',
        early_stopping_rounds=20
    )
    
    # Create MetaModel
    logger.info(f"\nCreating MetaModel (threshold={meta_threshold:.2f})...")
    meta_model = MetaModel(
        primary_model=primary_model,
        meta_model=meta_model_classifier,
        meta_threshold=meta_threshold
    )
    
    # Train
    logger.info("\n🚀 Starting training...")
    logger.info("   Stage 1: Primary model (Direction)")
    logger.info("   Stage 2: Meta model (Success)")
    
    meta_model.train(
        X=X_train,
        y_direction=y_dir_train,
        y_success=y_succ_train,
        eval_set=(X_test, y_dir_test, y_succ_test)
    )
    
    logger.info("✅ Training completed successfully!")
    
    return meta_model


def evaluate_model(meta_model, X_train, X_test, y_dir_train, y_dir_test, 
                  y_succ_train, y_succ_test, verbose=False):
    """
    Evaluate the trained MetaModel and generate comprehensive report.
    """
    logger.info("\n" + "=" * 70)
    logger.info("📊 STEP 6: MODEL EVALUATION")
    logger.info("=" * 70)
    
    # Predictions on test set
    logger.info("\nGenerating predictions on test set...")
    result = meta_model.predict_meta(X_test, return_all=True)
    
    final_signals = result['final_signal']
    meta_conf = result['meta_conf']
    raw_signals = result['raw_signal']
    
    # === Primary Model Performance ===
    logger.info("\n" + "-" * 70)
    logger.info("PRIMARY MODEL PERFORMANCE (Direction Prediction)")
    logger.info("-" * 70)
    
    primary_acc = accuracy_score(y_dir_test, raw_signals)
    logger.info(f"Accuracy: {primary_acc:.4f} ({primary_acc*100:.2f}%)")
    
    # Classification report for primary model
    logger.info("\nClassification Report (Primary):")
    print(classification_report(y_dir_test, raw_signals, 
                                target_names=['SELL', 'NEUTRAL', 'BUY'],
                                digits=4))
    
    # === Meta Model Filtering ===
    logger.info("\n" + "-" * 70)
    logger.info("META MODEL FILTERING")
    logger.info("-" * 70)
    
    n_raw_buy = (raw_signals == 1).sum()
    n_raw_sell = (raw_signals == -1).sum()
    n_raw_neutral = (raw_signals == 0).sum()
    n_raw_trades = n_raw_buy + n_raw_sell
    
    n_final_buy = (final_signals == 1).sum()
    n_final_sell = (final_signals == -1).sum()
    n_final_neutral = (final_signals == 0).sum()
    n_final_trades = n_final_buy + n_final_sell
    
    n_filtered = n_raw_trades - n_final_trades
    filter_rate = (n_filtered / max(n_raw_trades, 1)) * 100
    
    logger.info(f"\nSignal Statistics:")
    logger.info(f"   Raw signals (Primary):")
    logger.info(f"      BUY:     {n_raw_buy:5d} ({n_raw_buy/len(X_test)*100:5.1f}%)")
    logger.info(f"      SELL:    {n_raw_sell:5d} ({n_raw_sell/len(X_test)*100:5.1f}%)")
    logger.info(f"      NEUTRAL: {n_raw_neutral:5d} ({n_raw_neutral/len(X_test)*100:5.1f}%)")
    logger.info(f"      Total trades: {n_raw_trades}")
    
    logger.info(f"\n   Final signals (After Meta Filter):")
    logger.info(f"      BUY:     {n_final_buy:5d} ({n_final_buy/len(X_test)*100:5.1f}%)")
    logger.info(f"      SELL:    {n_final_sell:5d} ({n_final_sell/len(X_test)*100:5.1f}%)")
    logger.info(f"      NEUTRAL: {n_final_neutral:5d} ({n_final_neutral/len(X_test)*100:5.1f}%)")
    logger.info(f"      Total trades: {n_final_trades}")
    
    logger.info(f"\n   📉 Filtered: {n_filtered} trades ({filter_rate:.1f}%)")
    
    # === Meta Model Performance ===
    logger.info("\n" + "-" * 70)
    logger.info("META MODEL PERFORMANCE (On Filtered Trades)")
    logger.info("-" * 70)
    
    # Calculate accuracy on trades only (non-neutral)
    trades_mask = (raw_signals != 0)
    if trades_mask.sum() > 0:
        # For primary model on trades
        primary_trades_acc = accuracy_score(
            y_dir_test[trades_mask],
            raw_signals[trades_mask]
        )
        logger.info(f"Primary accuracy on trades: {primary_trades_acc:.4f} ({primary_trades_acc*100:.2f}%)")
    
    # For filtered trades
    filtered_mask = (final_signals != 0)
    if filtered_mask.sum() > 0:
        filtered_trades_acc = accuracy_score(
            y_dir_test[filtered_mask],
            final_signals[filtered_mask]
        )
        logger.info(f"Filtered accuracy on trades: {filtered_trades_acc:.4f} ({filtered_trades_acc*100:.2f}%)")
        
        improvement = (filtered_trades_acc - primary_trades_acc) * 100
        logger.info(f"🎯 Improvement: {improvement:+.2f} percentage points")
    
    # Meta confidence statistics
    logger.info(f"\n   Meta Confidence Statistics:")
    logger.info(f"      Mean: {meta_conf.mean():.4f}")
    logger.info(f"      Std:  {meta_conf.std():.4f}")
    logger.info(f"      Min:  {meta_conf.min():.4f}")
    logger.info(f"      Max:  {meta_conf.max():.4f}")
    logger.info(f"      High confidence (>0.7): {(meta_conf > 0.7).sum()} ({(meta_conf > 0.7).sum()/len(meta_conf)*100:.1f}%)")
    
    # === Success Rate Analysis ===
    logger.info("\n" + "-" * 70)
    logger.info("SUCCESS RATE ANALYSIS")
    logger.info("-" * 70)
    
    # Calculate actual success for raw signals
    raw_success_rate = {}
    for signal in [-1, 0, 1]:
        mask = (raw_signals == signal)
        if mask.sum() > 0:
            success = y_succ_test[mask].mean()
            raw_success_rate[signal] = success
    
    # Calculate actual success for filtered signals
    filtered_success_rate = {}
    for signal in [-1, 0, 1]:
        mask = (final_signals == signal)
        if mask.sum() > 0:
            success = y_succ_test[mask].mean()
            filtered_success_rate[signal] = success
    
    logger.info(f"\nSuccess Rate by Signal (Raw vs Filtered):")
    for signal in [-1, 1]:  # Only trades
        signal_name = {-1: 'SELL', 1: 'BUY'}[signal]
        raw_sr = raw_success_rate.get(signal, 0) * 100
        filt_sr = filtered_success_rate.get(signal, 0) * 100
        improvement = filt_sr - raw_sr
        logger.info(f"   {signal_name}:")
        logger.info(f"      Raw:      {raw_sr:5.1f}%")
        logger.info(f"      Filtered: {filt_sr:5.1f}% ({improvement:+.1f}pp)")
    
    # Overall success rate on trades
    raw_trades_mask = (raw_signals != 0)
    filtered_trades_mask = (final_signals != 0)
    
    if raw_trades_mask.sum() > 0:
        raw_overall_success = y_succ_test[raw_trades_mask].mean() * 100
        logger.info(f"\n   Overall success rate (Raw trades):      {raw_overall_success:5.1f}%")
    
    if filtered_trades_mask.sum() > 0:
        filtered_overall_success = y_succ_test[filtered_trades_mask].mean() * 100
        logger.info(f"   Overall success rate (Filtered trades): {filtered_overall_success:5.1f}%")
        if raw_trades_mask.sum() > 0:
            improvement = filtered_overall_success - raw_overall_success
            logger.info(f"   🎯 Improvement: {improvement:+.1f} percentage points")
    
    return {
        'primary_accuracy': primary_acc,
        'n_filtered': n_filtered,
        'filter_rate': filter_rate,
        'final_accuracy': filtered_trades_acc if filtered_mask.sum() > 0 else 0
    }


def save_model(meta_model, output_path: str, verbose=False):
    """
    Save the trained MetaModel to disk.
    """
    logger.info("\n" + "=" * 70)
    logger.info("💾 STEP 7: MODEL SAVING")
    logger.info("=" * 70)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving MetaModel to: {output_path}")
    
    # Save complete MetaModel object
    joblib.dump(meta_model, output_path)
    
    logger.info(f"✅ Model saved successfully!")
    logger.info(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Save metadata
    metadata_path = output_path.with_suffix('.json')
    metadata = {
        'model_type': 'MetaModel',
        'meta_threshold': meta_model.meta_threshold,
        'n_features': meta_model.n_features,
        'feature_names': meta_model.feature_names if hasattr(meta_model, 'feature_names') else None,
        'training_date': datetime.now().isoformat(),
        'primary_model': 'CatBoostClassifier',
        'meta_model': 'CatBoostClassifier'
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"   Metadata saved to: {metadata_path}")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("=" * 70)
    print("🚀 MetaModel Training Pipeline")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Step 1: Load data
        df = load_and_prepare_data(args.csv, args.verbose)
        
        # Step 2: Generate features
        df = generate_features(df, args.verbose)
        
        # Step 3: Generate labels
        df, y_direction, y_success = generate_labels_for_meta(
            df, 
            horizon=args.horizon,
            k=args.label_k,
            verbose=args.verbose
        )
        
        # Step 4: Train/test split
        X_train, X_test, y_dir_train, y_dir_test, y_succ_train, y_succ_test = prepare_train_test_split(
            df, y_direction, y_success,
            test_size=args.test_size,
            verbose=args.verbose
        )
        
        # Step 5: Train model
        primary_params = {
            'iterations': args.primary_iterations,
            'depth': args.primary_depth,
            'learning_rate': args.primary_lr
        }
        
        meta_params = {
            'iterations': args.meta_iterations,
            'depth': args.meta_depth,
            'learning_rate': args.meta_lr
        }
        
        meta_model = train_metamodel(
            X_train, X_test, y_dir_train, y_dir_test, y_succ_train, y_succ_test,
            primary_params, meta_params, args.threshold,
            verbose=args.verbose
        )
        
        # Step 6: Evaluate
        metrics = evaluate_model(
            meta_model, X_train, X_test, y_dir_train, y_dir_test,
            y_succ_train, y_succ_test,
            verbose=args.verbose
        )
        
        # Step 7: Save
        save_model(meta_model, args.output, args.verbose)
        
        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"   Model saved to: {args.output}")
        logger.info(f"   Primary accuracy: {metrics['primary_accuracy']:.4f}")
        logger.info(f"   Trades filtered: {metrics['n_filtered']} ({metrics['filter_rate']:.1f}%)")
        logger.info(f"   Final accuracy: {metrics['final_accuracy']:.4f}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    (project_root / 'logs').mkdir(exist_ok=True)
    
    main()
