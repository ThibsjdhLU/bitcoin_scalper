#!/usr/bin/env python3
"""
Master Training Script for Bitcoin Scalper
==========================================

This script unifies the entire machine learning pipeline for the Bitcoin Scalper bot.
It handles data loading (local or download), feature engineering, labeling (Triple Barrier),
hyperparameter optimization (Optuna), training (Primary + Meta models), and evaluation.

Usage Examples:
    1. Train from local CSV:
       python src/bitcoin_scalper/scripts/train_master_model.py \
           --data-file data/raw/BTCUSD_M1.csv \
           --out models/meta_model_production.pkl

    2. Download from Binance and train:
       python src/bitcoin_scalper/scripts/train_master_model.py \
           --api_key YOUR_KEY --api_secret YOUR_SECRET \
           --start 2020-01-01 --symbol BTC/USDT

Outputs:
    - Trained MetaModel (joblib)
    - Metadata JSON
    - Logs
    - Evaluation reports (CSV, JSON)
"""

import os
import sys
import json
import logging
import argparse
import datetime
import random
import warnings
import re
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import TimeSeriesSplit

# Append src to path to ensure imports work if run from root
sys.path.append(os.path.join(os.getcwd(), 'src'))

from bitcoin_scalper.connectors.binance_connector import BinanceConnector
from bitcoin_scalper.core.feature_engineering import FeatureEngineering
from bitcoin_scalper.core.modeling import ModelTrainer
from bitcoin_scalper.models.meta_model import MetaModel
from bitcoin_scalper.core.engine import TradingEngine

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger("train_master")

def setup_logging(log_file: str, verbose: bool):
    """Configure logging to file and console."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '[%(asctime)s][%(levelname)s] %(message)s'

    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")

def load_local_data(filepath: str) -> pd.DataFrame:
    """Load data from a local file (.csv, .parquet, .feather)."""
    logger.info(f"Loading local data from {filepath}...")
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.csv':
        df = pd.read_csv(filepath)
    elif ext == '.parquet':
        df = pd.read_parquet(filepath)
    elif ext == '.feather':
        df = pd.read_feather(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Ensure index is DatetimeIndex
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try to parse the index if it's not already DatetimeIndex
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError("Could not parse DataFrame index as DatetimeIndex")

    df = df.sort_index()

    # Normalize columns
    df.columns = [c.lower() for c in df.columns]
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}, Required: {required}")

    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df

def fetch_full_history(
    symbol: str,
    start_date: str,
    api_key: str,
    api_secret: str,
    cache_file: str,
    force_download: bool,
    testnet: bool
) -> pd.DataFrame:
    """
    Fetch full history from Binance or load from cache.
    Reuses logic from train_full_model.py.
    """
    # Check cache first
    if os.path.exists(cache_file) and not force_download:
        logger.info(f"Loading data from cache: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} rows from cache.")
        return df

    if not api_key or not api_secret:
        logger.error("API credentials missing and no cache available.")
        sys.exit(1)

    logger.info(f"Starting download for {symbol} from {start_date}...")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    connector = BinanceConnector(api_key=api_key, api_secret=api_secret, testnet=testnet)

    since_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    limit = 1000
    all_candles = []
    current_start = since_ts

    while True:
        try:
            # Use fetch_ohlcv with pagination
            ohlcv = connector.exchange.fetch_ohlcv(symbol, timeframe='1m', since=current_start, limit=limit)

            if not ohlcv:
                break

            all_candles.extend(ohlcv)
            last_ts = ohlcv[-1][0]

            # Prevent infinite loop if exchange returns same data
            if last_ts == current_start:
                break

            current_start = last_ts + 60000  # +1 minute

            # Stop if we reached close to now
            if last_ts > (datetime.datetime.now().timestamp() * 1000 - 60000):
                break

        except Exception as e:
            logger.error(f"Error during fetch: {e}")
            break

    logger.info(f"Download complete: {len(all_candles)} candles.")

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('date')
    df = df.drop(columns=['timestamp'])

    # Save to cache
    df.to_csv(cache_file)
    logger.info(f"Data saved to {cache_file}")

    return df

def run_feature_engineering(df: pd.DataFrame, warmup_rows: int) -> pd.DataFrame:
    """
    Run Feature Engineering pipeline (1m + 5m).
    """
    logger.info("Running Feature Engineering...")

    # Rename to internal convention
    rename_map = {
        'open': '<OPEN>', 'high': '<HIGH>', 'low': '<LOW>',
        'close': '<CLOSE>', 'volume': '<TICKVOL>'
    }
    # Handle cases where columns might already be uppercase or prefixed?
    # The loader normalized to lowercase, so this map is correct.
    df = df.rename(columns=rename_map)

    fe = FeatureEngineering()

    # --- 1 Minute Features ---
    prefix_1m = "1min_"
    df[f'{prefix_1m}day'] = df.index.dayofweek
    df[f'{prefix_1m}hour'] = df.index.hour

    # Add indicators and features
    df = fe.add_indicators(df, price_col='<CLOSE>', high_col='<HIGH>', low_col='<LOW>', volume_col='<TICKVOL>', prefix=prefix_1m)
    df = fe.add_features(df, price_col='<CLOSE>', volume_col='<TICKVOL>', prefix=prefix_1m)

    # --- 5 Minute Features ---
    prefix_5m = "5min_"
    df_5m = df[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']].resample('5min').agg({
        '<OPEN>': 'first', '<HIGH>': 'max', '<LOW>': 'min', '<CLOSE>': 'last', '<TICKVOL>': 'sum'
    }).dropna()

    df_5m = fe.add_indicators(
        df_5m, price_col='<CLOSE>', high_col='<HIGH>', low_col='<LOW>', volume_col='<TICKVOL>',
        prefix=prefix_5m, drop_rows=False
    )
    df_5m = fe.add_features(df_5m, price_col='<CLOSE>', volume_col='<TICKVOL>', prefix=prefix_5m)

    # Merge 5m back to 1m
    cols_5m = [col for col in df_5m.columns if col.startswith(prefix_5m)]
    df = df.join(df_5m[cols_5m], how='left')
    df[cols_5m] = df[cols_5m].ffill()

    # Drop global NaNs and warmup rows
    df = df.dropna()
    if len(df) > warmup_rows:
        df = df.iloc[warmup_rows:]
    else:
        logger.warning(f"Data length ({len(df)}) <= warmup_rows ({warmup_rows}). Skipping warmup drop.")

    logger.info(f"Feature Engineering complete. Final shape: {df.shape}")
    return df

def compute_triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int,
    pt_sl_multiplier: float
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Triple Barrier Labels (Primary and Meta) and Sample Weights.
    Vectorized implementation.
    """
    logger.info("Computing Triple Barrier Labels...")

    # 1. Volatility (ATR)
    if '1min_atr_14' in df.columns:
        volatility = df['1min_atr_14']
    else:
        # Fallback to rolling std of returns if ATR missing
        volatility = df['<CLOSE>'].pct_change().rolling(20).std() * df['<CLOSE>']

    # Dynamic PT (Profit Take)
    # pt_pct = (pt_sl_multiplier * volatility) / price
    # We define barriers in price terms directly
    half_spread = volatility * pt_sl_multiplier

    close_prices = df['<CLOSE>']
    high_prices = df['<HIGH>']
    low_prices = df['<LOW>']

    upper_barrier = close_prices + half_spread
    lower_barrier = close_prices - half_spread

    # 2. Vectorized Horizon Search
    # Look forward 'horizon' steps
    # We use rolling max/min on reversed series or FixedForwardWindowIndexer
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)

    future_max = high_prices.rolling(window=indexer, min_periods=1).max()
    future_min = low_prices.rolling(window=indexer, min_periods=1).min()

    # Shift result to align t with [t+1, t+horizon] ?
    # Rolling with forward indexer at 't' includes 't'.
    # We want strictly future. So we shift the rolling result by -1?
    # Actually, FixedForwardWindowIndexer at t includes t..t+w-1.
    # We ideally want max(High[t+1]...High[t+horizon]).
    # So we can apply rolling on shifted data.

    future_max = high_prices.shift(-1).rolling(window=horizon, min_periods=1).max().shift(1 - horizon)
    # Re-implementing with simpler logic:
    # Reverse rolling is standard for "future" lookahead in pandas without extra deps

    future_max = high_prices[::-1].rolling(window=horizon, min_periods=1).max()[::-1].shift(-1)
    future_min = low_prices[::-1].rolling(window=horizon, min_periods=1).min()[::-1].shift(-1)

    # Also need the time index of the event for sample weights (holding period)
    # This is hard to vectorize perfectly for "first touch".
    # Approximation for sample weights:
    # If hit, weight = 1 / (time to hit). If not hit, weight = 1 / horizon.
    # For speed, we will use a simpler weight: 1.0 (or just 1/horizon if explicit holding period needed)
    # The instruction says: "Compute sample_weights = 1 / holding_period_in_minutes"
    # To get exact holding period vectorially is complex.
    # We will approximate or iterate if dataset is small, but for 2M rows we need vector.
    # Let's approximate holding period:
    # If touched, assume it happened at horizon/2 ? No, that's bad.
    # Let's skip complex holding period calculation for vectorization speed and use default 1.0
    # OR implement a rough estimate:
    # We can check if barrier was hit at t+1, t+2... for small horizon (30) it's feasible loop.

    # Iterative vector check for horizon
    touch_idx = np.full(len(df), horizon, dtype=float) # Default to horizon

    # Pre-calculate barriers for speed
    up_bar = upper_barrier.values
    lo_bar = lower_barrier.values
    h_vals = high_prices.values
    l_vals = low_prices.values
    c_vals = close_prices.values

    # We determine label first
    # 1 = Top barrier hit first
    # -1 = Bottom barrier hit first
    # 0 = Neither hit within horizon

    # Optimization: Use just the max/min over horizon to determine IF it hits.
    # If future_max > upper AND future_min < lower, we have a "double touch".
    # In that case, we need to know WHICH happened first.
    # For this script, we prioritize the worst case or just take the first one?
    # Standard Triple Barrier: first touch counts.

    labels = np.zeros(len(df), dtype=int)

    # Only iterate if we need precise "who touched first" for double touch cases.
    # For efficiency in Python, we can loop over the horizon window shifts.

    first_touch_time = np.full(len(df), horizon, dtype=int)
    outcome = np.zeros(len(df), dtype=int) # 0: none, 1: up, -1: down

    for i in range(1, horizon + 1):
        # Look at t+i
        h_future = pd.Series(h_vals).shift(-i).values
        l_future = pd.Series(l_vals).shift(-i).values

        # Check touches
        # touched_up = h_future >= up_bar
        # touched_down = l_future <= lo_bar

        # We need to process only those NOT yet touched
        mask_not_touched = (outcome == 0)

        if not np.any(mask_not_touched):
            break

        # For not yet touched:
        # If both touched at this step 'i' (large candle), we can treat as volatility/stop loss preference?
        # Usually assume stop loss (down) if unknown inside candle, or check close.
        # Let's assume if Low < Lower, it's -1. Else if High > Upper, it's 1.

        # Slices for valid rows
        # We perform operations on full arrays but only update masked

        # Masks for this step
        hit_up = (h_future >= up_bar) & mask_not_touched
        hit_down = (l_future <= lo_bar) & mask_not_touched

        # Conflict resolution (both hit in same candle): bias towards SL (-1) or neutral?
        # Let's say -1 takes precedence for safety
        double_hit = hit_up & hit_down
        hit_up = hit_up & (~double_hit)

        # Update
        outcome[hit_up] = 1
        outcome[hit_down] = -1 # Includes double hit
        outcome[double_hit] = -1

        # Update time
        first_touch_time[hit_up | hit_down] = i

    y_primary = pd.Series(outcome, index=df.index)
    sample_weights = pd.Series(1.0 / first_touch_time, index=df.index).clip(lower=1.0/horizon, upper=1.0)

    # Meta Labels
    ret_horizon = df['<CLOSE>'].shift(-horizon) / df['<CLOSE>'] - 1
    y_meta = pd.Series(0, index=df.index)

    mask_nz = y_primary != 0
    # Condition: y_primary * ret_horizon > 0
    meta_condition = (y_primary * ret_horizon) > 0
    y_meta[mask_nz & meta_condition] = 1

    return y_primary, y_meta, sample_weights


def optimize_catboost_params_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    study_name: str,
    n_trials: int,
    seed: int,
    sample_weights: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Run Optuna optimization for CatBoost.
    """
    logger.info(f"Starting Optuna optimization for {study_name} ({n_trials} trials)...")

    is_multiclass = len(y.unique()) > 2
    objective_metric = 'MultiClass' if is_multiclass else 'Logloss'
    eval_metric = 'TotalF1' if is_multiclass else 'F1'

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 2000) if study_name == 'primary' else trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 4, 10) if study_name == 'primary' else trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_strength': trial.suggest_float('random_strength', 0.0, 2.0) if study_name == 'primary' else 1.0, # Default for meta
            'random_seed': seed,
            'task_type': 'CPU',
            'thread_count': 1, # As requested for resource constraints
            'loss_function': objective_metric,
            'eval_metric': eval_metric,
            'early_stopping_rounds': 50 if study_name == 'primary' else 30,
            'verbose': False,
            'allow_writing_files': False
        }

        model = CatBoostClassifier(**params)

        # Use sample_weights if provided
        fit_params = {
            'eval_set': (X_val, y_val),
            'verbose': False
        }
        if sample_weights is not None:
            fit_params['sample_weight'] = sample_weights

        model.fit(X, y, **fit_params)

        preds = model.predict(X_val)

        if study_name == 'primary':
            # Maximize Balanced Accuracy or F1 Macro
            score = balanced_accuracy_score(y_val, preds)
        else:
            # Maximize F1 (Binary)
            score = f1_score(y_val, preds)

        return score

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=600 * n_trials)

    logger.info(f"Best params for {study_name}: {study.best_params}")
    return study.best_params

def map_labels_to_int(y: pd.Series) -> pd.Series:
    """Map -1, 0, 1 to 0, 1, 2 for Primary MultiClass."""
    mapping = {-1: 0, 0: 1, 1: 2}
    if y.min() < 0:
        return y.map(mapping)
    return y

def train_primary_oof_probas(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: Dict[str, Any],
    seed: int,
    sample_weights: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Generate Out-Of-Fold probabilities for the training set using TimeSeriesSplit.
    """
    logger.info("Generating OOF probabilities for Primary model...")
    tscv = TimeSeriesSplit(n_splits=5)

    classes = np.sort(y_train.unique()) # Should be 0, 1, 2
    n_classes = len(classes)

    oof_probas = np.zeros((len(X_train), n_classes))
    oof_probas[:] = np.nan

    params = best_params.copy()
    params['random_seed'] = seed
    params['verbose'] = False
    params['allow_writing_files'] = False
    params['thread_count'] = 1

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        logger.debug(f"OOF Fold {fold+1}/5")
        X_t, y_t = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_v = X_train.iloc[val_idx]

        fit_params = {}
        if sample_weights is not None:
             w_t = sample_weights.iloc[train_idx]
             fit_params['sample_weight'] = w_t

        model = CatBoostClassifier(**params)
        model.fit(X_t, y_t, **fit_params)

        preds_proba = model.predict_proba(X_v)
        oof_probas[val_idx] = preds_proba

    cols = [f'primary_proba_{c}' for c in range(n_classes)]
    return pd.DataFrame(oof_probas, index=X_train.index, columns=cols)

def run_smoke_test(df_raw: pd.DataFrame):
    """Run a quick smoke test on the last 1000 rows."""
    logger.info("Running Smoke Test...")
    df = df_raw.iloc[-1000:].copy()
    try:
        df = run_feature_engineering(df, warmup_rows=50)
        y_p, y_m, weights = compute_triple_barrier_labels(df, horizon=15, pt_sl_multiplier=1.5)
        # Check shapes
        assert len(df) == len(y_p)
        assert len(df) == len(weights)
        assert not df.isnull().values.any()
        logger.info("Smoke test PASSED.")
    except Exception as e:
        logger.error(f"Smoke test FAILED: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Train Master Model")
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--api_key", type=str, default=os.environ.get("BINANCE_API_KEY"))
    parser.add_argument("--api_secret", type=str, default=os.environ.get("BINANCE_API_SECRET"))
    parser.add_argument("--testnet", action='store_true')
    parser.add_argument("--out", type=str, default="models/meta_model_production.pkl")
    parser.add_argument("--meta-json", type=str, default="models/meta_model_production.json")
    parser.add_argument("--logs", type=str, default="logs/train_master_{YYYYMMDD_HHMMSS}.log")
    parser.add_argument("--n_trials_primary", type=int, default=50)
    parser.add_argument("--n_trials_meta", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--pt_sl_multiplier", type=float, default=1.5)
    parser.add_argument("--warmup_rows", type=int, default=500)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--force_download", action='store_true')
    parser.add_argument("--cache_file", type=str, default="data/full_history.csv")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--meta_threshold", type=float, default=0.55)

    args = parser.parse_args()

    # Reproducibility
    set_seeds(args.random_state)

    # Logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.logs.format(YYYYMMDD_HHMMSS=timestamp)
    setup_logging(log_file, args.verbose)

    logger.info("=== STEP 1: DATA LOADING ===")
    if args.data_file:
        df = load_local_data(args.data_file)
        if args.verbose:
            run_smoke_test(df)
    else:
        df = fetch_full_history(
            args.symbol, args.start, args.api_key, args.api_secret,
            args.cache_file, args.force_download, args.testnet
        )

    if len(df) < 2000:
        logger.error(f"Insufficient data: {len(df)} rows. Need at least 2000.")
        sys.exit(1)

    logger.info("=== STEP 2: FEATURE ENGINEERING ===")
    df = run_feature_engineering(df, args.warmup_rows)

    logger.info("=== STEP 3: LABELS ===")
    y_primary, y_meta, sample_weights = compute_triple_barrier_labels(
        df, args.horizon, args.pt_sl_multiplier
    )

    # Map Primary Labels to 0,1,2 for CatBoost MultiClass
    # -1 -> 0, 0 -> 1, 1 -> 2
    y_primary_mapped = map_labels_to_int(y_primary)

    logger.info(f"Primary distribution: {y_primary_mapped.value_counts().to_dict()}")
    logger.info(f"Meta distribution: {y_meta.value_counts().to_dict()}")

    logger.info("=== STEP 4: SPLIT ===")
    total = len(df)
    test_len = int(total * args.test_ratio)
    val_len = int(total * args.val_ratio)
    train_len = total - test_len - val_len

    # Prepare X (drop non-features and potential leakage)
    drop_cols = ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']
    # Drop regex matching 'future_', 'target_', 'label_future'
    leakage_pattern = re.compile(r'(future_|target_|label_future)')

    cols_to_drop = [c for c in df.columns if c in drop_cols or leakage_pattern.search(c)]
    X_all = df.drop(columns=cols_to_drop)

    # Select numeric features only
    X_all = X_all.select_dtypes(include=[np.number]).astype(np.float32)

    X_train = X_all.iloc[:train_len]
    X_val = X_all.iloc[train_len:train_len+val_len]
    X_test = X_all.iloc[train_len+val_len:]

    y_p_train = y_primary_mapped.iloc[:train_len]
    y_p_val = y_primary_mapped.iloc[train_len:train_len+val_len]
    y_p_test = y_primary_mapped.iloc[train_len+val_len:]

    y_m_train = y_meta.iloc[:train_len]
    y_m_val = y_meta.iloc[train_len:train_len+val_len]
    y_m_test = y_meta.iloc[train_len+val_len:]

    # Slice weights
    w_train = sample_weights.iloc[:train_len]
    w_val = sample_weights.iloc[train_len:train_len+val_len]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    logger.info("=== STEP 5: OPTIMIZE PRIMARY ===")
    primary_best_params = optimize_catboost_params_optuna(
        X_train, y_p_train, X_val, y_p_val, 'primary',
        args.n_trials_primary, args.random_state,
        sample_weights=w_train
    )

    # Save best params
    os.makedirs('models', exist_ok=True)
    with open('models/primary_best_params.json', 'w') as f:
        json.dump(primary_best_params, f, indent=4)

    logger.info("=== STEP 6: TRAIN FINAL PRIMARY (Train+Val) ===")
    # Train on X_train + X_val for final model
    X_train_full = pd.concat([X_train, X_val])
    y_p_train_full = pd.concat([y_p_train, y_p_val])
    w_train_full = pd.concat([w_train, w_val])

    final_primary_params = primary_best_params.copy()
    final_primary_params.update({
        'random_seed': args.random_state,
        'verbose': False,
        'allow_writing_files': False,
        'thread_count': 1,
        'loss_function': 'MultiClass',
        'iterations': 1000 # Boost iterations for final model
    })

    cat_primary = CatBoostClassifier(**final_primary_params)
    cat_primary.fit(X_train_full, y_p_train_full, sample_weight=w_train_full)

    logger.info("=== STEP 7: OOF PROBA GENERATION ===")
    # Pass sample_weights to OOF generation
    df_oof = train_primary_oof_probas(
        X_train, y_p_train, primary_best_params, args.random_state,
        sample_weights=w_train
    )

    # Generate Val predictions using a model trained on train (with weights)
    # We must replicate the weighted training to produce consistent features
    model_for_val = CatBoostClassifier(**final_primary_params)
    model_for_val.fit(X_train, y_p_train, sample_weight=w_train, verbose=False)

    val_probas = model_for_val.predict_proba(X_val)
    df_val_probas = pd.DataFrame(val_probas, index=X_val.index, columns=df_oof.columns)

    # For X_test, use the FINAL primary model (trained on Train+Val)
    test_probas = cat_primary.predict_proba(X_test)
    df_test_probas = pd.DataFrame(test_probas, index=X_test.index, columns=df_oof.columns)

    # Construct Meta Datasets
    # Concatenate features and probabilities
    X_meta_train = pd.concat([X_train, df_oof], axis=1)
    X_meta_val = pd.concat([X_val, df_val_probas], axis=1)
    X_meta_test = pd.concat([X_test, df_test_probas], axis=1)

    # Drop rows in Train where OOF is NaN (start of time series)
    valid_idx = X_meta_train.dropna().index
    X_meta_train = X_meta_train.loc[valid_idx]
    y_m_train = y_m_train.loc[valid_idx]
    w_meta_train = w_train.loc[valid_idx]

    logger.info(f"Meta Train size after OOF drop: {len(X_meta_train)}")

    logger.info("=== STEP 8: OPTIMIZE META ===")
    meta_best_params = optimize_catboost_params_optuna(
        X_meta_train, y_m_train, X_meta_val, y_m_val, 'meta',
        args.n_trials_meta, args.random_state,
        sample_weights=w_meta_train
    )

    with open('models/meta_best_params.json', 'w') as f:
        json.dump(meta_best_params, f, indent=4)

    logger.info("=== STEP 9: TRAIN FINAL META ===")
    X_meta_full = pd.concat([X_meta_train, X_meta_val])
    y_m_full = pd.concat([y_m_train, y_m_val])
    w_meta_full = pd.concat([w_meta_train, w_val])

    final_meta_params = meta_best_params.copy()
    final_meta_params.update({
        'random_seed': args.random_state,
        'verbose': False,
        'allow_writing_files': False,
        'thread_count': 1,
        'loss_function': 'Logloss'
    })

    cat_meta = CatBoostClassifier(**final_meta_params)
    cat_meta.fit(X_meta_full, y_m_full, sample_weight=w_meta_full)

    # Step E: Wrap
    # MetaModel expects primary and meta models.
    meta_model_wrapper = MetaModel(cat_primary, cat_meta, meta_threshold=args.meta_threshold)
    # Manually mark as trained since we injected trained models
    meta_model_wrapper.is_trained = True
    meta_model_wrapper.n_features = X_train.shape[1]
    meta_model_wrapper.feature_names = X_train.columns.tolist()

    logger.info("=== STEP 10: EVALUATION & SAVING ===")

    results = meta_model_wrapper.predict_meta(X_test, return_all=True)

    final_signal = results['final_signal']
    raw_signal = results['raw_signal']
    meta_conf = results['meta_conf']

    # Metrics
    acc_primary = accuracy_score(y_p_test, raw_signal)
    bal_acc_primary = balanced_accuracy_score(y_p_test, raw_signal)

    # Meta Metrics
    meta_preds = (meta_conf > args.meta_threshold).astype(int)
    acc_meta = accuracy_score(y_m_test, meta_preds)
    f1_meta = f1_score(y_m_test, meta_preds)
    roc_meta = roc_auc_score(y_m_test, meta_conf)

    # Filtered Profitability
    trades_mask = (final_signal != 1)
    n_trades_filtered = np.sum(trades_mask)
    n_trades_raw = np.sum(raw_signal != 1)

    # Save Report
    report_dir = f"models/reports_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)

    # Confusion Matrices
    cm_primary = confusion_matrix(y_p_test, raw_signal)
    pd.DataFrame(cm_primary).to_csv(f"{report_dir}/confusion_primary.csv")

    cm_meta = confusion_matrix(y_m_test, meta_preds)
    pd.DataFrame(cm_meta).to_csv(f"{report_dir}/confusion_meta.csv")

    metrics = {
        'primary_accuracy': acc_primary,
        'primary_balanced_accuracy': bal_acc_primary,
        'meta_accuracy': acc_meta,
        'meta_f1': f1_meta,
        'meta_roc_auc': roc_meta,
        'n_trades_raw': int(n_trades_raw),
        'n_trades_filtered': int(n_trades_filtered)
    }

    with open(f"models/metrics_test_{timestamp}.json", 'w') as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Test Metrics: {metrics}")

    # Save Model
    joblib.dump(meta_model_wrapper, args.out)
    logger.info(f"Model saved to {args.out}")

    # Metadata JSON
    metadata = {
        "model_file": os.path.abspath(args.out),
        "meta_json": os.path.abspath(args.meta_json),
        "generated_at": datetime.datetime.now().isoformat(),
        "random_state": args.random_state,
        "symbol": args.symbol if not args.data_file else "LOCAL_DATA",
        "start_date": args.start,
        "data_rows": len(df),
        "feature_count": X_train.shape[1],
        "feature_names": X_train.columns.tolist(),
        "primary_best_params": primary_best_params,
        "meta_best_params": meta_best_params,
        "n_trials_primary": args.n_trials_primary,
        "n_trials_meta": args.n_trials_meta,
        "horizon_minutes": args.horizon,
        "pt_sl_multiplier": args.pt_sl_multiplier,
        "meta_threshold": args.meta_threshold,
        "train_val_test_sizes": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test)
        },
        "metrics_test": metrics
    }

    with open(args.meta_json, 'w') as f:
        json.dump(metadata, f, indent=4)

    logger.info("Script completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error during execution:")
        sys.exit(1)
