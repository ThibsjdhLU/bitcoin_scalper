import argparse
import pandas as pd
import os
import sys
import logging
import joblib
from bitcoin_scalper.core.backtesting import Backtester
from bitcoin_scalper.core.modeling import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger("verify_model")

def main():
    parser = argparse.ArgumentParser(description="Check if a trained model is profitable on a given dataset.")
    parser.add_argument('--data', required=True, help='Path to processed dataframe pickle (must contain features and price column)')
    parser.add_argument('--model', required=True, help='Path to trained model file (joblib, cbm, or ModelTrainer dump)')
    parser.add_argument('--price_col', default='1min_<CLOSE>', help='Name of the price column in the dataframe')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--fee', type=float, default=0.0005, help='Trading fee (0.0005 = 0.05%%)')
    parser.add_argument('--slippage', type=float, default=0.0002, help='Slippage estimate')
    parser.add_argument('--out_dir', default='verification_report', help='Output directory for report')

    args = parser.parse_args()

    # 1. Load Data
    logger.info(f"Loading data from {args.data}...")
    try:
        df = pd.read_pickle(args.data)
        logger.info(f"Loaded {len(df)} rows.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Verify price column exists
    # Try fuzzy matching if exact match not found
    if args.price_col not in df.columns:
        candidates = [c for c in df.columns if 'CLOSE' in c.upper()]
        if candidates:
            logger.warning(f"Price column '{args.price_col}' not found. Using '{candidates[0]}' instead.")
            args.price_col = candidates[0]
        else:
            logger.error(f"Price column '{args.price_col}' not found and no obvious candidates.")
            sys.exit(1)

    # 2. Load Model
    logger.info(f"Loading model from {args.model}...")
    try:
        # Try loading as raw object
        model = joblib.load(args.model)

        # If it's a ModelTrainer object, extract the internal model if needed,
        # but Backtester expects an object with .predict(). ModelTrainer has .predict(), so it's fine.
        # If it's a raw CatBoost/XGBoost model, they also have .predict().
        logger.info(f"Model loaded: {type(model)}")
    except Exception as e:
        # Fallback for CatBoost specific loading if joblib fails (rare but possible for raw .cbm)
        try:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(args.model)
            logger.info("Loaded as CatBoostClassifier")
        except Exception as e2:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)

    # 3. Predict Signals
    logger.info("Generating predictions...")
    try:
        # Prepare features (exclude label/price/etc if known, or rely on model to ignore unknown columns if features are named)
        # Assuming the dataframe contains exactly what's needed or model handles it.
        # Ideally we should filter columns.

        # Identify non-feature columns commonly used in this project
        exclude_cols = ['label', 'signal', 'timestamp', 'date', 'time', args.price_col, 'log_return_1m', '1min_log_return']
        # Also exclude other OHLCV if they aren't features (heuristic)

        # Best approach: Assume model knows its features (CatBoost stores feature names).
        # Or pass all columns and let model decide/fail.
        # ModelTrainer.predict takes X_test.

        if hasattr(model, 'feature_names_'):
            # CatBoost/sklearn style
            feature_names = model.feature_names_
            X = df[feature_names]
        elif hasattr(model, 'feature_names_in_'):
            # sklearn style
            feature_names = model.feature_names_in_
            X = df[feature_names]
        else:
            # Fallback: drop known non-features
            logger.warning("Model does not expose feature names. Using all numeric columns except known targets.")
            X = df.select_dtypes(include=['number']).drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')

        preds = model.predict(X)

        # Handle if preds is not 1D array
        if hasattr(preds, 'ndim') and preds.ndim > 1:
            preds = preds.ravel()

        # Add signals to dataframe for backtester
        # Backtester expects a dataframe with 'signal' column.
        # We need to ensure we don't modify the original DF in a way that breaks things,
        # but Backtester copies it.

        # Create a df for backtester
        bt_df = df.copy()
        bt_df['signal'] = preds

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)

    # 4. Run Backtest
    logger.info("Running backtest...")
    backtester = Backtester(
        df=bt_df,
        signal_col='signal',
        price_col=args.price_col,
        initial_capital=args.capital,
        fee=args.fee,
        slippage=args.slippage,
        out_dir=args.out_dir
    )

    _, _, kpis, _ = backtester.run()

    # 5. Report
    profit_color = "\033[92m" if kpis['final_return'] > 0 else "\033[91m"
    reset_color = "\033[0m"

    print("\n" + "="*60)
    print(f"REPORT FOR MODEL: {os.path.basename(args.model)}")
    print(f"DATASET: {os.path.basename(args.data)}")
    print("-" * 60)
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Final Capital:   ${kpis['final_capital']:,.2f}")
    print(f"Return:          {profit_color}{kpis['final_return']*100:+.2f}%{reset_color}")
    print(f"Sharpe Ratio:    {kpis['sharpe']:.4f}")
    print(f"Max Drawdown:    {kpis['max_drawdown']:.2f}")
    print(f"Win Rate:        {kpis['win_rate']*100:.2f}% ({kpis['nb_trades']} trades)")
    print(f"Profit Factor:   {kpis['profit_factor']:.2f}")
    print("="*60 + "\n")

    if kpis['final_return'] > 0 and kpis['sharpe'] > 1.0:
        print("✅ VERDICT: MODEL SEEMS PROFITABLE")
    elif kpis['final_return'] > 0:
        print("⚠️ VERDICT: PROFITABLE BUT RISKY (Check Sharpe/Drawdown)")
    else:
        print("❌ VERDICT: MODEL IS LOSING MONEY")

if __name__ == "__main__":
    main()
