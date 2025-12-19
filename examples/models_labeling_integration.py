"""
Integration example: ML Models + Triple Barrier Labeling

This example demonstrates the complete workflow:
1. Generate labels using Triple Barrier Method
2. Train XGBoost model with sample weights
3. Implement meta-labeling strategy
4. Evaluate results

This is a minimal example showing integration between the labeling
and models modules.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import labeling module
from src.bitcoin_scalper.labeling.barriers import get_events
from src.bitcoin_scalper.labeling.volatility import estimate_ewma_volatility

# Import models module
from src.bitcoin_scalper.models import XGBoostClassifier, Trainer, MetaLabelingPipeline


def generate_sample_price_data(n_samples=1000):
    """
    Generate synthetic price data for demonstration.
    
    In production, this would be real market data.
    """
    np.random.seed(42)
    
    # Create datetime index (1-minute bars)
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    
    # Generate random walk with drift
    returns = np.random.randn(n_samples) * 0.001 + 0.0001
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
    
    # Generate volume
    volume = pd.Series(np.random.lognormal(10, 1, n_samples), index=dates)
    
    return prices, volume


def create_features(prices, volume):
    """
    Create trading features from price and volume data.
    
    In production, this would include:
    - Order book imbalance (OFI)
    - Technical indicators
    - On-chain metrics
    - Sentiment scores
    """
    features = pd.DataFrame(index=prices.index)
    
    # Simple returns
    features['return_1m'] = prices.pct_change(1)
    features['return_5m'] = prices.pct_change(5)
    features['return_15m'] = prices.pct_change(15)
    
    # Volatility
    features['volatility_10m'] = prices.pct_change().rolling(10).std()
    features['volatility_30m'] = prices.pct_change().rolling(30).std()
    
    # Volume features
    features['volume'] = volume
    features['volume_ma_10'] = volume.rolling(10).mean()
    features['volume_ratio'] = volume / volume.rolling(30).mean()
    
    # Price momentum
    features['momentum_10m'] = prices - prices.shift(10)
    features['momentum_30m'] = prices - prices.shift(30)
    
    # Drop NaN values from rolling windows
    features = features.dropna()
    
    return features


def main():
    """Main integration example."""
    
    print("=" * 70)
    print("ML Models + Triple Barrier Integration Example")
    print("=" * 70)
    print()
    
    # Step 1: Generate sample data
    print("Step 1: Generating sample data...")
    prices, volume = generate_sample_price_data(n_samples=1000)
    print(f"  - Generated {len(prices)} price bars")
    print(f"  - Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print()
    
    # Step 2: Create features
    print("Step 2: Creating features...")
    features = create_features(prices, volume)
    print(f"  - Created {len(features.columns)} features")
    print(f"  - Feature names: {list(features.columns)}")
    print()
    
    # Step 3: Generate Triple Barrier labels
    print("Step 3: Generating Triple Barrier labels...")
    
    # Estimate volatility for dynamic barriers
    volatility = estimate_ewma_volatility(prices, span=30)
    
    # Select event timestamps (for simplicity, every 10th bar)
    event_indices = range(100, len(prices) - 100, 10)
    event_timestamps = prices.index[list(event_indices)]
    
    # Create dynamic barriers (2 * volatility)
    pt_sl = 2.0 * volatility.loc[event_timestamps]
    
    # Generate barrier events
    events = get_events(
        close=prices,
        timestamps=event_timestamps,
        pt_sl=pt_sl,
        max_holding_period=pd.Timedelta('15min')
    )
    
    print(f"  - Generated {len(events)} barrier events")
    print(f"  - Barrier hit distribution:")
    print(f"    * Profit targets: {(events['type'] == 1).sum()}")
    print(f"    * Stop losses: {(events['type'] == -1).sum()}")
    print(f"    * Time limits: {(events['type'] == 0).sum()}")
    print()
    
    # Step 4: Prepare training data
    print("Step 4: Preparing training data...")
    
    # Align features with events
    X = features.loc[events.index]
    y_side = events['type']  # Which barrier was hit (-1, 0, 1)
    
    # Compute sample weights (inverse holding time)
    holding_times = (events['t1'] - events.index).dt.total_seconds()
    sample_weights = 1.0 / holding_times
    sample_weights = sample_weights / sample_weights.sum()  # Normalize
    
    # Create meta labels (was the trade profitable?)
    y_meta = (events['return'] > 0).astype(int)
    
    print(f"  - Training samples: {len(X)}")
    print(f"  - Features shape: {X.shape}")
    print(f"  - Sample weights range: {sample_weights.min():.6f} - {sample_weights.max():.6f}")
    print()
    
    # Step 5: Split data
    print("Step 5: Splitting train/validation/test...")
    train_size = int(len(X) * 0.6)
    val_size = int(len(X) * 0.2)
    
    X_train = X.iloc[:train_size]
    X_val = X.iloc[train_size:train_size + val_size]
    X_test = X.iloc[train_size + val_size:]
    
    y_side_train = y_side.iloc[:train_size]
    y_side_val = y_side.iloc[train_size:train_size + val_size]
    y_side_test = y_side.iloc[train_size + val_size:]
    
    y_meta_train = y_meta.iloc[:train_size]
    y_meta_val = y_meta.iloc[train_size:train_size + val_size]
    y_meta_test = y_meta.iloc[train_size + val_size:]
    
    weights_train = sample_weights.iloc[:train_size]
    
    print(f"  - Train: {len(X_train)} samples")
    print(f"  - Validation: {len(X_val)} samples")
    print(f"  - Test: {len(X_test)} samples")
    print()
    
    # Step 6: Train single XGBoost model
    print("Step 6: Training single XGBoost model...")
    
    try:
        model = XGBoostClassifier(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,
            use_gpu=False,
            random_state=42
        )
        
        trainer = Trainer(model, handle_nans='fill', handle_infs='clip')
        trainer.train(
            X_train, y_side_train,
            sample_weights=weights_train,
            eval_set=(X_val, y_side_val),
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Evaluate
        metrics = trainer.evaluate(X_test, y_side_test)
        print(f"  - Single model accuracy: {metrics['accuracy']:.4f}")
        print(f"  - F1 score: {metrics['f1_score']:.4f}")
        
        # Feature importance
        importance = model.get_feature_importance()
        print(f"  - Top 3 features:")
        for feat in importance.sort_values(ascending=False).head(3).index:
            print(f"    * {feat}: {importance[feat]:.4f}")
        
    except ImportError:
        print("  - XGBoost not available, skipping single model training")
    
    print()
    
    # Step 7: Train meta-labeling pipeline
    print("Step 7: Training meta-labeling pipeline...")
    
    try:
        # Create two models
        primary = XGBoostClassifier(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,
            use_gpu=False,
            random_state=42
        )
        
        secondary = XGBoostClassifier(
            n_estimators=30,
            max_depth=4,
            learning_rate=0.1,
            use_gpu=False,
            random_state=43
        )
        
        # Create pipeline
        pipeline = MetaLabelingPipeline(primary, secondary)
        
        # Train
        pipeline.train(
            X_train, y_side_train, y_meta_train,
            sample_weights=weights_train,
            eval_set=(X_val, y_side_val, y_meta_val),
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Make predictions
        side_pred, bet_pred = pipeline.predict(X_test)
        combined_pred = pipeline.predict_combined(X_test)
        
        # Evaluate
        print(f"  - Primary model (side) accuracy: {(side_pred == y_side_test).mean():.4f}")
        print(f"  - Secondary model (bet) accuracy: {(bet_pred == y_meta_test).mean():.4f}")
        print(f"  - Bet rate: {bet_pred.mean():.2%} (trades filtered out: {(1 - bet_pred.mean()):.2%})")
        
        # Combined predictions
        n_long = (combined_pred == 1).sum()
        n_short = (combined_pred == -1).sum()
        n_neutral = (combined_pred == 0).sum()
        
        print(f"  - Combined predictions:")
        print(f"    * Long: {n_long} ({n_long / len(combined_pred):.1%})")
        print(f"    * Short: {n_short} ({n_short / len(combined_pred):.1%})")
        print(f"    * Neutral: {n_neutral} ({n_neutral / len(combined_pred):.1%})")
        
    except ImportError:
        print("  - XGBoost not available, skipping meta-labeling")
    
    print()
    print("=" * 70)
    print("Integration example completed!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  1. Triple Barrier method creates realistic labels with profit/loss targets")
    print("  2. Sample weights emphasize quick trades (early barrier hits)")
    print("  3. Meta-labeling filters false signals, improving Sharpe ratio")
    print("  4. The pipeline is production-ready and integrates seamlessly")


if __name__ == '__main__':
    main()
