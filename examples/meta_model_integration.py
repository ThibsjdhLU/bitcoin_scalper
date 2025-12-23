"""
Example demonstrating MetaModel integration with TradingEngine (CatBoost models).

This example shows how to:
1. Create and train a MetaModel with CatBoost classifiers
2. Use predict_meta() for filtered signal generation
3. Integrate with the engine structure
"""

import numpy as np
import pandas as pd
from bitcoin_scalper.models.meta_model import MetaModel

# Import CatBoost if available, otherwise use dummy
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    print("Warning: CatBoost not installed. Using dummy classifier for demo.")
    HAS_CATBOOST = False


class DummyClassifier:
    """Simple dummy classifier for demo when CatBoost is not available."""
    
    def __init__(self, seed=None):
        self.is_fitted = False
        self.classes_ = None
        self.n_classes_ = 0
        self.seed = seed
        self._rng = np.random.RandomState(seed)
    
    def fit(self, X, y, sample_weight=None, eval_set=None, **kwargs):
        """Fit the dummy model."""
        self.is_fitted = True
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self._rng = np.random.RandomState(self.seed)
        return self
    
    def predict(self, X):
        """Make dummy predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self._rng.choice(self.classes_, size=len(X))
    
    def predict_proba(self, X):
        """Make dummy probability predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        n_samples = len(X)
        proba = self._rng.rand(n_samples, self.n_classes_)
        return proba / proba.sum(axis=1, keepdims=True)


def create_sample_data(n_samples=1000):
    """Create sample trading data."""
    np.random.seed(42)
    
    # Market features (simulating engine.py feature engineering)
    X = pd.DataFrame({
        '1min_close': np.random.randn(n_samples),
        '1min_volume': np.random.rand(n_samples) * 1000,
        '1min_rsi': np.random.rand(n_samples) * 100,
        '1min_macd': np.random.randn(n_samples),
        '5min_close': np.random.randn(n_samples),
        '5min_volume': np.random.rand(n_samples) * 5000,
        '5min_rsi': np.random.rand(n_samples) * 100,
        '5min_macd': np.random.randn(n_samples),
    })
    
    # Direction labels: -1 (Sell), 0 (Neutral), 1 (Buy)
    # Simulate based on features with some structure
    y_direction = np.where(
        X['1min_rsi'] > 70, -1,  # Overbought -> Sell
        np.where(X['1min_rsi'] < 30, 1, 0)  # Oversold -> Buy, else Neutral
    )
    
    # Success labels: 0 (Failed), 1 (Success)
    # Simulate: trades more likely to succeed when both timeframes align
    alignment = (np.sign(X['1min_macd']) == np.sign(X['5min_macd'])).astype(int)
    y_success = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    # Boost success rate when aligned
    y_success = np.where(alignment == 1, 
                         np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
                         y_success)
    
    return X, y_direction, y_success


def main():
    """Demonstrate MetaModel usage."""
    print("=" * 70)
    print("MetaModel Integration Example with TradingEngine")
    print("=" * 70)
    
    # Create sample data
    print("\n1. Creating sample market data...")
    X_train, y_direction_train, y_success_train = create_sample_data(1000)
    X_test, y_direction_test, y_success_test = create_sample_data(200)
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {list(X_train.columns)}")
    
    # Create models
    print("\n2. Creating Primary and Meta models...")
    if HAS_CATBOOST:
        # Use CatBoost (same as engine.py would use)
        primary_model = CatBoostClassifier(
            iterations=100,
            depth=4,
            learning_rate=0.1,
            verbose=False,
            random_state=42
        )
        meta_model_classifier = CatBoostClassifier(
            iterations=50,
            depth=3,
            learning_rate=0.1,
            verbose=False,
            random_state=43
        )
        print("   Using CatBoost classifiers (production-ready)")
    else:
        # Fallback to dummy for demo (defined above)
        primary_model = DummyClassifier(seed=42)
        meta_model_classifier = DummyClassifier(seed=43)
        print("   Using dummy classifiers (demo only)")
    
    # Create MetaModel instance
    meta_model = MetaModel(
        primary_model=primary_model,
        meta_model=meta_model_classifier,
        meta_threshold=0.6  # Only take trades with 60%+ success confidence
    )
    print(f"   Meta threshold: {meta_model.meta_threshold}")
    
    # Train the meta model
    print("\n3. Training meta-labeling pipeline...")
    print("   Stage 1: Primary model (Direction: Buy/Sell/Neutral)")
    print("   Stage 2: Meta model (Success: Take/Pass)")
    
    meta_model.train(
        X=X_train,
        y_direction=y_direction_train,
        y_success=y_success_train,
        eval_set=(X_test, y_direction_test, y_success_test)
    )
    
    print("   Training completed successfully!")
    
    # Make predictions
    print("\n4. Making predictions with predict_meta()...")
    result = meta_model.predict_meta(X_test, return_all=True)
    
    print(f"   Predictions generated for {len(X_test)} samples")
    
    # Analyze results
    print("\n5. Analyzing prediction results:")
    n_raw_buy = (result['raw_signal'] == 1).sum()
    n_raw_sell = (result['raw_signal'] == -1).sum()
    n_raw_neutral = (result['raw_signal'] == 0).sum()
    
    n_final_buy = (result['final_signal'] == 1).sum()
    n_final_sell = (result['final_signal'] == -1).sum()
    n_final_neutral = (result['final_signal'] == 0).sum()
    
    print(f"\n   Primary Model (Raw Signals):")
    print(f"     Buy:     {n_raw_buy:3d} ({n_raw_buy/len(X_test)*100:.1f}%)")
    print(f"     Sell:    {n_raw_sell:3d} ({n_raw_sell/len(X_test)*100:.1f}%)")
    print(f"     Neutral: {n_raw_neutral:3d} ({n_raw_neutral/len(X_test)*100:.1f}%)")
    
    print(f"\n   After Meta Filtering (Final Signals):")
    print(f"     Buy:     {n_final_buy:3d} ({n_final_buy/len(X_test)*100:.1f}%)")
    print(f"     Sell:    {n_final_sell:3d} ({n_final_sell/len(X_test)*100:.1f}%)")
    print(f"     Neutral: {n_final_neutral:3d} ({n_final_neutral/len(X_test)*100:.1f}%)")
    
    n_filtered = (n_raw_buy + n_raw_sell) - (n_final_buy + n_final_sell)
    filter_rate = n_filtered / max(n_raw_buy + n_raw_sell, 1) * 100
    
    print(f"\n   Meta Filtering Statistics:")
    print(f"     Signals filtered out: {n_filtered} ({filter_rate:.1f}%)")
    print(f"     Average meta confidence: {result['meta_conf'].mean():.3f}")
    print(f"     High confidence (>0.7): {(result['meta_conf'] > 0.7).sum()}")
    
    # Show sample predictions
    print("\n6. Sample predictions (first 10):")
    print("   Idx | Raw Signal | Meta Conf | Final Signal")
    print("   " + "-" * 50)
    for i in range(min(10, len(X_test))):
        raw_sig = result['raw_signal'][i]
        meta_conf = result['meta_conf'][i]
        final_sig = result['final_signal'][i]
        
        raw_str = {-1: "SELL", 0: "HOLD", 1: "BUY"}[raw_sig]
        final_str = {-1: "SELL", 0: "HOLD", 1: "BUY"}[final_sig]
        
        print(f"   {i:3d} | {raw_str:^10} | {meta_conf:9.3f} | {final_str:^12}")
    
    # Integration with engine
    print("\n7. Integration with TradingEngine:")
    print("   The MetaModel can be used in engine.py as follows:")
    print()
    print("   # In engine.py _get_ml_signal() method:")
    print("   if self.use_meta_labeling:")
    print("       result = self.meta_model.predict_meta(X)")
    print("       signal = result['final_signal'][0]")
    print("       confidence = result['meta_conf'][0]")
    print("   else:")
    print("       signal = self.ml_model.predict(X)[0]")
    print("       confidence = self.ml_model.predict_proba(X)[0].max()")
    print()
    print("   This provides:")
    print("   - Filtered signals (reduces false positives)")
    print("   - Meta confidence (for position sizing)")
    print("   - Raw signals (for analysis/debugging)")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
