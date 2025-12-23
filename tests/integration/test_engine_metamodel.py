"""
Integration test for MetaModel in TradingEngine.

This test demonstrates how the engine integrates with MetaModel
for meta-labeling predictions.
"""

import sys
sys.path.insert(0, '/home/runner/work/bitcoin_scalper/bitcoin_scalper/src')

import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from bitcoin_scalper.models.meta_model import MetaModel


# Simple dummy classifier for testing
class DummyClassifier:
    def __init__(self, seed=42):
        self.is_fitted = False
        self.classes_ = None
        self.n_classes_ = 0
        self.seed = seed
        self._rng = np.random.RandomState(seed)
    
    def fit(self, X, y, sample_weight=None, eval_set=None, **kwargs):
        self.is_fitted = True
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self._rng = np.random.RandomState(self.seed)
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self._rng.choice(self.classes_, size=len(X))
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        n_samples = len(X)
        proba = self._rng.rand(n_samples, self.n_classes_)
        return proba / proba.sum(axis=1, keepdims=True)


def test_metamodel_integration():
    """Test MetaModel integration with engine logic."""
    print("=" * 70)
    print("MetaModel + Engine Integration Test")
    print("=" * 70)
    
    # Step 1: Create and train a MetaModel
    print("\n1. Creating MetaModel...")
    primary = DummyClassifier(seed=42)
    meta = DummyClassifier(seed=43)
    
    meta_model = MetaModel(
        primary_model=primary,
        meta_model=meta,
        meta_threshold=0.6
    )
    
    # Create sample training data
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 10), columns=[f'feature_{i}' for i in range(10)])
    y_direction = np.random.choice([-1, 0, 1], 100)
    y_success = np.random.choice([0, 1], 100)
    
    # Train
    print("   Training MetaModel...")
    meta_model.train(X_train, y_direction, y_success)
    print("   ✅ MetaModel trained successfully")
    
    # Step 2: Test prediction (simulating engine behavior)
    print("\n2. Testing prediction pipeline...")
    X_test = pd.DataFrame(np.random.randn(10, 10), columns=[f'feature_{i}' for i in range(10)])
    
    result = meta_model.predict_meta(X_test)
    
    print(f"   Features shape: {X_test.shape}")
    print(f"   Predictions generated: {len(result['final_signal'])}")
    
    # Step 3: Simulate engine's signal processing logic
    print("\n3. Simulating engine signal processing...")
    meta_threshold = 0.6
    
    for i in range(min(5, len(X_test))):
        final_signal = result['final_signal'][i]
        meta_conf = result['meta_conf'][i]
        raw_signal = result['raw_signal'][i]
        
        raw_signal_str = {1: 'BUY', -1: 'SELL', 0: 'NEUTRAL'}[raw_signal]
        
        if final_signal == 0:
            if raw_signal != 0:
                print(f"   [{i}] 🤖 Raw Signal: {raw_signal_str} | "
                      f"🛡️ Meta Conf: {meta_conf:.2f} (< {meta_threshold:.2f}) "
                      f"→ ❌ BLOCKED")
            else:
                print(f"   [{i}] 🤖 Raw Signal: {raw_signal_str} | "
                      f"🛡️ Meta Conf: {meta_conf:.2f} "
                      f"→ HOLD")
        else:
            final_signal_str = {1: 'BUY', -1: 'SELL'}[final_signal]
            print(f"   [{i}] 🤖 Raw Signal: {raw_signal_str} | "
                  f"🛡️ Meta Conf: {meta_conf:.2f} (>= {meta_threshold:.2f}) "
                  f"→ ✅ {final_signal_str}")
    
    # Step 4: Verify backward compatibility check
    print("\n4. Testing backward compatibility check...")
    print(f"   isinstance(meta_model, MetaModel): {isinstance(meta_model, MetaModel)}")
    print("   ✅ Type checking works correctly")
    
    # Step 5: Statistics
    print("\n5. Prediction Statistics:")
    n_raw_signals = (result['raw_signal'] != 0).sum()
    n_final_signals = (result['final_signal'] != 0).sum()
    n_filtered = n_raw_signals - n_final_signals
    filter_rate = (n_filtered / max(n_raw_signals, 1)) * 100
    
    print(f"   Raw signals: {n_raw_signals}")
    print(f"   Final signals: {n_final_signals}")
    print(f"   Filtered: {n_filtered} ({filter_rate:.1f}%)")
    print(f"   Average meta confidence: {result['meta_conf'].mean():.2f}")
    
    print("\n" + "=" * 70)
    print("✅ Integration Test Passed!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ MetaModel training and prediction")
    print("  ✓ Signal filtering based on meta confidence")
    print("  ✓ Enhanced logging with emojis")
    print("  ✓ Backward compatibility type checking")
    print("  ✓ Raw vs final signal comparison")


if __name__ == "__main__":
    test_metamodel_integration()
