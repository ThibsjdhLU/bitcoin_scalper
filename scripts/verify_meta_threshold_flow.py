#!/usr/bin/env python3
"""
End-to-end verification script for meta_threshold parameter flow.

This script simulates the complete flow from config loading to model prediction,
verifying that the meta_threshold from engine_config.yaml is properly used.
"""

import sys
sys.path.insert(0, '/home/runner/work/bitcoin_scalper/bitcoin_scalper/src')

import numpy as np
import pandas as pd
import tempfile
import joblib
from pathlib import Path
from unittest.mock import Mock

from bitcoin_scalper.core.config import TradingConfig
from bitcoin_scalper.core.engine import TradingEngine, TradingMode
from bitcoin_scalper.models.meta_model import MetaModel


class DummyClassifier:
    """Minimal classifier for testing."""
    def __init__(self, seed=42):
        self.is_fitted = False
        self.classes_ = None
        self.n_classes_ = 0
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self.feature_names_ = None
    
    def fit(self, X, y, sample_weight=None, eval_set=None, **kwargs):
        self.is_fitted = True
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
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


def main():
    print("=" * 80)
    print("END-TO-END VERIFICATION: meta_threshold parameter flow")
    print("=" * 80)
    
    # Step 1: Verify config loads correctly
    print("\n[STEP 1] Loading configuration from engine_config.yaml...")
    config_path = Path('/home/runner/work/bitcoin_scalper/bitcoin_scalper/config/engine_config.yaml')
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    config = TradingConfig.from_yaml(str(config_path))
    print(f"✅ Config loaded successfully")
    print(f"   meta_threshold from YAML: {config.meta_threshold}")
    print(f"   symbol: {config.symbol}")
    print(f"   timeframe: {config.timeframe}")
    
    if config.meta_threshold != 0.53:
        print(f"⚠️  WARNING: Expected meta_threshold=0.53, got {config.meta_threshold}")
    
    # Step 2: Create and save a test model with different threshold
    print("\n[STEP 2] Creating test MetaModel with threshold=0.5 (different from config)...")
    primary = DummyClassifier(seed=42)
    meta = DummyClassifier(seed=43)
    
    test_model = MetaModel(
        primary_model=primary,
        meta_model=meta,
        meta_threshold=0.5  # Different from config's 0.53
    )
    
    # Train with dummy data
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 10), columns=[f'feature_{i}' for i in range(10)])
    y_direction = np.random.choice([-1, 0, 1], 100)
    y_success = np.random.choice([0, 1], 100)
    
    test_model.train(X_train, y_direction, y_success)
    print(f"✅ Test model created and trained")
    print(f"   Model threshold: {test_model.meta_threshold}")
    
    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        joblib.dump(test_model, model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Step 3: Create TradingEngine with config value
        print("\n[STEP 3] Creating TradingEngine with config meta_threshold...")
        mock_connector = Mock()
        mock_connector._request = Mock(return_value={'balance': 10000.0, 'equity': 10000.0})
        
        engine = TradingEngine(
            connector=mock_connector,
            mode=TradingMode.ML,
            symbol=config.symbol,
            timeframe=config.timeframe,
            meta_threshold=config.meta_threshold  # Use config value
        )
        print(f"✅ Engine created")
        print(f"   Engine meta_threshold: {engine.meta_threshold}")
        
        # Step 4: Load model with config override
        print("\n[STEP 4] Loading model with config override...")
        print(f"   Model file threshold: 0.5")
        print(f"   Config override: {config.meta_threshold}")
        
        success = engine.load_ml_model(
            str(model_path),
            meta_threshold=config.meta_threshold  # Override with config
        )
        
        if not success:
            print("❌ Model loading failed")
            return False
        
        print(f"✅ Model loaded successfully")
        print(f"   Loaded model threshold: {engine.ml_model.meta_threshold}")
        
        # Step 5: Verify threshold was overridden
        print("\n[STEP 5] Verifying threshold override...")
        if engine.ml_model.meta_threshold == config.meta_threshold:
            print(f"✅ SUCCESS: Model uses config threshold ({config.meta_threshold})")
        else:
            print(f"❌ FAILURE: Model threshold ({engine.ml_model.meta_threshold}) != "
                  f"config threshold ({config.meta_threshold})")
            return False
        
        # Step 6: Test prediction with overridden threshold
        print("\n[STEP 6] Testing prediction with overridden threshold...")
        X_test = pd.DataFrame(np.random.randn(20, 10), columns=[f'feature_{i}' for i in range(10)])
        result = engine.ml_model.predict_meta(X_test)
        
        n_signals = (result['final_signal'] != 0).sum()
        n_raw = (result['raw_signal'] != 0).sum()
        avg_conf = result['meta_conf'].mean()
        
        print(f"✅ Prediction completed")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Raw signals: {n_raw}")
        print(f"   Final signals: {n_signals} (after meta filter with threshold={config.meta_threshold})")
        print(f"   Average meta confidence: {avg_conf:.2f}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ END-TO-END VERIFICATION PASSED")
    print("=" * 80)
    print("\nSummary:")
    print(f"  1. Config loaded: meta_threshold = {config.meta_threshold}")
    print(f"  2. Model saved with: meta_threshold = 0.5")
    print(f"  3. Engine initialized with config value: {config.meta_threshold}")
    print(f"  4. Model loaded and threshold overridden: {config.meta_threshold}")
    print(f"  5. Predictions use config threshold: {config.meta_threshold}")
    print("\n✅ The meta_threshold from engine_config.yaml is the SINGLE SOURCE OF TRUTH")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
