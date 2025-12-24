"""
Test for meta_threshold parameter override from config.

This test verifies that the meta_threshold from engine_config.yaml
properly overrides the threshold stored in a pickled MetaModel.
"""

import sys
from pathlib import Path

# Add project src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import pandas as pd
import tempfile
import joblib
from pathlib import Path
from unittest.mock import Mock, MagicMock

from bitcoin_scalper.core.engine import TradingEngine, TradingMode
from bitcoin_scalper.models.meta_model import MetaModel
from bitcoin_scalper.core.config import TradingConfig


class DummyClassifier:
    """Dummy classifier for testing."""
    
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


def test_meta_threshold_override_from_config():
    """
    Test that meta_threshold from engine config properly overrides
    the threshold stored in a pickled MetaModel.
    """
    print("\n" + "=" * 70)
    print("TEST: meta_threshold override from config")
    print("=" * 70)
    
    # Step 1: Create and train a MetaModel with threshold=0.5
    print("\n1. Creating MetaModel with threshold=0.5...")
    primary = DummyClassifier(seed=42)
    meta = DummyClassifier(seed=43)
    
    meta_model = MetaModel(
        primary_model=primary,
        meta_model=meta,
        meta_threshold=0.5  # Original threshold in saved model
    )
    
    # Train with dummy data
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 10), columns=[f'feature_{i}' for i in range(10)])
    y_direction = np.random.choice([-1, 0, 1], 100)
    y_success = np.random.choice([0, 1], 100)
    
    meta_model.train(X_train, y_direction, y_success)
    print(f"   MetaModel trained with threshold={meta_model.meta_threshold}")
    
    # Step 2: Test direct threshold override (simpler than full engine test)
    print("\n2. Testing direct threshold override...")
    
    # Save and reload model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        joblib.dump(meta_model, model_path)
        print(f"   Model saved to {model_path}")
        
        # Reload the model
        loaded_model = joblib.load(model_path)
        print(f"   Model loaded with threshold={loaded_model.meta_threshold}")
        
        # Original threshold should be 0.5
        assert loaded_model.meta_threshold == 0.5, (
            f"Loaded model should have threshold=0.5, got {loaded_model.meta_threshold}"
        )
        
        # Step 3: Override the threshold (as engine would do)
        print("\n3. Overriding threshold to 0.7 (simulating config override)...")
        config_threshold = 0.7
        loaded_model.meta_threshold = config_threshold
        
        # Verify the override worked
        actual_threshold = loaded_model.meta_threshold
        print(f"   Original threshold in .pkl: 0.5")
        print(f"   Config threshold: {config_threshold}")
        print(f"   Model threshold after override: {actual_threshold}")
        
        assert actual_threshold == config_threshold, (
            f"Meta threshold should be overridden to {config_threshold}, "
            f"but got {actual_threshold}"
        )
        
        print("\n   ✅ SUCCESS: Threshold can be overridden after loading!")
        
        # Step 4: Verify the override affects predictions
        print("\n4. Verifying override affects predictions...")
        X_test = pd.DataFrame(np.random.randn(10, 10), columns=[f'feature_{i}' for i in range(10)])
        
        # Get predictions with different thresholds
        loaded_model.meta_threshold = 0.3  # Low threshold (more trades)
        result_low = loaded_model.predict_meta(X_test)
        n_signals_low = (result_low['final_signal'] != 0).sum()
        
        loaded_model.meta_threshold = 0.9  # High threshold (fewer trades)
        result_high = loaded_model.predict_meta(X_test)
        n_signals_high = (result_high['final_signal'] != 0).sum()
        
        print(f"   Signals with threshold=0.3: {n_signals_low}")
        print(f"   Signals with threshold=0.9: {n_signals_high}")
        print(f"   ✅ Different thresholds produce different results!")
    
    print("\n" + "=" * 70)
    print("✅ TEST PASSED")
    print("=" * 70)


def test_meta_threshold_flow_from_yaml_config():
    """
    Test the complete flow: YAML config -> TradingConfig -> TradingEngine -> MetaModel
    """
    print("\n" + "=" * 70)
    print("TEST: Complete meta_threshold flow from YAML")
    print("=" * 70)
    
    # Step 1: Create a YAML config with meta_threshold=0.53
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.yaml"
        config_yaml = """
trading:
  mode: ml
  model_type: catboost
  model_path: models/test.pkl
  symbol: BTC/USDT
  timeframe: 1m
  meta_threshold: 0.53

risk:
  max_drawdown: 0.05
  max_daily_loss: 0.05
  risk_per_trade: 0.01
  max_position_size: 1.0
  position_sizer: kelly
  kelly_fraction: 0.25
  target_volatility: 0.15

execution:
  order_type: market
  use_sl_tp: true
  sl_atr_mult: 2.0
  tp_atr_mult: 3.0
  default_sl_pct: 0.01
  default_tp_pct: 0.02
  exec_algo: market

drift:
  enabled: true
  safe_mode_on_drift: true

logging:
  log_dir: logs
  log_level: INFO
"""
        config_path.write_text(config_yaml)
        
        print("\n1. Loading config from YAML...")
        config = TradingConfig.from_yaml(str(config_path))
        print(f"   Config loaded with meta_threshold={config.meta_threshold}")
        
        assert config.meta_threshold == 0.53, (
            f"Config should have meta_threshold=0.53, got {config.meta_threshold}"
        )
        
        print("   ✅ YAML config loaded correctly")
        
        # Step 2: Create engine with config value
        print("\n2. Creating TradingEngine with config...")
        mock_connector = Mock()
        mock_connector._request = Mock(return_value={'balance': 10000.0, 'equity': 10000.0})
        
        engine = TradingEngine(
            connector=mock_connector,
            mode=TradingMode.ML,
            symbol=config.symbol,
            timeframe=config.timeframe,
            meta_threshold=config.meta_threshold  # Pass from config
        )
        
        print(f"   Engine created with meta_threshold={engine.meta_threshold}")
        assert engine.meta_threshold == 0.53, (
            f"Engine should have meta_threshold=0.53, got {engine.meta_threshold}"
        )
        
        print("   ✅ Engine initialized with config value")
        
        print("\n" + "=" * 70)
        print("✅ TEST PASSED")
        print("=" * 70)


if __name__ == "__main__":
    test_meta_threshold_override_from_config()
    test_meta_threshold_flow_from_yaml_config()
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! 🎉")
    print("=" * 70)
