"""
Unit tests for gradient boosting models (XGBoost and CatBoost).
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

# These tests will be skipped if XGBoost is not installed
try:
    from src.bitcoin_scalper.models.gradient_boosting import (
        XGBoostClassifier,
        XGBoostRegressor,
        _HAS_XGBOOST
    )
    skip_xgboost = not _HAS_XGBOOST
except ImportError:
    skip_xgboost = True


@pytest.mark.skipif(skip_xgboost, reason="XGBoost not installed")
class TestXGBoostClassifier:
    """Test suite for XGBoost classifier."""
    
    @pytest.fixture
    def classification_data(self):
        """Create sample classification data."""
        np.random.seed(42)
        n_samples = 200
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'feature4': np.random.randn(n_samples),
            'feature5': np.random.randn(n_samples)
        })
        
        # Create labels with some pattern
        y = pd.Series(
            ((X['feature1'] + X['feature2'] > 0).astype(int) - 
             (X['feature3'] + X['feature4'] < 0).astype(int))
        )
        
        return X, y
    
    @pytest.fixture
    def model(self):
        """Create XGBoost classifier."""
        return XGBoostClassifier(
            n_estimators=20,  # Small for fast testing
            max_depth=3,
            learning_rate=0.1,
            use_gpu=False,  # Disable GPU for testing
            random_state=42
        )
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.params['n_estimators'] == 20
        assert model.params['max_depth'] == 3
        assert model.params['learning_rate'] == 0.1
        assert model.is_fitted is False
    
    def test_train_basic(self, model, classification_data):
        """Test basic training."""
        X, y = classification_data
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]
        
        model.train(X_train, y_train, eval_set=(X_val, y_val))
        
        assert model.is_fitted is True
        assert model.feature_names == list(X.columns)
        assert model.n_features == 5
        assert len(model.classes_) == 3  # -1, 0, 1
    
    def test_predict(self, model, classification_data):
        """Test prediction."""
        X, y = classification_data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(pred in model.classes_ for pred in predictions)
    
    def test_predict_proba(self, model, classification_data):
        """Test probability prediction."""
        X, y = classification_data
        X_train, X_test = X[:150], X[150:]
        y_train = y[:150]
        
        model.train(X_train, y_train)
        proba = model.predict_proba(X_test)
        
        assert proba.shape == (len(X_test), len(model.classes_))
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all() and (proba <= 1).all()
    
    def test_sample_weights(self, model, classification_data):
        """Test training with sample weights."""
        X, y = classification_data
        X_train = X[:150]
        y_train = y[:150]
        
        # Create sample weights (e.g., from Triple Barrier)
        weights = np.random.rand(len(X_train))
        weights = weights / weights.sum()  # Normalize
        
        model.train(X_train, y_train, sample_weights=weights)
        
        assert model.is_fitted is True
    
    def test_early_stopping(self, model, classification_data):
        """Test early stopping."""
        X, y = classification_data
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]
        
        model.train(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=5
        )
        
        assert model.is_fitted is True
        # Model should stop before max iterations due to early stopping
        # (hard to test without seeing actual iterations)
    
    def test_feature_importance(self, model, classification_data):
        """Test feature importance extraction."""
        X, y = classification_data
        model.train(X, y)
        
        importance = model.get_feature_importance()
        
        assert isinstance(importance, pd.Series)
        assert len(importance) == 5
        assert importance.index.tolist() == list(X.columns)
        # Feature importance should be non-negative
        assert (importance >= 0).all()
    
    def test_save_load(self, model, classification_data):
        """Test model persistence."""
        X, y = classification_data
        model.train(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            
            # Save
            model.save(path)
            assert path.exists()
            assert path.with_suffix('.metadata.json').exists()
            
            # Load
            new_model = XGBoostClassifier()
            new_model.load(path)
            
            assert new_model.is_fitted is True
            assert new_model.feature_names == model.feature_names
            assert new_model.n_features == model.n_features
            assert np.array_equal(new_model.classes_, model.classes_)
            
            # Predictions should match
            predictions_original = model.predict(X)
            predictions_loaded = new_model.predict(X)
            assert np.array_equal(predictions_original, predictions_loaded)
    
    def test_binary_classification(self):
        """Test with binary classification (2 classes)."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.choice([0, 1], size=100))
        
        model = XGBoostClassifier(n_estimators=10, use_gpu=False, random_state=42)
        model.train(X, y)
        
        predictions = model.predict(X)
        proba = model.predict_proba(X)
        
        assert len(model.classes_) == 2
        assert proba.shape == (len(X), 2)
    
    def test_multiclass_classification(self):
        """Test with multi-class classification (>3 classes)."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.choice([0, 1, 2, 3, 4], size=100))
        
        model = XGBoostClassifier(n_estimators=10, use_gpu=False, random_state=42)
        model.train(X, y)
        
        predictions = model.predict(X)
        proba = model.predict_proba(X)
        
        assert len(model.classes_) == 5
        assert proba.shape == (len(X), 5)


@pytest.mark.skipif(skip_xgboost, reason="XGBoost not installed")
class TestXGBoostRegressor:
    """Test suite for XGBoost regressor."""
    
    @pytest.fixture
    def regression_data(self):
        """Create sample regression data."""
        np.random.seed(42)
        n_samples = 200
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        
        # Create continuous target
        y = pd.Series(
            X['feature1'] * 2 + X['feature2'] * -1 + 
            X['feature3'] * 0.5 + np.random.randn(n_samples) * 0.1
        )
        
        return X, y
    
    @pytest.fixture
    def model(self):
        """Create XGBoost regressor."""
        return XGBoostRegressor(
            n_estimators=20,
            max_depth=3,
            learning_rate=0.1,
            use_gpu=False,
            random_state=42
        )
    
    def test_train_basic(self, model, regression_data):
        """Test basic training."""
        X, y = regression_data
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]
        
        model.train(X_train, y_train, eval_set=(X_val, y_val))
        
        assert model.is_fitted is True
        assert model.n_features == 3
    
    def test_predict(self, model, regression_data):
        """Test prediction."""
        X, y = regression_data
        X_train, X_test = X[:150], X[150:]
        y_train = y[:150]
        
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        # Predictions should be continuous values
        assert predictions.dtype == np.float32 or predictions.dtype == np.float64
    
    def test_predict_proba_not_available(self, model, regression_data):
        """Test that predict_proba is not available for regression."""
        X, y = regression_data
        model.train(X, y)
        
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)
    
    def test_feature_importance(self, model, regression_data):
        """Test feature importance for regression."""
        X, y = regression_data
        model.train(X, y)
        
        importance = model.get_feature_importance()
        
        assert isinstance(importance, pd.Series)
        assert len(importance) == 3
        # Feature 1 should have highest importance (coefficient = 2)
        assert importance.idxmax() == 'feature1'


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.skipif(skip_xgboost, reason="XGBoost not installed")
    def test_predict_before_training(self):
        """Test prediction before training raises error."""
        model = XGBoostClassifier(use_gpu=False)
        X = pd.DataFrame(np.random.randn(10, 5))
        
        with pytest.raises(ValueError, match="must be trained"):
            model.predict(X)
    
    @pytest.mark.skipif(skip_xgboost, reason="XGBoost not installed")
    def test_save_unfitted_model(self):
        """Test that saving unfitted model raises error."""
        model = XGBoostClassifier(use_gpu=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            
            with pytest.raises(ValueError, match="Cannot save unfitted model"):
                model.save(path)
    
    @pytest.mark.skipif(skip_xgboost, reason="XGBoost not installed")
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        model = XGBoostClassifier(use_gpu=False)
        
        with pytest.raises(FileNotFoundError):
            model.load("/nonexistent/path/model.json")
    
    @pytest.mark.skipif(skip_xgboost, reason="XGBoost not installed")
    def test_imbalanced_data(self):
        """Test with highly imbalanced data."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5))
        # 95% class 0, 5% class 1
        y = pd.Series([0] * 95 + [1] * 5)
        
        model = XGBoostClassifier(n_estimators=10, use_gpu=False)
        model.train(X, y)
        
        # Should handle imbalanced data gracefully
        predictions = model.predict(X)
        assert len(predictions) == len(X)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
