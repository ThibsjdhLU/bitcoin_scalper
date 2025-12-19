"""
Unit tests for base model interface and validation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.bitcoin_scalper.models.base import BaseModel


class DummyModel(BaseModel):
    """Dummy model for testing base class functionality."""
    
    def __init__(self):
        super().__init__()
        self.train_called = False
        self.predict_called = False
    
    def train(self, X, y, sample_weights=None, eval_set=None, **kwargs):
        self.validate_inputs(X, y)
        self.feature_names = self._extract_feature_names(X)
        self.n_features = len(self.feature_names)
        self.classes_ = np.unique(y)
        self.is_fitted = True
        self.train_called = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        self.validate_inputs(X)
        self.predict_called = True
        # Return dummy predictions
        n_samples = X.shape[0] if isinstance(X, np.ndarray) else len(X)
        return np.random.choice(self.classes_, size=n_samples)
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        self.validate_inputs(X)
        n_samples = X.shape[0] if isinstance(X, np.ndarray) else len(X)
        n_classes = len(self.classes_)
        # Return dummy probabilities
        proba = np.random.rand(n_samples, n_classes)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba
    
    def save(self, path):
        pass
    
    def load(self, path):
        return self


class TestBaseModel:
    """Test suite for BaseModel functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        y = pd.Series(np.random.choice([-1, 0, 1], size=100))
        return X, y
    
    @pytest.fixture
    def model(self):
        """Create a dummy model instance."""
        return DummyModel()
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.model is None
        assert model.is_fitted is False
        assert model.feature_names is None
        assert model.n_features is None
        assert model.classes_ is None
    
    def test_train_basic(self, model, sample_data):
        """Test basic training."""
        X, y = sample_data
        model.train(X, y)
        
        assert model.is_fitted is True
        assert model.train_called is True
        assert model.feature_names == ['feature1', 'feature2', 'feature3']
        assert model.n_features == 3
        assert len(model.classes_) == 3
    
    def test_predict_before_training(self, model, sample_data):
        """Test that prediction fails before training."""
        X, _ = sample_data
        
        with pytest.raises(ValueError, match="must be trained"):
            model.predict(X)
    
    def test_predict_after_training(self, model, sample_data):
        """Test prediction after training."""
        X, y = sample_data
        model.train(X, y)
        
        predictions = model.predict(X)
        
        assert model.predict_called is True
        assert len(predictions) == len(X)
        assert all(pred in model.classes_ for pred in predictions)
    
    def test_predict_proba(self, model, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        model.train(X, y)
        
        proba = model.predict_proba(X)
        
        assert proba.shape == (len(X), len(model.classes_))
        # Check probabilities sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)
        # Check all probabilities are between 0 and 1
        assert (proba >= 0).all() and (proba <= 1).all()
    
    def test_validate_empty_data(self, model):
        """Test validation with empty data."""
        X = np.array([]).reshape(0, 3)
        y = np.array([])
        
        with pytest.raises(ValueError, match="empty"):
            model.validate_inputs(X, y)
    
    def test_validate_nan_values(self, model):
        """Test validation with NaN values."""
        X = np.array([[1, 2, np.nan], [4, 5, 6]])
        
        with pytest.raises(ValueError, match="NaN"):
            model.validate_inputs(X)
    
    def test_validate_inf_values(self, model):
        """Test validation with infinite values."""
        X = np.array([[1, 2, np.inf], [4, 5, 6]])
        
        with pytest.raises(ValueError, match="infinite"):
            model.validate_inputs(X)
    
    def test_validate_shape_mismatch(self, model, sample_data):
        """Test validation with feature count mismatch."""
        X, y = sample_data
        model.train(X, y)
        
        # Try to predict with wrong number of features
        X_wrong = np.random.randn(10, 5)  # 5 features instead of 3
        
        with pytest.raises(ValueError, match="Feature count mismatch"):
            model.validate_inputs(X_wrong)
    
    def test_validate_label_mismatch(self, model):
        """Test validation with X/y length mismatch."""
        X = np.random.randn(100, 3)
        y = np.random.randn(50)  # Wrong length
        
        with pytest.raises(ValueError, match="Feature and label count mismatch"):
            model.validate_inputs(X, y)
    
    def test_feature_name_extraction_dataframe(self, model):
        """Test feature name extraction from DataFrame."""
        X = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        feature_names = model._extract_feature_names(X)
        assert feature_names == ['col1', 'col2']
    
    def test_feature_name_extraction_array(self, model):
        """Test feature name extraction from numpy array."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        
        feature_names = model._extract_feature_names(X)
        assert feature_names == ['feature_0', 'feature_1', 'feature_2']
    
    def test_repr(self, model, sample_data):
        """Test string representation."""
        # Before training
        repr_str = repr(model)
        assert "not fitted" in repr_str
        
        # After training
        X, y = sample_data
        model.train(X, y)
        repr_str = repr(model)
        assert "fitted" in repr_str
        assert "3 features" in repr_str
    
    def test_train_with_sample_weights(self, model, sample_data):
        """Test training with sample weights."""
        X, y = sample_data
        weights = np.random.rand(len(X))
        
        model.train(X, y, sample_weights=weights)
        assert model.is_fitted is True
    
    def test_train_with_eval_set(self, model, sample_data):
        """Test training with evaluation set."""
        X, y = sample_data
        X_val = X.iloc[:20]
        y_val = y.iloc[:20]
        
        model.train(X, y, eval_set=(X_val, y_val))
        assert model.is_fitted is True
    
    def test_numpy_input(self, model):
        """Test with numpy arrays instead of DataFrames."""
        X = np.random.randn(50, 5)
        y = np.random.choice([-1, 0, 1], size=50)
        
        model.train(X, y)
        predictions = model.predict(X)
        
        assert model.is_fitted is True
        assert len(predictions) == len(X)
    
    def test_get_feature_importance_default(self, model, sample_data):
        """Test default feature importance (returns None)."""
        X, y = sample_data
        model.train(X, y)
        
        importance = model.get_feature_importance()
        assert importance is None


class TestInputValidation:
    """Test input validation edge cases."""
    
    @pytest.fixture
    def model(self):
        return DummyModel()
    
    def test_mixed_data_types(self, model):
        """Test with mixed pandas and numpy types."""
        X_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        y_series = pd.Series([0, 1, 0])
        
        model.train(X_df, y_series)
        
        # Predict with numpy
        X_np = np.array([[1, 2], [3, 4]])
        predictions = model.predict(X_np)
        assert len(predictions) == 2
    
    def test_single_sample(self, model):
        """Test with single sample."""
        X = pd.DataFrame({'a': [1], 'b': [2]})
        y = pd.Series([0])
        
        model.train(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == 1
    
    def test_large_dataset(self, model):
        """Test with large dataset."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(10000, 50))
        y = pd.Series(np.random.choice([-1, 0, 1], size=10000))
        
        model.train(X, y)
        assert model.n_features == 50
        assert model.is_fitted is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
