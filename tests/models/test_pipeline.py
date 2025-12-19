"""
Unit tests for training pipeline and meta-labeling.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from src.bitcoin_scalper.models.base import BaseModel
from src.bitcoin_scalper.models.pipeline import Trainer, MetaLabelingPipeline
from tests.models.test_base import DummyModel


class TestTrainer:
    """Test suite for Trainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        y = pd.Series(np.random.choice([-1, 0, 1], size=100))
        weights = np.random.rand(100)
        return X, y, weights
    
    @pytest.fixture
    def model(self):
        """Create dummy model."""
        return DummyModel()
    
    def test_initialization(self, model):
        """Test trainer initialization."""
        trainer = Trainer(model)
        
        assert trainer.model is model
        assert trainer.handle_nans == 'error'
        assert trainer.handle_infs == 'error'
    
    def test_train_basic(self, model, sample_data):
        """Test basic training."""
        X, y, _ = sample_data
        trainer = Trainer(model)
        
        result = trainer.train(X, y)
        
        assert result is model
        assert model.is_fitted is True
    
    def test_train_with_weights(self, model, sample_data):
        """Test training with sample weights."""
        X, y, weights = sample_data
        trainer = Trainer(model)
        
        trainer.train(X, y, sample_weights=weights)
        
        assert model.is_fitted is True
    
    def test_train_with_eval_set(self, model, sample_data):
        """Test training with evaluation set."""
        X, y, _ = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        
        trainer = Trainer(model)
        trainer.train(X_train, y_train, eval_set=(X_val, y_val))
        
        assert model.is_fitted is True
    
    def test_handle_nans_error(self, model):
        """Test that NaN values raise error by default."""
        X = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, 6, 7, 8]
        })
        y = pd.Series([0, 1, 0, 1])
        
        trainer = Trainer(model, handle_nans='error')
        
        with pytest.raises(ValueError, match="NaN"):
            trainer.train(X, y)
    
    def test_handle_nans_fill(self, model):
        """Test NaN filling."""
        X = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, 6, 7, 8]
        })
        y = pd.Series([0, 1, 0, 1])
        
        trainer = Trainer(model, handle_nans='fill', fill_value=0.0)
        trainer.train(X, y)
        
        assert model.is_fitted is True
    
    def test_handle_nans_drop(self, model):
        """Test NaN dropping."""
        X = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, 6, 7, 8]
        })
        y = pd.Series([0, 1, 0, 1])
        
        trainer = Trainer(model, handle_nans='drop')
        trainer.train(X, y)
        
        # Should train on 3 samples (1 dropped)
        assert model.is_fitted is True
    
    def test_handle_infs_error(self, model):
        """Test that infinite values raise error by default."""
        X = pd.DataFrame({
            'a': [1, 2, np.inf, 4],
            'b': [5, 6, 7, 8]
        })
        y = pd.Series([0, 1, 0, 1])
        
        trainer = Trainer(model, handle_infs='error')
        
        with pytest.raises(ValueError, match="infinite"):
            trainer.train(X, y)
    
    def test_handle_infs_clip(self, model):
        """Test infinite value clipping."""
        X = pd.DataFrame({
            'a': [1, 2, np.inf, 4],
            'b': [5, 6, 7, 8]
        })
        y = pd.Series([0, 1, 0, 1])
        
        trainer = Trainer(model, handle_infs='clip', clip_value=1000.0)
        trainer.train(X, y)
        
        assert model.is_fitted is True
    
    def test_handle_infs_drop(self, model):
        """Test infinite value dropping."""
        X = pd.DataFrame({
            'a': [1, 2, np.inf, 4],
            'b': [5, 6, 7, 8]
        })
        y = pd.Series([0, 1, 0, 1])
        
        trainer = Trainer(model, handle_infs='drop')
        trainer.train(X, y)
        
        # Should train on 3 samples (1 dropped)
        assert model.is_fitted is True
    
    def test_evaluate(self, model, sample_data):
        """Test model evaluation."""
        X, y, _ = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        trainer = Trainer(model)
        trainer.train(X_train, y_train)
        
        metrics = trainer.evaluate(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'n_samples' in metrics
        assert metrics['n_samples'] == len(X_test)


class TestMetaLabelingPipeline:
    """Test suite for MetaLabelingPipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for meta-labeling."""
        np.random.seed(42)
        n_samples = 150
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        
        # Side labels: -1 (Short), 0 (Neutral), 1 (Long)
        y_side = pd.Series(np.random.choice([-1, 0, 1], size=n_samples))
        
        # Meta labels: 0 (Pass), 1 (Bet)
        # Simulating that trades are successful ~60% of the time
        y_meta = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]))
        
        weights = np.random.rand(n_samples)
        
        return X, y_side, y_meta, weights
    
    @pytest.fixture
    def models(self):
        """Create primary and secondary models."""
        primary = DummyModel()
        secondary = DummyModel()
        return primary, secondary
    
    @pytest.fixture
    def pipeline(self, models):
        """Create meta-labeling pipeline."""
        primary, secondary = models
        return MetaLabelingPipeline(primary, secondary)
    
    def test_initialization(self, pipeline, models):
        """Test pipeline initialization."""
        primary, secondary = models
        
        assert pipeline.primary_model is primary
        assert pipeline.secondary_model is secondary
    
    def test_train_basic(self, pipeline, sample_data):
        """Test basic training."""
        X, y_side, y_meta, weights = sample_data
        
        X_train = X[:100]
        y_side_train = y_side[:100]
        y_meta_train = y_meta[:100]
        weights_train = weights[:100]
        
        pipeline.train(
            X_train, y_side_train, y_meta_train,
            sample_weights=weights_train
        )
        
        assert pipeline.primary_model.is_fitted is True
        assert pipeline.secondary_model.is_fitted is True
    
    def test_train_with_eval_set(self, pipeline, sample_data):
        """Test training with evaluation set."""
        X, y_side, y_meta, weights = sample_data
        
        X_train, X_val = X[:100], X[100:]
        y_side_train, y_side_val = y_side[:100], y_side[100:]
        y_meta_train, y_meta_val = y_meta[:100], y_meta[100:]
        weights_train = weights[:100]
        
        pipeline.train(
            X_train, y_side_train, y_meta_train,
            sample_weights=weights_train,
            eval_set=(X_val, y_side_val, y_meta_val)
        )
        
        assert pipeline.primary_model.is_fitted is True
        assert pipeline.secondary_model.is_fitted is True
    
    def test_predict(self, pipeline, sample_data):
        """Test prediction."""
        X, y_side, y_meta, _ = sample_data
        X_train, X_test = X[:100], X[100:]
        y_side_train = y_side[:100]
        y_meta_train = y_meta[:100]
        
        pipeline.train(X_train, y_side_train, y_meta_train)
        
        side, bet = pipeline.predict(X_test)
        
        assert len(side) == len(X_test)
        assert len(bet) == len(X_test)
        # Side should be -1, 0, or 1
        assert all(s in [-1, 0, 1] for s in side)
        # Bet should be 0 or 1
        assert all(b in [0, 1] for b in bet)
    
    def test_predict_combined(self, pipeline, sample_data):
        """Test combined prediction."""
        X, y_side, y_meta, _ = sample_data
        X_train, X_test = X[:100], X[100:]
        y_side_train = y_side[:100]
        y_meta_train = y_meta[:100]
        
        pipeline.train(X_train, y_side_train, y_meta_train)
        
        combined = pipeline.predict_combined(X_test)
        
        assert len(combined) == len(X_test)
        # Combined should be -1, 0, or 1
        assert all(c in [-1, 0, 1] for c in combined)
    
    def test_secondary_features(self, pipeline, sample_data):
        """Test that secondary model receives probabilities as features."""
        X, y_side, y_meta, _ = sample_data
        X_train = X[:100]
        y_side_train = y_side[:100]
        y_meta_train = y_meta[:100]
        
        pipeline.train(X_train, y_side_train, y_meta_train)
        
        # Secondary model should have more features than primary
        # (original features + primary probabilities)
        assert pipeline.secondary_model.n_features > pipeline.primary_model.n_features
    
    def test_save_load(self, pipeline, sample_data):
        """Test pipeline persistence."""
        X, y_side, y_meta, _ = sample_data
        pipeline.train(X, y_side, y_meta)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            primary_path = Path(tmpdir) / "primary.pkl"
            secondary_path = Path(tmpdir) / "secondary.pkl"
            
            # Save
            pipeline.save(primary_path, secondary_path)
            
            # Load into new pipeline
            new_primary = DummyModel()
            new_secondary = DummyModel()
            new_pipeline = MetaLabelingPipeline(new_primary, new_secondary)
            new_pipeline.load(primary_path, secondary_path)
            
            # Both models should be fitted
            assert new_pipeline.primary_model.is_fitted is True
            assert new_pipeline.secondary_model.is_fitted is True


class TestIntegration:
    """Integration tests combining Trainer and MetaLabeling."""
    
    def test_trainer_with_pipeline(self):
        """Test using Trainer with MetaLabelingPipeline."""
        np.random.seed(42)
        
        # Create data
        X = pd.DataFrame(np.random.randn(100, 5))
        y_side = pd.Series(np.random.choice([-1, 0, 1], size=100))
        y_meta = pd.Series(np.random.choice([0, 1], size=100))
        
        # Create pipeline
        primary = DummyModel()
        secondary = DummyModel()
        pipeline = MetaLabelingPipeline(primary, secondary)
        
        # Train
        pipeline.train(X, y_side, y_meta)
        
        # Predict
        side, bet = pipeline.predict(X)
        
        assert len(side) == len(X)
        assert len(bet) == len(X)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
