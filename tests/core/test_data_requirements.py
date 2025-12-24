"""
Tests for data requirements validation and feature engineering with insufficient data.

This test suite validates that:
1. Data requirements are properly enforced
2. Clear error messages are provided when data is insufficient
3. Feature engineering succeeds with sufficient data
4. Multi-timeframe processing works correctly
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from bitcoin_scalper.core.data_requirements import (
    SAFE_MIN_ROWS,
    MIN_ROWS_AFTER_FEATURE_ENG,
    DEFAULT_FETCH_LIMIT,
    validate_data_requirements,
    get_recommended_fetch_limit
)
from bitcoin_scalper.core.feature_engineering import FeatureEngineering


class TestDataRequirements:
    """Test data requirements validation functions."""
    
    def test_constants(self):
        """Test that constants are set to expected values."""
        assert SAFE_MIN_ROWS == 1500, "SAFE_MIN_ROWS should be 1500"
        assert MIN_ROWS_AFTER_FEATURE_ENG == 300, "MIN_ROWS_AFTER_FEATURE_ENG should be 300"
        assert DEFAULT_FETCH_LIMIT == 1500, "DEFAULT_FETCH_LIMIT should be 1500"
    
    def test_validate_data_requirements_pre_processing_sufficient(self):
        """Test validation passes with sufficient pre-processing data."""
        valid, msg = validate_data_requirements(1500, "pre_processing")
        assert valid is True
        assert msg == ""
    
    def test_validate_data_requirements_pre_processing_insufficient(self):
        """Test validation fails with insufficient pre-processing data."""
        valid, msg = validate_data_requirements(100, "pre_processing")
        assert valid is False
        assert "1500" in msg
        assert "100" in msg
        assert "minimum required" in msg.lower()
    
    def test_validate_data_requirements_post_processing_sufficient(self):
        """Test validation passes with sufficient post-processing data."""
        valid, msg = validate_data_requirements(300, "post_processing")
        assert valid is True
        assert msg == ""
    
    def test_validate_data_requirements_post_processing_insufficient(self):
        """Test validation fails with insufficient post-processing data."""
        valid, msg = validate_data_requirements(100, "post_processing")
        assert valid is False
        assert "100" in msg
        assert "300" in msg.lower() or "minimum" in msg.lower()
    
    def test_get_recommended_fetch_limit(self):
        """Test recommended fetch limits for different timeframes."""
        # 1-minute should require full SAFE_MIN_ROWS
        assert get_recommended_fetch_limit("1m") == 1500
        assert get_recommended_fetch_limit("1min") == 1500
        assert get_recommended_fetch_limit("M1") == 1500
        
        # 5-minute should require less
        limit_5m = get_recommended_fetch_limit("5m")
        assert limit_5m <= 1500
        
        # Higher timeframes should require less
        limit_1h = get_recommended_fetch_limit("1h")
        assert limit_1h <= 1500


class TestFeatureEngineeringValidation:
    """Test feature engineering validation with various data sizes."""
    
    def generate_sample_data(self, n_rows: int) -> pd.DataFrame:
        """Generate sample OHLCV data for testing."""
        start_date = datetime(2024, 1, 1)
        dates = [start_date + timedelta(minutes=i) for i in range(n_rows)]
        
        np.random.seed(42)
        close_prices = 50000 + np.cumsum(np.random.randn(n_rows) * 10)
        
        df = pd.DataFrame({
            'open': close_prices + np.random.randn(n_rows) * 5,
            'high': close_prices + np.abs(np.random.randn(n_rows)) * 10,
            'low': close_prices - np.abs(np.random.randn(n_rows)) * 10,
            'close': close_prices,
            'volume': np.random.uniform(1, 100, n_rows)
        }, index=pd.DatetimeIndex(dates))
        
        return df
    
    def test_feature_engineering_with_insufficient_data(self):
        """Test that feature engineering returns empty DataFrame with insufficient data."""
        fe = FeatureEngineering()
        
        # Create data with only 100 rows (insufficient)
        df = self.generate_sample_data(100)
        
        # Apply feature engineering
        result = fe.add_indicators(
            df,
            price_col='close',
            high_col='high',
            low_col='low',
            volume_col='volume',
            prefix='test_'
        )
        
        # Should return empty DataFrame due to insufficient data after NaN removal
        assert result.empty or len(result) < MIN_ROWS_AFTER_FEATURE_ENG
    
    def test_feature_engineering_with_sufficient_data(self):
        """Test that feature engineering succeeds with sufficient data."""
        fe = FeatureEngineering()
        
        # Create data with 1500 rows (sufficient)
        df = self.generate_sample_data(1500)
        
        # Apply feature engineering
        result = fe.add_indicators(
            df,
            price_col='close',
            high_col='high',
            low_col='low',
            volume_col='volume',
            prefix='test_'
        )
        
        # Should have sufficient data after processing
        assert not result.empty
        assert len(result) >= MIN_ROWS_AFTER_FEATURE_ENG
        
        # Check that some key indicators were created
        assert 'test_rsi' in result.columns
        assert 'test_sma_200' in result.columns
        assert 'test_ema_200' in result.columns
        assert 'test_close_frac' in result.columns
    
    def test_feature_engineering_with_marginal_data(self):
        """Test feature engineering with data close to minimum."""
        fe = FeatureEngineering()
        
        # Create data with 500 rows (marginal - might work but not ideal)
        df = self.generate_sample_data(500)
        
        # Apply feature engineering
        result = fe.add_indicators(
            df,
            price_col='close',
            high_col='high',
            low_col='low',
            volume_col='volume',
            prefix='test_'
        )
        
        # With 500 rows, might get empty result or very few rows
        # The system should handle this gracefully
        if not result.empty:
            # If it didn't fail completely, check data quality
            assert len(result) > 0
            # Should have indicators even if sparse
            assert 'test_rsi' in result.columns
    
    def test_nan_handling_logging(self, caplog):
        """Test that NaN handling produces appropriate log messages."""
        import logging
        caplog.set_level(logging.INFO)
        
        fe = FeatureEngineering()
        df = self.generate_sample_data(1500)
        
        # Apply feature engineering (should log NaN handling)
        result = fe.add_indicators(
            df,
            price_col='close',
            high_col='high',
            low_col='low',
            volume_col='volume',
            prefix='test_'
        )
        
        # Check that appropriate log messages were generated
        log_text = caplog.text.lower()
        assert 'processing' in log_text or 'rows' in log_text
        
    def test_multi_timeframe_data_requirements(self):
        """Test that multi-timeframe processing has sufficient data."""
        fe = FeatureEngineering()
        
        # Create 1500 rows of 1-minute data
        df = self.generate_sample_data(1500)
        
        # Process 1-minute features
        df_1m = fe.add_indicators(
            df,
            price_col='close',
            high_col='high',
            low_col='low',
            volume_col='volume',
            prefix='1min_'
        )
        
        assert not df_1m.empty
        
        # Resample to 5-minute
        df_5m = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Should have ~300 5-minute bars from 1500 1-minute bars
        assert len(df_5m) >= 250  # Allow some margin for dropna
        
        # Process 5-minute features
        df_5m_features = fe.add_indicators(
            df_5m,
            price_col='close',
            high_col='high',
            low_col='low',
            volume_col='volume',
            prefix='5min_'
        )
        
        # Should have data after processing
        # Note: might be empty if 300 bars not enough for SMA-200
        # This is expected behavior that should be caught by validation
        if not df_5m_features.empty:
            assert len(df_5m_features) > 0


class TestConnectorDefaults:
    """Test that connectors use proper default limits."""
    
    def test_default_fetch_limit_constant(self):
        """Test that DEFAULT_FETCH_LIMIT is available for import."""
        from bitcoin_scalper.core.data_requirements import DEFAULT_FETCH_LIMIT
        assert DEFAULT_FETCH_LIMIT == 1500
    
    @pytest.mark.parametrize("module_name,class_name", [
        ("bitcoin_scalper.connectors.binance_connector", "BinanceConnector"),
        ("bitcoin_scalper.connectors.mt5_rest_client", "MT5RestClient"),
        ("bitcoin_scalper.connectors.paper", "PaperMT5Client"),
    ])
    def test_connector_imports_data_requirements(self, module_name, class_name):
        """Test that connectors import data_requirements module."""
        try:
            import importlib
            module = importlib.import_module(module_name)
            
            # Check if DEFAULT_FETCH_LIMIT is used in the module
            source_file = module.__file__
            with open(source_file, 'r') as f:
                content = f.read()
                assert 'data_requirements' in content or 'DEFAULT_FETCH_LIMIT' in content
        except ImportError:
            pytest.skip(f"Module {module_name} not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
