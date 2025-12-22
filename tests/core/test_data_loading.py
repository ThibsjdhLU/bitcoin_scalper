"""
Tests for data_loading module.
Tests both Legacy MT5 format and Binance/Standard format support.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from bitcoin_scalper.core.data_loading import load_minute_csv


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def legacy_mt5_csv(temp_dir):
    """Create a sample Legacy MT5 format CSV."""
    csv_path = temp_dir / "legacy_mt5.csv"
    
    # Create sample data in MT5 format
    data = [
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>",
        "2023.01.01\t00:00:00\t16512.74\t16514.97\t16511.86\t16514.97\t38\t0\t3305",
        "2023.01.01\t00:01:00\t16514.97\t16514.97\t16511.47\t16511.47\t26\t0\t3305",
        "2023.01.01\t00:02:00\t16511.47\t16511.97\t16509.66\t16510.97\t44\t0\t3305",
        "2023.01.01\t00:03:00\t16510.97\t16510.98\t16506.66\t16506.98\t22\t0\t3304",
        "2023.01.01\t00:04:00\t16506.98\t16507.00\t16504.00\t16505.00\t30\t0\t3304",
    ]
    
    csv_path.write_text("\n".join(data))
    return csv_path


@pytest.fixture
def binance_csv(temp_dir):
    """Create a sample Binance/Standard format CSV."""
    csv_path = temp_dir / "binance.csv"
    
    # Create sample data in Binance format (lowercase columns)
    data = [
        "date,open,high,low,close,volume",
        "2023-01-01 00:00:00+00:00,16512.74,16514.97,16511.86,16514.97,38.5",
        "2023-01-01 00:01:00+00:00,16514.97,16514.97,16511.47,16511.47,26.3",
        "2023-01-01 00:02:00+00:00,16511.47,16511.97,16509.66,16510.97,44.2",
        "2023-01-01 00:03:00+00:00,16510.97,16510.98,16506.66,16506.98,22.1",
        "2023-01-01 00:04:00+00:00,16506.98,16507.00,16504.00,16505.00,30.0",
    ]
    
    csv_path.write_text("\n".join(data))
    return csv_path


@pytest.fixture
def binance_csv_with_timestamp(temp_dir):
    """Create a Binance CSV with 'timestamp' column instead of 'date'."""
    csv_path = temp_dir / "binance_timestamp.csv"
    
    # Create sample data with 'timestamp' column
    data = [
        "timestamp,open,high,low,close,volume",
        "2023-01-01 00:00:00+00:00,16512.74,16514.97,16511.86,16514.97,38.5",
        "2023-01-01 00:01:00+00:00,16514.97,16514.97,16511.47,16511.47,26.3",
        "2023-01-01 00:02:00+00:00,16511.47,16511.97,16509.66,16510.97,44.2",
        "2023-01-01 00:03:00+00:00,16510.97,16510.98,16506.66,16506.98,22.1",
        "2023-01-01 00:04:00+00:00,16506.98,16507.00,16504.00,16505.00,30.0",
    ]
    
    csv_path.write_text("\n".join(data))
    return csv_path


class TestLegacyMT5Format:
    """Tests for Legacy MT5 format CSV loading."""
    
    def test_load_legacy_mt5_csv(self, legacy_mt5_csv):
        """Test loading a Legacy MT5 format CSV."""
        df = load_minute_csv(str(legacy_mt5_csv))
        
        # Check shape
        assert len(df) == 5
        
        # Check columns - should have <TAGS> format
        expected_cols = ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']
        assert list(df.columns) == expected_cols
        
        # Check index is DatetimeIndex
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Check data types
        assert df['<OPEN>'].dtype == np.float32
        assert df['<HIGH>'].dtype == np.float32
        assert df['<LOW>'].dtype == np.float32
        assert df['<CLOSE>'].dtype == np.float32
        assert df['<TICKVOL>'].dtype == np.float32
        
        # Check first value
        assert df.iloc[0]['<CLOSE>'] == pytest.approx(16514.97, rel=1e-5)
        
        # Check that <VOL> and <SPREAD> were removed
        assert '<VOL>' not in df.columns
        assert '<SPREAD>' not in df.columns
    
    def test_legacy_mt5_sorted_by_time(self, legacy_mt5_csv):
        """Test that the DataFrame is sorted by time."""
        df = load_minute_csv(str(legacy_mt5_csv))
        
        # Check that index is sorted
        assert df.index.is_monotonic_increasing
        
        # Check time increments
        assert df.index[1] - df.index[0] == pd.Timedelta(minutes=1)


class TestBinanceFormat:
    """Tests for Binance/Standard format CSV loading."""
    
    def test_load_binance_csv(self, binance_csv):
        """Test loading a Binance format CSV."""
        df = load_minute_csv(str(binance_csv))
        
        # Check shape
        assert len(df) == 5
        
        # Check columns - should be renamed to <TAGS> format
        expected_cols = ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']
        assert list(df.columns) == expected_cols
        
        # Check index is DatetimeIndex
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Check data types
        assert df['<OPEN>'].dtype == np.float32
        assert df['<HIGH>'].dtype == np.float32
        assert df['<LOW>'].dtype == np.float32
        assert df['<CLOSE>'].dtype == np.float32
        assert df['<TICKVOL>'].dtype == np.float32
        
        # Check first value
        assert df.iloc[0]['<CLOSE>'] == pytest.approx(16514.97, rel=1e-5)
    
    def test_load_binance_with_timestamp_column(self, binance_csv_with_timestamp):
        """Test loading Binance CSV with 'timestamp' column instead of 'date'."""
        df = load_minute_csv(str(binance_csv_with_timestamp))
        
        # Check shape
        assert len(df) == 5
        
        # Check columns
        expected_cols = ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']
        assert list(df.columns) == expected_cols
        
        # Check index is DatetimeIndex
        assert isinstance(df.index, pd.DatetimeIndex)
    
    def test_binance_sorted_by_time(self, binance_csv):
        """Test that the DataFrame is sorted by time."""
        df = load_minute_csv(str(binance_csv))
        
        # Check that index is sorted
        assert df.index.is_monotonic_increasing
        
        # Check time increments
        assert df.index[1] - df.index[0] == pd.Timedelta(minutes=1)
    
    def test_binance_volume_renamed_to_tickvol(self, binance_csv):
        """Test that 'volume' is renamed to '<TICKVOL>'."""
        df = load_minute_csv(str(binance_csv))
        
        # Check that <TICKVOL> exists
        assert '<TICKVOL>' in df.columns
        
        # Check that 'volume' doesn't exist
        assert 'volume' not in df.columns
        
        # Check value
        assert df.iloc[0]['<TICKVOL>'] == pytest.approx(38.5, rel=1e-5)


class TestFormatDetection:
    """Tests for automatic format detection."""
    
    def test_detects_legacy_format(self, legacy_mt5_csv, caplog):
        """Test that Legacy MT5 format is detected."""
        load_minute_csv(str(legacy_mt5_csv))
        assert "Legacy MT5" in caplog.text
    
    def test_detects_binance_format(self, binance_csv, caplog):
        """Test that Binance format is detected."""
        load_minute_csv(str(binance_csv))
        assert "Binance/Standard" in caplog.text
    
    def test_detects_binance_with_timestamp(self, binance_csv_with_timestamp, caplog):
        """Test that Binance format with 'timestamp' column is detected."""
        load_minute_csv(str(binance_csv_with_timestamp))
        assert "Binance/Standard" in caplog.text
    
    def test_unknown_format_raises_error(self, temp_dir):
        """Test that unknown format raises an error."""
        csv_path = temp_dir / "unknown.csv"
        csv_path.write_text("foo,bar,baz\n1,2,3\n4,5,6\n")
        
        with pytest.raises(ValueError, match="Format CSV non reconnu"):
            load_minute_csv(str(csv_path))


class TestOutputConsistency:
    """Tests that both formats produce identical output structure."""
    
    def test_both_formats_produce_same_columns(self, legacy_mt5_csv, binance_csv):
        """Test that both formats produce the same column names."""
        df_legacy = load_minute_csv(str(legacy_mt5_csv))
        df_binance = load_minute_csv(str(binance_csv))
        
        # Both should have the same columns
        assert list(df_legacy.columns) == list(df_binance.columns)
    
    def test_both_formats_produce_datetime_index(self, legacy_mt5_csv, binance_csv):
        """Test that both formats produce a DatetimeIndex."""
        df_legacy = load_minute_csv(str(legacy_mt5_csv))
        df_binance = load_minute_csv(str(binance_csv))
        
        # Both should have DatetimeIndex
        assert isinstance(df_legacy.index, pd.DatetimeIndex)
        assert isinstance(df_binance.index, pd.DatetimeIndex)
    
    def test_both_formats_produce_same_dtypes(self, legacy_mt5_csv, binance_csv):
        """Test that both formats produce the same data types."""
        df_legacy = load_minute_csv(str(legacy_mt5_csv))
        df_binance = load_minute_csv(str(binance_csv))
        
        # Both should have the same dtypes
        assert all(df_legacy.dtypes == df_binance.dtypes)


class TestDataValidation:
    """Tests for data validation."""
    
    def test_removes_duplicates(self, temp_dir):
        """Test that duplicate timestamps are removed."""
        csv_path = temp_dir / "duplicates.csv"
        data = [
            "date,open,high,low,close,volume",
            "2023-01-01 00:00:00+00:00,16512.74,16514.97,16511.86,16514.97,38.5",
            "2023-01-01 00:00:00+00:00,16512.74,16514.97,16511.86,16514.97,38.5",  # duplicate
            "2023-01-01 00:01:00+00:00,16514.97,16514.97,16511.47,16511.47,26.3",
        ]
        csv_path.write_text("\n".join(data))
        
        df = load_minute_csv(str(csv_path))
        
        # Should have 2 rows, not 3
        assert len(df) == 2
    
    def test_removes_nan_rows(self, temp_dir):
        """Test that rows with NaN are removed (when fill_method=None)."""
        csv_path = temp_dir / "with_nan.csv"
        data = [
            "date,open,high,low,close,volume",
            "2023-01-01 00:00:00+00:00,16512.74,16514.97,16511.86,16514.97,38.5",
            "2023-01-01 00:01:00+00:00,,16514.97,16511.47,16511.47,26.3",  # NaN in open
            "2023-01-01 00:02:00+00:00,16511.47,16511.97,16509.66,16510.97,44.2",
        ]
        csv_path.write_text("\n".join(data))
        
        df = load_minute_csv(str(csv_path))
        
        # Should have 2 rows, not 3
        assert len(df) == 2


class TestFillMethods:
    """Tests for fill methods."""
    
    def test_ffill_method(self, temp_dir):
        """Test forward fill method."""
        csv_path = temp_dir / "with_gaps.csv"
        data = [
            "date,open,high,low,close,volume",
            "2023-01-01 00:00:00+00:00,100,105,95,100,10",
            "2023-01-01 00:01:00+00:00,,,,110,11",  # missing OHLC
            "2023-01-01 00:02:00+00:00,120,125,115,120,12",
        ]
        csv_path.write_text("\n".join(data))
        
        df = load_minute_csv(str(csv_path), fill_method='ffill')
        
        # Should have 3 rows
        assert len(df) == 3
        
        # Second row should have forward-filled values
        assert df.iloc[1]['<OPEN>'] == pytest.approx(100, rel=1e-5)
        assert df.iloc[1]['<CLOSE>'] == pytest.approx(110, rel=1e-5)  # close has value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
