import pytest
import pandas as pd
import numpy as np
from scripts.prepare_features import generate_signal, check_temporal_integrity
from bitcoin_scalper.core.feature_engineering import FeatureEngineering

def make_df():
    # Génère un DataFrame OHLCV minimal mais suffisant
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 30),
        'high': np.linspace(101, 111, 30),
        'low': np.linspace(99, 109, 30),
        'tickvol': np.random.randint(1, 10, 30)
    })
    fe = FeatureEngineering()
    df = fe.add_indicators(df, price_col='close', high_col='high', low_col='low', volume_col='tickvol')
    df['close_sma_3'] = df['close'].shift(1).rolling(window=3, min_periods=1).mean()
    df['atr_sma_20'] = df['atr'].shift(1).rolling(window=20, min_periods=1).mean()
    return df

def test_generate_signal_balance():
    df = make_df()
    df = generate_signal(df)
    # Vérifie qu'il y a au moins deux classes dans le signal
    assert df['signal'].nunique() >= 2

def test_temporal_integrity():
    df = make_df()
    df = generate_signal(df)
    assert check_temporal_integrity(df)

def test_nan_handling():
    df = make_df()
    df.iloc[0, df.columns.get_loc('ema_21')] = np.nan
    df = generate_signal(df)
    # On droppe les NaN comme dans le pipeline
    df = df.dropna(subset=['ema_21', 'ema_50', 'rsi', 'atr', 'supertrend', 'close_sma_3', 'atr_sma_20', 'signal'])
    assert not df.isnull().any().any()

def test_incomplete_df():
    # DataFrame sans 'high' ou 'low'
    df = pd.DataFrame({'close': [1, 2, 3], 'tickvol': [1, 1, 1]})
    fe = FeatureEngineering()
    out = fe.add_indicators(df, price_col='close', high_col='close', low_col='close', volume_col='tickvol')
    assert isinstance(out, pd.DataFrame) 