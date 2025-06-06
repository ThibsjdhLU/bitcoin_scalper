import pytest
import pandas as pd
import numpy as np
from bitcoin_scalper.core.feature_engineering import FeatureEngineering

def make_df(n=20):
    idx = pd.date_range("2024-06-01 00:00", periods=n, freq="min", tz="UTC")
    data = {
        "close": np.linspace(70000, 70000+n, n),
        "high": np.linspace(70010, 70010+n, n),
        "low": np.linspace(69990, 69990+n, n),
        "volume": np.random.randint(90, 130, n).astype(np.float32),
    }
    return pd.DataFrame(data, index=idx)

def test_multi_timeframe_basic():
    fe = FeatureEngineering()
    dfs = {"1min": make_df(20), "5min": make_df(4)}
    df_feat = fe.multi_timeframe(dfs, price_col="close", high_col="high", low_col="low", volume_col="volume")
    assert isinstance(df_feat, pd.DataFrame)
    assert any("1min_close" in c for c in df_feat.columns)
    assert any("5min_close" in c for c in df_feat.columns)

def test_add_indicators_shape():
    fe = FeatureEngineering()
    df = make_df(30)
    df_ind = fe.add_indicators(df)
    assert df_ind.shape[0] == 30
    assert "rsi" in df_ind.columns
    assert "macd" in df_ind.columns

def test_multi_timeframe_missing_col():
    fe = FeatureEngineering()
    dfs = {"1min": make_df(10)}
    # Supprime une colonne critique
    dfs["1min"] = dfs["1min"].drop(columns=["high"])
    df_feat = fe.multi_timeframe(dfs, price_col="close", high_col="high", low_col="low", volume_col="volume")
    # Vérifie que la colonne manquante n'est pas dans le résultat
    assert not any("1min_high" in c for c in df_feat.columns)

def test_add_indicators_nan():
    fe = FeatureEngineering()
    df = make_df(10)
    df.iloc[0, 0] = np.nan
    df_ind = fe.add_indicators(df)
    assert df_ind.isnull().any().any() 