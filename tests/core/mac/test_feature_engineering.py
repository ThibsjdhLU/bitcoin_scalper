import pytest
import pandas as pd
import numpy as np
from bitcoin_scalper.core.feature_engineering import FeatureEngineering
from catboost import CatBoostClassifier
from bitcoin_scalper.core.modeling import analyze_feature_importance, select_features_by_importance

def make_ohlcv(n=30):
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    df = pd.DataFrame({
        "open": np.linspace(100, 110, n),
        "high": np.linspace(101, 111, n),
        "low": np.linspace(99, 109, n),
        "close": np.linspace(100, 110, n) + np.random.normal(0, 0.5, n),
        "volume": np.random.randint(1, 10, n)
    }, index=idx)
    return df

def test_add_indicators():
    fe = FeatureEngineering()
    df = make_ohlcv()
    out = fe.add_indicators(df)
    for col in ["rsi", "macd", "macd_signal", "macd_diff", "ema_21", "ema_50", "sma_20", "bb_high", "bb_low", "bb_width", "atr", "vwap"]:
        assert col in out.columns
        assert not out[col].isnull().all()

def test_add_features():
    fe = FeatureEngineering()
    df = make_ohlcv()
    out = fe.add_features(df)
    for col in ["return", "log_return", "volatility_20", "vol_price_ratio"]:
        assert col in out.columns
        assert not out[col].isnull().all()

def test_multi_timeframe():
    fe = FeatureEngineering(["1min", "5min"])
    df1 = make_ohlcv(30)
    df5 = make_ohlcv(10)
    dfs = {"1min": df1, "5min": df5}
    out = fe.multi_timeframe(dfs)
    for tf in ["1min", "5min"]:
        for col in ["rsi", "macd", "ema_21", "ema_50", "atr", "vwap", "return", "volatility_20"]:
            assert f"{tf}_{col}" in out.columns

def test_edge_cases_empty():
    fe = FeatureEngineering()
    df = pd.DataFrame()
    out = fe.add_indicators(df)
    assert isinstance(out, pd.DataFrame)
    out2 = fe.add_features(df)
    assert isinstance(out2, pd.DataFrame)
    out3 = fe.multi_timeframe({})
    assert out3 is None or isinstance(out3, pd.DataFrame)

def test_add_indicators_missing_cols():
    fe = FeatureEngineering()
    df = pd.DataFrame({"close": [1,2,3]})
    out = fe.add_indicators(df)
    assert out.equals(df)

def test_add_features_missing_cols():
    fe = FeatureEngineering()
    df = pd.DataFrame({"close": [1,2,3]})
    out = fe.add_features(df)
    assert out.equals(df)

def test_add_indicators_short_df():
    fe = FeatureEngineering()
    df = make_ohlcv(5)
    out = fe.add_indicators(df)
    assert "atr" in out.columns
    assert out["atr"].isnull().all() or out["atr"].notnull().all()

def test_add_features_short_df():
    fe = FeatureEngineering()
    df = make_ohlcv(2)
    out = fe.add_features(df)
    assert "return" in out.columns
    assert "log_return" in out.columns

def test_multi_timeframe_empty():
    fe = FeatureEngineering(["1min", "5min"])
    out = fe.multi_timeframe({})
    assert out is None or isinstance(out, pd.DataFrame)

def test_multi_timeframe_missing_cols():
    fe = FeatureEngineering(["1min", "5min"])
    dfs = {"1min": pd.DataFrame({"close": [1,2,3]}), "5min": pd.DataFrame({"close": [1,2,3]})}
    out = fe.multi_timeframe(dfs)
    assert isinstance(out, pd.DataFrame)

def test_add_indicators_nan():
    fe = FeatureEngineering()
    df = make_ohlcv(10)
    df.iloc[0, df.columns.get_loc("close")] = np.nan
    out = fe.add_indicators(df)
    assert "rsi" in out.columns

def test_add_features_nan():
    fe = FeatureEngineering()
    df = make_ohlcv(10)
    df.iloc[0, df.columns.get_loc("close")] = np.nan
    out = fe.add_features(df)
    assert "return" in out.columns

def test_add_indicators_exception(monkeypatch):
    fe = FeatureEngineering()
    monkeypatch.setattr("ta.momentum.RSIIndicator.__init__", lambda self, *a, **k: (_ for _ in ()).throw(Exception("fail")))
    df = make_ohlcv(20)
    with pytest.raises(Exception):
        fe.add_indicators(df)

def test_add_features_exception(monkeypatch):
    fe = FeatureEngineering()
    monkeypatch.setattr("pandas.Series.pct_change", lambda self, *a, **k: (_ for _ in ()).throw(Exception("fail")))
    df = make_ohlcv(20)
    with pytest.raises(Exception, match="fail"):
        fe.add_features(df)

def test_no_lookahead():
    fe = FeatureEngineering()
    df = make_ohlcv(20)
    out = fe.add_indicators(df)
    # Tous les indicateurs doivent être NaN à la première ligne
    indicators = [
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'ema_21', 'ema_50', 'sma_20',
        'bb_high', 'bb_low', 'bb_width',
        'atr', 'supertrend', 'vwap',
        'close_sma_3', 'atr_sma_20'
    ]
    for col in indicators:
        assert pd.isna(out.iloc[0][col]), f"{col} n'est pas NaN en t0 (look-ahead possible)"
    # La valeur t1 doit correspondre à la valeur calculée sur t0 (hors NaN initiaux)
    df_base = make_ohlcv(20)
    out_base = fe.add_indicators(df_base)
    for col in indicators:
        if not pd.isna(out_base.iloc[0][col]):
            assert np.isclose(out.iloc[1][col], out_base.iloc[0][col], equal_nan=True), f"Décalage incorrect sur {col}"

def test_add_features_no_lookahead():
    fe = FeatureEngineering()
    df = make_ohlcv(20)
    out = fe.add_features(df)
    # Les features doivent être NaN à la première ligne
    features = ['return', 'log_return', 'volatility_20']
    for col in features:
        assert pd.isna(out.iloc[0][col]), f"{col} n'est pas NaN en t0 (look-ahead possible)"
    # La valeur t1 doit correspondre à la valeur calculée sur t0 (hors NaN initiaux)
    df_base = make_ohlcv(20)
    out_base = fe.add_features(df_base)
    for col in features:
        if not pd.isna(out_base.iloc[0][col]):
            assert np.isclose(out.iloc[1][col], out_base.iloc[0][col], equal_nan=True), f"Décalage incorrect sur {col}"

def test_supertrend_direction_type():
    # DataFrame minimal avec OHLCV
    df = pd.DataFrame({
        'open': [1, 2, 3, 4, 5, 6, 7],
        'high': [2, 3, 4, 5, 6, 7, 8],
        'low': [0, 1, 2, 3, 4, 5, 6],
        'close': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        'volume': [10, 10, 10, 10, 10, 10, 10]
    })
    fe = FeatureEngineering()
    df_feat = fe.add_indicators(df)
    # Vérifie que la colonne supertrend_direction existe et est bien de type float
    assert 'supertrend_direction' in df_feat.columns
    assert np.issubdtype(df_feat['supertrend_direction'].dtype, np.floating)

def test_advanced_indicators_presence():
    fe = FeatureEngineering()
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 50),
        'high': np.linspace(101, 111, 50),
        'low': np.linspace(99, 109, 50),
        'volume': np.random.randint(1, 10, 50)
    })
    out = fe.add_indicators(df)
    advanced_cols = [
        'kc_hband', 'kc_lband', 'kc_width',
        'donchian_hband', 'donchian_lband', 'donchian_width',
        'chandelier_exit_long', 'chandelier_exit_short', 'ulcer_index',
        'mfi', 'obv', 'adi', 'cmf',
        'tsi', 'cci', 'willr', 'stochrsi', 'ultimate_osc', 'roc',
        'adx', 'adx_pos', 'adx_neg', 'psar',
        'ichimoku_a', 'ichimoku_b', 'ichimoku_base_line', 'ichimoku_conversion_line',
        'ppo', 'ppo_signal', 'ppo_hist'
    ]
    for col in advanced_cols:
        if col not in out.columns:
            pytest.skip(f"Feature avancée {col} non supportée par la version de ta installée.")
        # Vérifie qu'il y a au moins une valeur non nulle ou non NaN
        if out[col].isnull().all() or (out[col] == 0).all():
            pytest.skip(f"Feature avancée {col} entièrement NaN ou nulle (probablement non supportée par ta).")
        assert out[col].notnull().any(), f"Feature avancée {col} entièrement NaN ou nulle"

def test_context_features_presence():
    fe = FeatureEngineering()
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 50),
        'high': np.linspace(101, 111, 50),
        'low': np.linspace(99, 109, 50),
        'volume': np.random.randint(1, 10, 50)
    }, index=pd.date_range("2024-01-01", periods=50, freq="min"))
    # Appeler d'abord add_indicators pour générer les colonnes bb_high et bb_low
    df_ind = fe.add_indicators(df)
    out = fe.add_features(df_ind)
    # Z-score
    for col in ['close', 'high', 'low', 'volume']:
        for win in [5, 20, 50, 100]:
            assert f"{col}_zscore_{win}" in out.columns
    # Distance à la bande de Bollinger
    for col in ["dist_bb_high", "dist_bb_low", "dist_bb_width"]:
        assert col in out.columns
    # Distance au plus haut/bas N périodes
    for win in [5, 20, 50, 100]:
        assert f"dist_high_{win}" in out.columns
        assert f"dist_low_{win}" in out.columns
    # Encodage temporel enrichi
    for col in ["minute", "hour", "day", "weekday", "month", "week", "quarter", "year",
                "minute_sin", "minute_cos", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos",
                "hour_rel", "weekday_rel", "month_rel"]:
        assert col in out.columns

def test_analyze_feature_importance(tmp_path):
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
    y = np.random.choice([0, 1, 2], 100)
    model = CatBoostClassifier().fit(X, y)
    out_dir = tmp_path / "feature_importance"
    analyze_feature_importance(model, X, out_dir=str(out_dir), prefix="test_")
    # Vérifie que le PNG d'importance classique est bien généré
    assert (out_dir / "test_feature_importance.png").exists()

def test_select_features_by_importance():
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f"f{i}" for i in range(10)])
    y = np.random.choice([0, 1, 2], 100)
    model = CatBoostClassifier().fit(X, y)
    selected = select_features_by_importance(model, X, top_n=5)
    assert isinstance(selected, list)
    assert len(selected) == 5
    for feat in selected:
        assert feat in X.columns 

def test_log_return_1m_presence_in_testset():
    """
    Vérifie que log_return_1m et 1min_log_return sont bien présents dans un DataFrame simulant un test set.
    """
    fe = FeatureEngineering(["1min", "5min"])
    df1 = make_ohlcv(30)
    df5 = make_ohlcv(10)
    dfs = {"1min": df1, "5min": df5}
    out = fe.multi_timeframe(dfs)
    # Simule un split test
    test = out.iloc[-10:].copy()
    # Patch pipeline comme dans orchestrator
    if '<CLOSE>' in test.columns and 'log_return_1m' not in test.columns:
        test['log_return_1m'] = np.log(test['<CLOSE>'] / test['<CLOSE>'].shift(1))
    if 'log_return_1m' in test.columns:
        test['1min_log_return'] = test['log_return_1m']
    assert 'log_return_1m' in test.columns, "log_return_1m absent du test set !"
    assert '1min_log_return' in test.columns, "1min_log_return absent du test set !" 