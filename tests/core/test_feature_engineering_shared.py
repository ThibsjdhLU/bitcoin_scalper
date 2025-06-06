import sys
from unittest.mock import patch
import pytest
import pandas as pd
import numpy as np
from bitcoin_scalper.core.feature_engineering import add_features

class DummyConfig:
    def __init__(self, *a, **kw): pass
    def get(self, key, default=None):
        return {
            "MT5_REST_URL": "http://localhost:8080",
            "MT5_REST_API_KEY": "fakekey",
            "TSDB_HOST": "localhost",
            "TSDB_PORT": "5432",
            "TSDB_NAME": "btcdb",
            "TSDB_USER": "btcuser",
            "TSDB_PASSWORD": "btcpass",
            "TSDB_SSLMODE": "disable",
            "ML_MODEL_PATH": "model_rf.pkl"
        }.get(key, default)

patcher = patch("bitcoin_scalper.core.config.SecureConfig", DummyConfig)
patcher.start()

with patch("bitcoin_scalper.core.config.SecureConfig", DummyConfig):
    def make_df():
        # Génère un DataFrame minute synthétique sur 40 minutes
        idx = pd.date_range('2024-06-01 00:00', periods=40, freq='min', tz='UTC')
        data = {
            '<OPEN>': np.linspace(70000, 70400, 40),
            '<HIGH>': np.linspace(70010, 70410, 40),
            '<LOW>': np.linspace(69990, 70390, 40),
            '<CLOSE>': np.linspace(70005, 70405, 40),
            '<TICKVOL>': np.random.randint(90, 130, 40).astype(np.float32),
        }
        return pd.DataFrame(data, index=idx)

    def test_add_features_shape_and_presence():
        df = make_df()
        df_feat = add_features(df)
        # Colonnes attendues
        expected = [
            'HL2', 'HLC3', 'OC2',
            'log_return_1m', 'log_return_3m', 'log_return_5m', 'log_return_15m',
            'log_return_1m_std_5m', 'log_return_1m_std_15m', 'log_return_1m_std_30m',
            'log_return_3m_std_5m', 'log_return_3m_std_15m', 'log_return_3m_std_30m',
            'log_return_5m_std_5m', 'log_return_5m_std_15m', 'log_return_5m_std_30m',
            'log_return_15m_std_5m', 'log_return_15m_std_15m', 'log_return_15m_std_30m',
            'SMA_5', 'EMA_5', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20',
            'RSI_14', 'MACD', 'MACD_signal', 'ATR_14',
            'BB_MA_20', 'BB_UPPER_20', 'BB_LOWER_20', 'BB_WIDTH_20',
            'Z_SCORE_30', 'SLOPE_15',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
        ]
        for col in expected:
            assert col in df_feat.columns, f"Colonne manquante : {col}"
        assert df_feat.shape[0] == 40

    def test_add_features_causality():
        df = make_df()
        df_feat = add_features(df)
        # Vérifie qu'aucune valeur future n'est utilisée (ex : log_return_1m doit être NaN en t=0)
        assert np.isnan(df_feat['log_return_1m'].iloc[0])
        assert np.isnan(df_feat['log_return_3m'].iloc[0:3]).all()
        assert np.isnan(df_feat['log_return_5m'].iloc[0:5]).all()
        assert np.isnan(df_feat['log_return_15m'].iloc[0:15]).all()
        # Slope doit être NaN sur les 14 premiers points
        assert np.isnan(df_feat['SLOPE_15'].iloc[0:14]).all()

    def test_add_features_nan_tracking(caplog):
        df = make_df()
        with caplog.at_level('WARNING'):
            df_feat = add_features(df)
            # Doit logger un warning pour les colonnes avec NaN
            assert any('Colonnes avec NaN générés' in r for r in caplog.text.split('\n'))

    def test_add_features_no_future_leak():
        df = make_df()
        df_feat = add_features(df)
        # Pour chaque rolling, la valeur en t ne doit dépendre que de t et du passé
        # On modifie la dernière valeur et vérifie que seule la dernière ligne change
        df2 = df.copy()
        df2.iloc[-1, df2.columns.get_loc('<CLOSE>')] += 1000
        df_feat2 = add_features(df2)
        # Les valeurs sauf la dernière ligne doivent être identiques
        pd.testing.assert_frame_equal(df_feat.iloc[:-1], df_feat2.iloc[:-1]) 