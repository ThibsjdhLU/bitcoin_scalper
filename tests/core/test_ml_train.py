import pandas as pd
import numpy as np
import os
import tempfile
import joblib
import pytest
from bitcoin_scalper.core import ml_train

def test_compute_label_up():
    df = pd.DataFrame({'close': [100, 101, 102, 104, 105, 106]})
    label = ml_train.compute_label(df, price_col='close', horizon=3, up_thr=0.03, down_thr=-0.02)
    # Le prix monte de 100 à 104 (+4%) sans baisser de -2% dans les 3 prochaines bougies
    assert label.iloc[0] == 1

def test_compute_label_down():
    df = pd.DataFrame({'close': [100, 99, 98, 97, 96, 95]})
    label = ml_train.compute_label(df, price_col='close', horizon=3, up_thr=0.03, down_thr=-0.02)
    # Le prix baisse de 100 à 97 (-3%)
    assert label.iloc[0] == 0

def test_prepare_features():
    df = pd.DataFrame({
        'close': [1, 2, 3],
        'signal': [0, 1, -1],
        'label': [1, 0, 1],
        'timestamp': pd.date_range('2020-01-01', periods=3)
    })
    features = ml_train.prepare_features(df)
    assert 'signal' not in features.columns
    assert 'label' not in features.columns
    assert 'timestamp' not in features.columns

def test_train_ml_model_pipeline():
    # DataFrame factice
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 100),
        'ema_10': np.random.rand(100),
        'ema_21': np.random.rand(100),
        'ema_50': np.random.rand(100),
        'rsi': np.random.rand(100)*100,
        'macd': np.random.rand(100),
        'supertrend': np.random.rand(100),
        'atr': np.random.rand(100),
        'bb_high': np.random.rand(100),
        'bb_low': np.random.rand(100),
        'bb_width': np.random.rand(100),
    })
    df['label'] = np.random.randint(0, 2, size=100)
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, 'features.csv')
        model_path = os.path.join(tmpdir, 'model.pkl')
        scaler_path = os.path.join(tmpdir, 'scaler.pkl')
        df.to_csv(csv_path)
        clf, scaler = ml_train.train_ml_model(csv_path, model_out=model_path, scaler_out=scaler_path, test_size=0.2)
        assert os.path.exists(model_path)
        assert os.path.exists(scaler_path)
        loaded_clf = joblib.load(model_path)
        loaded_scaler = joblib.load(scaler_path)
        assert hasattr(loaded_clf, 'predict')
        assert hasattr(loaded_scaler, 'transform')

def test_analyse_label_balance_equilibre():
    df = pd.DataFrame({"signal": [0, 1, 0, 1]})
    counts = ml_train.analyse_label_balance(df, label_col="signal")
    assert abs(counts[0] - 0.5) < 1e-6
    assert abs(counts[1] - 0.5) < 1e-6

def test_analyse_label_balance_desequilibre():
    df = pd.DataFrame({"signal": [0, 0, 0, 1]})
    counts = ml_train.analyse_label_balance(df, label_col="signal")
    assert abs(counts[0] - 0.75) < 1e-6
    assert abs(counts[1] - 0.25) < 1e-6

def test_analyse_label_balance_absent():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    try:
        ml_train.analyse_label_balance(df, label_col="signal")
        assert False, "Doit lever une ValueError si colonne absente"
    except ValueError as e:
        assert "absente" in str(e)

def test_train_ml_model_with_smote(monkeypatch):
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 100),
        'ema_21': np.random.rand(100),
        'ema_50': np.random.rand(100),
        'rsi': np.random.rand(100)*100,
        'supertrend': np.random.rand(100),
        'atr': np.random.rand(100),
        'label': [0]*90 + [1]*10  # Déséquilibre fort
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, 'features.csv')
        model_path = os.path.join(tmpdir, 'model.pkl')
        scaler_path = os.path.join(tmpdir, 'scaler.pkl')
        df.to_csv(csv_path)
        # Mock SMOTE si imblearn absent
        if not hasattr(ml_train, '_HAS_SMOTE') or not ml_train._HAS_SMOTE:
            class DummySMOTE:
                def __init__(self, random_state=None): pass
                def fit_resample(self, X, y): return X, y
            monkeypatch.setattr(ml_train, 'SMOTE', DummySMOTE)
            monkeypatch.setattr(ml_train, '_HAS_SMOTE', True)
        clf, scaler = ml_train.train_ml_model(csv_path, model_out=model_path, scaler_out=scaler_path, use_smote=True)
        assert os.path.exists(model_path)
        assert os.path.exists(scaler_path)
        loaded_clf = joblib.load(model_path)
        loaded_scaler = joblib.load(scaler_path)
        assert hasattr(loaded_clf, 'predict')
        assert hasattr(loaded_scaler, 'transform') 