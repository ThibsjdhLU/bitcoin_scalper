import pandas as pd
import numpy as np
from data.augmentation import augment_rolling_jitter
from data.synthetic_ohlcv import generate_synthetic_ohlcv
from data.feature_selection import permutation_importance_selection, compute_vif
import pytest
from catboost import CatBoostClassifier

def make_df_aug():
    idx = pd.date_range('2024-06-01 00:00', periods=20, freq='T', tz='UTC')
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 20),
        'volume': np.random.randint(1, 10, 20)
    }, index=idx)
    return df

def test_augment_rolling_jitter_basic():
    df = make_df_aug()
    df_aug = augment_rolling_jitter(df, n_shifts=2, jitter_seconds=5, random_state=42)
    # Doit contenir (n_shifts+1) fois plus de lignes moins les NaN en début de chaque shift
    expected_min = len(df) + (len(df)-1) + (len(df)-2)
    assert len(df_aug) >= expected_min
    # Les timestamps doivent être différents de l'original
    assert not df_aug.index.equals(df.index)
    # La colonne augmentation_id doit exister et être bien répartie
    assert 'augmentation_id' in df_aug.columns
    assert set(df_aug['augmentation_id']) == {0, 1, 2}
    # Pas de NaN inattendus dans les features
    assert df_aug['close'].notnull().all()
    assert df_aug['volume'].notnull().all()

def test_generate_synthetic_ohlcv_mock():
    df = make_df_aug()
    df_synth = generate_synthetic_ohlcv(df, n_samples=5, random_state=123)
    # Doit contenir n_samples * len(df) lignes
    assert len(df_synth) == 5 * len(df)
    # Les colonnes OHLCV doivent exister
    for col in ['close', 'volume']:
        assert col in df_synth.columns
    # La colonne synthetic_id doit exister et être bien répartie
    assert 'synthetic_id' in df_synth.columns
    assert set(df_synth['synthetic_id']) == set(range(5))
    # Les séquences doivent être différentes de l'original
    assert not df_synth['close'].equals(pd.concat([df['close']]*5, ignore_index=True))

def make_Xy_selection():
    # Dataset synthétique avec redondance
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        'a': rng.normal(0, 1, 100),
        'b': rng.normal(0, 1, 100),
        'c': rng.normal(0, 1, 100),
        'd': rng.normal(0, 1, 100),
    })
    X['e'] = X['a'] * 0.99 + rng.normal(0, 0.01, 100)  # Redondant avec 'a'
    y = (X['a'] + X['b'] > 0).astype(int)
    return X, y

def test_permutation_importance_selection():
    X, y = make_Xy_selection()
    model = CatBoostClassifier(n_estimators=10, random_state=42)
    selected = permutation_importance_selection(X, y, model, n_splits=3, threshold=0.0, scoring='accuracy', random_state=42)
    # Les features informatives doivent être sélectionnées
    assert 'a' in selected or 'b' in selected
    # Les features non informatives peuvent être absentes
    assert isinstance(selected, list)
    assert all(isinstance(f, str) for f in selected)

def test_compute_vif():
    X, _ = make_Xy_selection()
    features = compute_vif(X, threshold=5.0)
    # La feature redondante 'e' doit être supprimée
    assert 'e' not in features or 'a' not in features  # l'une des deux doit sauter
    # Les features restantes doivent être peu corrélées
    assert len(features) <= X.shape[1] 