import pytest
import pandas as pd
import numpy as np
from bitcoin_scalper.core.balancing import balance_by_block

def make_df_balancing():
    # 2 jours, 3 classes, déséquilibre, colonnes additionnelles
    idx = pd.date_range('2024-06-01', periods=48, freq='H', tz='UTC')
    labels = np.array([1]*10 + [0]*6 + [-1]*2 + [1]*8 + [0]*8 + [-1]*6 + [1]*4 + [0]*2 + [-1]*2)
    # Pad si besoin
    if len(labels) < len(idx):
        labels = np.concatenate([labels, np.random.choice([-1,0,1], len(idx)-len(labels))])
    df = pd.DataFrame({
        'feature1': np.arange(len(idx)),
        'feature2': np.arange(len(idx))*2,
        'label': labels
    }, index=idx)
    return df

def test_balance_by_block_strict_equality():
    df = make_df_balancing()
    balanced = balance_by_block(df, label_col='label', block_duration='1D', min_block_size=10)
    # Pour chaque bloc, égalité stricte des classes
    for block_start, block in balanced.groupby(pd.Grouper(freq='1D')):
        counts = block['label'].value_counts()
        assert counts.nunique() == 1
        assert set(counts.index) <= {-1,0,1}

def test_balance_by_block_preserve_columns():
    df = make_df_balancing()
    balanced = balance_by_block(df, label_col='label', block_duration='1D', min_block_size=10)
    assert set(['feature1','feature2','label']).issubset(balanced.columns)

def test_balance_by_block_min_block_size():
    df = make_df_balancing()
    # Met un bloc trop petit
    df_small = df.iloc[:5].copy()
    df_rest = df.iloc[5:].copy()
    df_all = pd.concat([df_small, df_rest])
    balanced = balance_by_block(df_all, label_col='label', block_duration='1D', min_block_size=10)
    # Le premier bloc doit être ignoré
    assert balanced.index.min() > df_small.index.max()

def test_balance_by_block_classes_absentes():
    df = make_df_balancing()
    # Force un bloc sans classe -1
    df.loc[df.index[:12], 'label'] = 1
    balanced = balance_by_block(df, label_col='label', block_duration='1D', min_block_size=10)
    # Le bloc sans classe -1 doit être ignoré
    assert (balanced.index < df.index[12]).sum() == 0

def test_balance_by_block_shuffle():
    df = make_df_balancing()
    balanced1 = balance_by_block(df, label_col='label', block_duration='1D', min_block_size=10, shuffle=False)
    balanced2 = balance_by_block(df, label_col='label', block_duration='1D', min_block_size=10, shuffle=True)
    # Même index, mais potentiellement ordre différent
    assert set(balanced1.index) == set(balanced2.index)
    # Les valeurs de feature1 peuvent différer dans l'ordre
    assert not balanced1.equals(balanced2)

def test_balance_by_block_error_cases():
    df = make_df_balancing()
    # Colonne label absente
    with pytest.raises(ValueError):
        balance_by_block(df.drop(columns=['label']))
    # Aucun bloc équilibré possible
    df2 = df.copy()
    df2['label'] = 1
    with pytest.raises(ValueError):
        balance_by_block(df2) 