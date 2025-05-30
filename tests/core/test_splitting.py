import pytest
import pandas as pd
import numpy as np
from bitcoin_scalper.core.splitting import split_dataset

def make_df_splitting():
    idx = pd.date_range('2024-06-01', periods=100, freq='H', tz='UTC')
    df = pd.DataFrame({
        'feature1': np.arange(len(idx)),
        'label': np.random.choice([-1,0,1], len(idx))
    }, index=idx)
    return df

def test_split_dataset_fixed_proportions():
    df = make_df_splitting()
    train, val, test = split_dataset(df, method='fixed', train_frac=0.6, val_frac=0.2, test_frac=0.2)
    n = len(df)
    assert abs(len(train) - 0.6*n) <= 1
    assert abs(len(val) - 0.2*n) <= 1
    assert abs(len(test) - 0.2*n) <= 2
    # Aucune intersection
    assert set(train.index).isdisjoint(val.index)
    assert set(val.index).isdisjoint(test.index)
    assert set(train.index).isdisjoint(test.index)

def test_split_dataset_purge():
    df = make_df_splitting()
    train, val, test = split_dataset(df, method='fixed', train_frac=0.6, val_frac=0.2, test_frac=0.2, purge_window=5)
    # Les 5 premiers/derniers de val et les 5 premiers de test sont purgés
    assert len(val) <= int(0.2*len(df)) - 10
    assert len(test) <= int(0.2*len(df)) - 5

def test_split_dataset_purged_kfold():
    df = make_df_splitting()
    train, val, test = split_dataset(df, method='purged_kfold', purge_window=3)
    # Les folds ne se chevauchent pas
    assert set(train.index).isdisjoint(val.index)
    assert set(val.index).isdisjoint(test.index)
    assert set(train.index).isdisjoint(test.index)
    # La purge est bien appliquée
    assert val.index.min() > train.index.max()
    assert test.index.min() > val.index.max()

def test_split_dataset_error_cases():
    df = make_df_splitting()
    # Mauvaise méthode
    with pytest.raises(ValueError):
        split_dataset(df, method='unknown')
    # Purged sans purge_window
    with pytest.raises(ValueError):
        split_dataset(df, method='purged_kfold')

def test_split_dataset_missing_timestamp():
    df = make_df_splitting().reset_index(drop=True)
    # Index non temporel : doit fonctionner mais respecter l'ordre
    train, val, test = split_dataset(df, method='fixed', train_frac=0.5, val_frac=0.3, test_frac=0.2)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()

def test_split_dataset_small():
    df = make_df_splitting().iloc[:10]
    train, val, test = split_dataset(df, method='fixed', train_frac=0.5, val_frac=0.3, test_frac=0.2)
    # Les tailles sont correctes même sur petit jeu
    assert len(train) + len(val) + len(test) == 10 