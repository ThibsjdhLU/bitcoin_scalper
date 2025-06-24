import pytest
import pandas as pd
import numpy as np
from bitcoin_scalper.core.splitting import split_dataset, temporal_train_val_test_split, generate_time_series_folds, generate_purged_kfold_folds

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

def test_temporal_train_val_test_split_proportions(tmp_path):
    df = make_df_splitting()
    report_path = tmp_path / "split_report.json"
    train, val, test = temporal_train_val_test_split(
        df, train_frac=0.6, val_frac=0.2, test_frac=0.2, horizon=2, report_path=str(report_path)
    )
    # Proportions
    n = len(df)
    assert abs(len(train) - 0.6*n + 2) <= 2  # horizon exclus
    assert abs(len(val) - 0.2*n + 2) <= 2
    assert abs(len(test) - 0.2*n) <= 2
    # Pas d'overlap
    assert set(train.index).isdisjoint(val.index)
    assert set(val.index).isdisjoint(test.index)
    assert set(train.index).isdisjoint(test.index)
    # Rapport JSON généré
    assert report_path.exists()
    import json
    with open(report_path) as f:
        report = json.load(f)
    assert not report["overlap"]
    assert not report["leakage_risk"]

def test_temporal_train_val_test_split_dates(tmp_path):
    df = make_df_splitting()
    idx = df.index
    train_start = str(idx[0])
    train_end = str(idx[60])
    val_start = str(idx[60])
    val_end = str(idx[80])
    test_start = str(idx[80])
    test_end = str(idx[-1])
    report_path = tmp_path / "split_report2.json"
    train, val, test = temporal_train_val_test_split(
        df,
        train_start=train_start, train_end=train_end,
        val_start=val_start, val_end=val_end,
        test_start=test_start, test_end=test_end,
        horizon=1, report_path=str(report_path)
    )
    # Les bornes sont respectées
    assert train.index.min() == pd.Timestamp(train_start)
    assert train.index.max() < pd.Timestamp(train_end)
    assert val.index.min() == pd.Timestamp(val_start)
    assert val.index.max() < pd.Timestamp(val_end)
    assert test.index.min() == pd.Timestamp(test_start)
    # Pas d'overlap
    assert set(train.index).isdisjoint(val.index)
    assert set(val.index).isdisjoint(test.index)
    assert set(train.index).isdisjoint(test.index)
    # Rapport JSON généré
    assert report_path.exists()

def test_generate_time_series_folds(tmp_path):
    df = make_df_splitting()
    n_splits = 4
    report_path = tmp_path / "tscv_folds.json"
    folds = generate_time_series_folds(df, n_splits=n_splits, report_path=str(report_path))
    assert len(folds) == n_splits
    for train_idx, val_idx in folds:
        # Pas d'overlap
        assert set(train_idx).isdisjoint(val_idx)
        # Train avant val
        assert max(train_idx) < min(val_idx)
    assert report_path.exists()

def test_generate_purged_kfold_folds(tmp_path):
    df = make_df_splitting()
    n_splits = 3
    purge_window = 2
    report_path = tmp_path / "purged_folds.json"
    folds = generate_purged_kfold_folds(df, n_splits=n_splits, purge_window=purge_window, report_path=str(report_path))
    assert len(folds) == n_splits
    for train_idx, val_idx in folds:
        # Pas d'overlap
        assert set(train_idx).isdisjoint(val_idx)
        # Fenêtre de purge : aucun train_idx n'est dans la zone de purge autour de val_idx
        if len(val_idx) > 0:
            purge_zone = set(range(val_idx[0]-purge_window, val_idx[-1]+purge_window+1))
            assert set(train_idx).isdisjoint(purge_zone & set(range(len(df))))
    assert report_path.exists() 