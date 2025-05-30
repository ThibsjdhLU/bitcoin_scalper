import pytest
import pandas as pd
import numpy as np
from bitcoin_scalper.core.labeling import generate_labels

def make_df_labeling():
    # Génère un DataFrame minute synthétique sur 50 minutes
    idx = pd.date_range('2024-06-01 00:00', periods=50, freq='T', tz='UTC')
    close = np.concatenate([
        np.linspace(100, 110, 20),  # tendance haussière
        np.linspace(110, 100, 15),  # tendance baissière
        np.full(15, 105)            # neutre
    ])
    log_return_1m = np.log(np.concatenate([[np.nan], close[1:] / close[:-1]]))
    df = pd.DataFrame({'<CLOSE>': close, 'log_return_1m': log_return_1m}, index=idx)
    return df

def test_generate_labels_all_classes():
    df = make_df_labeling()
    labels = generate_labels(df, horizon=5, k=0.2)
    # Doit contenir les 3 classes
    assert set(labels.unique()) == {-1, 0, 1}
    # Index aligné, pas de NaN
    assert labels.isna().sum() == 0
    assert labels.index.is_monotonic_increasing

def test_generate_labels_temporality():
    df = make_df_labeling()
    labels = generate_labels(df, horizon=5, k=0.2)
    # Vérifie qu'aucune ligne n'utilise le futur pour le label
    # On modifie CLOSE après t et vérifie que le label en t ne change pas
    df2 = df.copy()
    df2.iloc[10+5, df2.columns.get_loc('<CLOSE>')] += 1000
    labels2 = generate_labels(df2, horizon=5, k=0.2)
    # Les labels avant t=10 doivent être identiques
    pd.testing.assert_series_equal(labels.iloc[:10], labels2.iloc[:10])

def test_generate_labels_threshold_coherence():
    df = make_df_labeling()
    # On force une forte volatilité locale sur une fenêtre
    df.loc[df.index[30:40], 'log_return_1m'] = 0.5
    labels = generate_labels(df, horizon=5, k=0.5)
    # Les labels dans cette zone doivent être plus souvent neutres (seuil élevé)
    neutres = (labels.loc[df.index[30:40]] == 0).sum()
    assert neutres >= 5

def test_generate_labels_invalid_columns():
    df = pd.DataFrame({'<CLOSE>': [1,2,3]})
    with pytest.raises(ValueError):
        generate_labels(df)
    df = pd.DataFrame({'log_return_1m': [0,0,0]})
    with pytest.raises(ValueError):
        generate_labels(df)

def test_generate_labels_nan_and_window():
    df = make_df_labeling()
    # Les 29 premières lignes doivent être exclues (rolling_std min_periods=30)
    labels = generate_labels(df, horizon=5, k=0.2)
    assert labels.index[0] == df.index[29]
    # Les dernières lignes sans horizon suffisant doivent être exclues
    assert labels.index[-1] == df.index[-6] 