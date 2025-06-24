import pandas as pd
import numpy as np
from bitcoin_scalper.core.labeling import generate_q_values, generate_labels

def test_generate_q_values_basic():
    df = pd.DataFrame({
        '<CLOSE>': np.linspace(10000, 10100, 100)
    })
    q_df = generate_q_values(df, horizon=5, fee=0.001, spread=0.0005, slippage=0.0002)
    assert all(col in q_df.columns for col in ['q_buy', 'q_sell', 'q_hold'])
    assert np.allclose(q_df['q_hold'], 0, equal_nan=True)
    # Vérifie que q_buy et q_sell sont opposés (hors frais)
    mask = ~q_df['q_buy'].isna() & ~q_df['q_sell'].isna()
    assert np.allclose(q_df.loc[mask, 'q_buy'] + q_df.loc[mask, 'q_sell'], -4 * (0.001 + 0.0005 + 0.0002), atol=1e-8) 

def test_generate_labels_column_variants():
    # Cas 1 : colonnes classiques
    df1 = pd.DataFrame({'<CLOSE>': np.linspace(100, 110, 50), 'log_return_1m': np.random.randn(50)})
    labels1 = generate_labels(df1, horizon=5)
    assert len(labels1) > 0
    # Cas 2 : colonnes préfixées 1min_
    df2 = pd.DataFrame({'1min_<CLOSE>': np.linspace(100, 110, 50), 'log_return_1m': np.random.randn(50)})
    labels2 = generate_labels(df2, horizon=5)
    assert len(labels2) > 0
    # Cas 3 : colonnes 1min_<CLOSE> et 1min_log_return
    df3 = pd.DataFrame({'1min_<CLOSE>': np.linspace(100, 110, 50), '1min_log_return': np.random.randn(50)})
    labels3 = generate_labels(df3, horizon=5)
    assert len(labels3) > 0
    # Cas 4 : colonnes close et log_return
    df4 = pd.DataFrame({'close': np.linspace(100, 110, 50), 'log_return': np.random.randn(50)})
    labels4 = generate_labels(df4, horizon=5)
    assert len(labels4) > 0 