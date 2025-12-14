import pytest
import pandas as pd
import numpy as np
import os
from bitcoin_scalper.core.ml_orchestrator import run_ml_pipeline

def make_synthetic_ml_data(n=100):
    idx = pd.date_range('2024-06-01', periods=n, freq='H', tz='UTC')
    X = np.random.randn(n, 5)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f'feat{i}' for i in range(5)], index=idx)
    df['label'] = y
    return df

def test_run_ml_pipeline(tmp_path):
    df = make_synthetic_ml_data(120)
    out_dir = tmp_path / "ml_report"
    report = run_ml_pipeline(
        df, label_col="label", model_type="catboost",
        split_params={"train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2, "horizon": 2},
        cv_params={"n_splits": 3},
        out_dir=str(out_dir),
        random_state=123
    )
    # Vérifie que tous les rapports sont générés
    for key, path in report.items():
        if key == 'model_object':
            continue
        assert os.path.exists(path), f"Fichier manquant : {path}"
    # Vérifie la cohérence des métriques
    import json
    with open(report["metrics"]) as f:
        metrics = json.load(f)
    assert "val" in metrics and "test" in metrics
    assert 0 <= metrics["val"]["accuracy"] <= 1
    assert 0 <= metrics["test"]["accuracy"] <= 1
    # Vérifie la non-fuite (aucune intersection entre splits)
    val_pred = pd.read_csv(report["val_predictions"], index_col=0)
    test_pred = pd.read_csv(report["test_predictions"], index_col=0)
    assert set(val_pred.index).isdisjoint(test_pred.index) 