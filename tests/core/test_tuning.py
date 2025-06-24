import pytest
import pandas as pd
import numpy as np
import os
from bitcoin_scalper.core.tuning import tune_model_hyperparams

def make_synthetic_tuning_data(n=60):
    idx = pd.date_range('2024-06-01', periods=n, freq='H', tz='UTC')
    X = np.random.randn(n, 4)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f'feat{i}' for i in range(4)], index=idx)
    df['label'] = y
    return df

def test_tune_model_hyperparams_grid(tmp_path):
    df = make_synthetic_tuning_data(60)
    param_grid = {"max_depth": [2, 3], "n_estimators": [10, 20]}
    out_dir = tmp_path / "tuning_report"
    report = tune_model_hyperparams(
        df, label_col="label", model_type="lightgbm",
        param_grid=param_grid, method="grid", n_iter=4,
        cv_params={"n_splits": 2}, out_dir=str(out_dir), random_state=123
    )
    # Vérifie que tous les rapports sont générés
    for key, path in report.items():
        assert os.path.exists(path), f"Fichier manquant : {path}"
    # Vérifie la cohérence des résultats
    import json
    with open(report["best_params"]) as f:
        best = json.load(f)
    assert "best_params" in best and "best_score" in best
    results = pd.read_csv(report["tuning_results"])
    assert not results.empty
    # Vérifie la courbe des scores
    assert os.path.exists(report["tuning_scores"]) 