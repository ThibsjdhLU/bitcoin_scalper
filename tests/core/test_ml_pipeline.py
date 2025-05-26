import pytest
import pandas as pd
import numpy as np
from app.core.ml_pipeline import MLPipeline
from unittest.mock import patch, MagicMock

def make_data(n=100):
    np.random.seed(42)
    X = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
        "feat3": np.random.randn(n)
    })
    y = (X["feat1"] + X["feat2"] * 0.5 + np.random.randn(n) * 0.1 > 0).astype(int)
    return X, y

def make_seq_data(n=100, seq=5, features=3):
    np.random.seed(42)
    X = np.random.randn(n, seq, features)
    y = (X.mean(axis=(1,2)) > 0).astype(int)
    return X, pd.Series(y)

@pytest.fixture
def tabular_data():
    X = pd.DataFrame(np.random.randn(100, 8), columns=[f"f{i}" for i in range(8)])
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y

@pytest.fixture
def seq_data():
    X = np.random.randn(100, 10, 8)  # [n, seq, features]
    y = np.random.randint(0, 2, 100)
    return X, y

@patch("app.core.ml_pipeline.DVCManager")
@patch("app.core.ml_pipeline.joblib.dump")
@patch("app.core.ml_pipeline.joblib.load")
def test_rf_fit_predict(mock_load, mock_dump, mock_dvc, tabular_data):
    X, y = tabular_data
    pipe = MLPipeline(model_type="random_forest", dvc_track=True)
    metrics = pipe.fit(X, y)
    assert "val_accuracy" in metrics
    preds = pipe.predict(X)
    assert preds.shape == (100,)
    probas = pipe.predict_proba(X)
    assert probas.shape[0] == 100
    pipe.save("model_rf.pkl")
    pipe.load("model_rf.pkl")

@patch("app.core.ml_pipeline.DVCManager")
def test_xgb_fit_predict(mock_dvc, tabular_data):
    X, y = tabular_data
    pipe = MLPipeline(model_type="xgboost", dvc_track=True)
    metrics = pipe.fit(X, y)
    assert "val_accuracy" in metrics
    preds = pipe.predict(X)
    assert preds.shape == (100,)
    probas = pipe.predict_proba(X)
    assert probas.shape[0] == 100

@patch("app.core.ml_pipeline.DVCManager")
def test_dnn_fit_predict(mock_dvc, seq_data):
    X, y = seq_data
    pipe = MLPipeline(model_type="dnn", params={"input_dim":8, "output_dim":2}, epochs=2)
    metrics = pipe.fit(X, y, epochs=2)
    assert "val_accuracy" in metrics
    preds = pipe.predict(X)
    assert preds.shape == (100,)
    probas = pipe.predict_proba(X)
    assert probas.shape[0] == 100

@patch("app.core.ml_pipeline.DVCManager")
def test_lstm_fit_predict(mock_dvc, seq_data):
    X, y = seq_data
    pipe = MLPipeline(model_type="lstm", params={"input_dim":8, "output_dim":2}, epochs=2)
    metrics = pipe.fit(X, y, epochs=2)
    assert "val_accuracy" in metrics
    preds = pipe.predict(X)
    assert preds.shape == (100,)

@patch("app.core.ml_pipeline.DVCManager")
def test_transformer_fit_predict(mock_dvc, seq_data):
    X, y = seq_data
    pipe = MLPipeline(model_type="transformer", params={"input_dim":8, "output_dim":2}, epochs=2)
    metrics = pipe.fit(X, y, epochs=2)
    assert "val_accuracy" in metrics
    preds = pipe.predict(X)
    assert preds.shape == (100,)

@patch("app.core.ml_pipeline.DVCManager")
def test_cnn1d_fit_predict(mock_dvc, seq_data):
    X, y = seq_data
    pipe = MLPipeline(model_type="cnn1d", params={"input_dim":8, "output_dim":2}, epochs=2)
    metrics = pipe.fit(X, y, epochs=2)
    assert "val_accuracy" in metrics
    preds = pipe.predict(X)
    assert preds.shape == (100,)

@patch("app.core.ml_pipeline.shap.TreeExplainer")
def test_explain_shap(mock_shap, tabular_data):
    X, y = tabular_data
    pipe = MLPipeline(model_type="random_forest")
    pipe.fit(X, y)
    mock_shap.return_value.shap_values.return_value = np.random.randn(*X.shape)
    vals = pipe.explain(X, method="shap")
    assert vals.shape[0] == X.shape[0]

@patch("app.core.ml_pipeline.DVCManager")
def test_tune_gridsearch(mock_dvc, tabular_data):
    X, y = tabular_data
    pipe = MLPipeline(model_type="random_forest")
    metrics = pipe.tune(X, y, param_grid={"n_estimators":[5,10]}, cv=2)
    assert "best_params" in metrics

@patch("app.core.ml_pipeline.DVCManager")
def test_invalid_model_type(mock_dvc):
    with pytest.raises(ValueError):
        MLPipeline(model_type="invalid")

@patch("app.core.ml_pipeline.DVCManager")
def test_explain_not_implemented(mock_dvc, tabular_data):
    X, y = tabular_data
    pipe = MLPipeline(model_type="dnn")
    with pytest.raises(NotImplementedError):
        pipe.explain(X, method="lime") 