import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from bitcoin_scalper.core.probability_calibration import ProbabilityCalibrator

def test_platt_scaling():
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    model = LogisticRegression().fit(X, y)
    calibrator = ProbabilityCalibrator(method="sigmoid")
    calibrator.fit(model, X, y)
    proba = calibrator.predict_proba(X)
    assert proba.shape == (200, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)

def test_isotonic_regression():
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    model = LogisticRegression().fit(X, y)
    calibrator = ProbabilityCalibrator(method="isotonic")
    calibrator.fit(model, X, y)
    proba = calibrator.predict_proba(X)
    assert proba.shape == (200, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)

def test_save_and_load(tmp_path):
    X, y = make_classification(n_samples=100, n_features=3, random_state=0)
    model = LogisticRegression().fit(X, y)
    calibrator = ProbabilityCalibrator(method="sigmoid")
    calibrator.fit(model, X, y)
    file_path = tmp_path / "calibrator.joblib"
    calibrator.save(str(file_path))
    loaded = ProbabilityCalibrator.load(str(file_path))
    proba = loaded.predict_proba(X)
    assert proba.shape == (100, 2) 