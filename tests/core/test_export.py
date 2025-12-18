import pytest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from bitcoin_scalper.core.export import save_objects, load_objects

def make_model():
    X = pd.DataFrame({'a': np.random.randn(30), 'b': np.random.randn(30)})
    y = np.random.choice([-1, 0, 1], 30)
    model = CatBoostClassifier(loss_function='MultiClass', iterations=10, verbose=0)
    model.fit(X, y)
    return model, X

def test_save_and_load_objects(tmp_path):
    model, X = make_model()
    pipeline = {'dummy': 1}
    encoders = {'label': {0: 'zero', 1: 'one', -1: 'minus'}}
    scaler = {'mean': 0, 'std': 1}
    prefix = str(tmp_path / "test_export")
    save_objects(model, pipeline, encoders, scaler, prefix)
    loaded = load_objects(prefix)
    # Vérifie que tout est bien chargé
    assert loaded['model'] is not None
    assert loaded['pipeline'] == pipeline
    assert loaded['encoders'] == encoders
    assert loaded['scaler'] == scaler
    # Prédictions identiques
    y_pred1 = model.predict(X)
    y_pred2 = loaded['model'].predict(X)
    assert np.allclose(y_pred1, y_pred2)

def test_save_and_load_pipeline(tmp_path):
    """Test saving a sklearn Pipeline containing a CatBoost model"""
    model, X = make_model()
    # Create a pipeline with scaler and model
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', model)
    ])
    # Fit the scaler
    pipeline.named_steps['scaler'].fit(X)
    
    prefix = str(tmp_path / "test_pipeline_export")
    # Pass the pipeline as the model parameter
    save_objects(pipeline, None, None, None, prefix)
    loaded = load_objects(prefix)
    
    # Vérifie que le modèle est chargé
    assert loaded['model'] is not None
    # Vérifie que le pipeline est sauvegardé
    assert loaded['pipeline'] is not None
    # Vérifie que les prédictions sont identiques via le modèle extrait
    y_pred1 = model.predict(X)
    y_pred2 = loaded['model'].predict(X)
    assert np.allclose(y_pred1, y_pred2)

def test_save_load_missing_file(tmp_path):
    prefix = str(tmp_path / "missing_export")
    # Aucun fichier sauvegardé
    with pytest.raises(FileNotFoundError):
        load_objects(prefix)

def test_save_load_permission_error(monkeypatch, tmp_path):
    model, X = make_model()
    prefix = str(tmp_path / "perm_export")
    # Simule une erreur d'ouverture fichier
    def bad_open(*a, **k):
        raise PermissionError("forbidden")
    monkeypatch.setattr("builtins.open", bad_open)
    with pytest.raises(PermissionError):
        # We must provide at least one artifact to trigger `open()`
        save_objects(model, {'pipeline': True}, None, None, prefix)