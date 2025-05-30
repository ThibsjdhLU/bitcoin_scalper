import pytest
import numpy as np
import pandas as pd
from bitcoin_scalper.core.modeling import train_model, predict
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

def make_data():
    # Génère un jeu synthétique multiclasse, 3 features, 3 classes
    n = 120
    X = pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
        'f3': np.random.randn(n)
    })
    y = np.random.choice([-1, 0, 1], n)
    # Corrélation faible mais non nulle
    X.loc[y == 1, 'f1'] += 2
    X.loc[y == -1, 'f2'] -= 2
    return X.iloc[:80], y[:80], X.iloc[80:100], y[80:100], X.iloc[100:], y[100:]

def test_train_model_grid():
    X_train, y_train, X_val, y_val, X_test, y_test = make_data()
    model = train_model(X_train, y_train, X_val, y_val, method='grid', early_stopping_rounds=5)
    y_pred = predict(model, X_test)
    score = f1_score(y_test, y_pred, average='macro')
    # Score doit être supérieur à 0.2 (aléatoire = 0.33 max)
    assert score > 0.2

def test_train_model_optuna():
    X_train, y_train, X_val, y_val, X_test, y_test = make_data()
    try:
        model = train_model(X_train, y_train, X_val, y_val, method='optuna', early_stopping_rounds=5)
    except ImportError:
        pytest.skip('Optuna non installé')
    y_pred = predict(model, X_test)
    score = f1_score(y_test, y_pred, average='macro')
    assert score > 0.2

def test_train_model_early_stopping():
    X_train, y_train, X_val, y_val, _, _ = make_data()
    model = train_model(X_train, y_train, X_val, y_val, method='grid', early_stopping_rounds=2)
    # Vérifie que l'attribut best_iteration_ existe (early stopping activé)
    assert hasattr(model, 'best_iteration_')

def test_train_model_error_shape():
    X_train, y_train, X_val, y_val, _, _ = make_data()
    with pytest.raises(ValueError):
        train_model(X_train.iloc[:-1], y_train, X_val, y_val)
    with pytest.raises(ValueError):
        train_model(X_train, y_train, X_val.iloc[:-1], y_val)

def test_train_model_error_classes():
    X_train, y_train, X_val, y_val, _, _ = make_data()
    y_train2 = y_train.copy()
    y_train2[:] = 99
    with pytest.raises(ValueError):
        train_model(X_train, y_train2, X_val, y_val)

def test_predict_error():
    X_train, y_train, X_val, y_val, X_test, _ = make_data()
    model = train_model(X_train, y_train, X_val, y_val, method='grid', early_stopping_rounds=5)
    with pytest.raises(ValueError):
        predict('not_a_model', X_test) 