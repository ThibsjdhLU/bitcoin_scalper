import pytest
import numpy as np
import pandas as pd
from bitcoin_scalper.core.evaluation import evaluate_classification, evaluate_financial, plot_pnl_curve

def test_evaluate_classification_scores():
    y_true = np.array([1, 0, -1, 1, 0, -1])
    y_pred = np.array([1, 0, -1, 0, 0, -1])
    res = evaluate_classification(y_true, y_pred)
    assert 0 <= res['accuracy'] <= 1
    assert 0 <= res['f1_macro'] <= 1
    assert isinstance(res['confusion_matrix'], np.ndarray)
    assert isinstance(res['classification_report'], str)

def test_evaluate_classification_shape_error():
    y_true = np.array([1, 0, -1])
    y_pred = np.array([1, 0])
    with pytest.raises(ValueError):
        evaluate_classification(y_true, y_pred)

def test_evaluate_financial_basic():
    y_pred = np.array([1, 0, -1, 1, 0, -1])
    returns = np.array([0.01, 0.0, -0.02, 0.03, 0.0, -0.01])
    res = evaluate_financial(y_pred, returns)
    assert 'pnl_cum' in res and 'sharpe' in res and 'max_drawdown' in res
    assert isinstance(res['pnl_cum_curve'], np.ndarray)

def test_evaluate_financial_index():
    y_pred = np.array([1, 0, -1, 1, 0, -1])
    returns = np.array([0.01, 0.0, -0.02, 0.03, 0.0, -0.01])
    idx = pd.date_range('2024-01-01', periods=6, freq='T')
    res = evaluate_financial(y_pred, returns, index=idx)
    assert isinstance(res['pnl_cum_curve'], pd.Series)
    assert (res['pnl_cum_curve'].index == idx).all()

def test_evaluate_financial_shape_error():
    y_pred = np.array([1, 0, -1])
    returns = np.array([0.01, 0.0])
    with pytest.raises(ValueError):
        evaluate_financial(y_pred, returns)
    idx = pd.date_range('2024-01-01', periods=2, freq='T')
    with pytest.raises(ValueError):
        evaluate_financial(y_pred, np.array([0.01, 0.0, -0.02]), index=idx)

def test_evaluate_financial_nan():
    y_pred = np.array([1, 0, -1, 1])
    returns = np.array([0.01, np.nan, -0.02, 0.03])
    res = evaluate_financial(y_pred, returns)
    # Les NaN doivent être traités comme 0
    assert not np.isnan(res['pnl_cum'])

def test_plot_pnl_curve():
    pnl = pd.Series(np.cumsum(np.random.randn(100)), index=pd.date_range('2024-01-01', periods=100, freq='T'))
    fig = plot_pnl_curve(pnl)
    assert hasattr(fig, 'savefig') 