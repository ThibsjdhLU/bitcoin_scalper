import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

logger = logging.getLogger("bitcoin_scalper.evaluation")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcule les métriques de classification standard (macro) pour un problème multi-classes.

    :param y_true: Labels réels
    :param y_pred: Prédictions du modèle
    :return: Dictionnaire des scores (accuracy, precision, recall, f1, confusion_matrix, classification_report)
    """
    if len(y_true) != len(y_pred):
        logger.error("y_true et y_pred de tailles différentes")
        raise ValueError("y_true et y_pred de tailles différentes")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    logger.info(f"Accuracy={acc:.4f}, F1_macro={f1:.4f}")
    return {
        'accuracy': acc,
        'precision_macro': prec,
        'recall_macro': rec,
        'f1_macro': f1,
        'confusion_matrix': cm,
        'classification_report': report
    }

def evaluate_financial(y_pred: np.ndarray, returns: np.ndarray, index: Optional[pd.Index] = None) -> Dict[str, float]:
    """
    Calcule les métriques financières (PnL, Sharpe, drawdown) à partir d'un signal discret et des retours log.

    :param y_pred: Signal discret (1, 0, -1)
    :param returns: log_return_1m aligné
    :param index: Index temporel (optionnel)
    :return: Dictionnaire des métriques financières
    """
    if len(y_pred) != len(returns):
        logger.error("y_pred et returns de tailles différentes")
        raise ValueError("y_pred et returns de tailles différentes")
    if index is not None and len(index) != len(y_pred):
        logger.error("Index non aligné avec y_pred/returns")
        raise ValueError("Index non aligné avec y_pred/returns")
    pnl = y_pred * returns
    pnl = np.nan_to_num(pnl)
    pnl_cum = np.cumsum(pnl)
    # Annualisation (minute -> an)
    periods_per_year = 365*24*60
    mean = np.mean(pnl) * periods_per_year
    std = np.std(pnl) * np.sqrt(periods_per_year)
    sharpe = mean / std if std > 0 else 0.0
    # Max drawdown
    running_max = np.maximum.accumulate(pnl_cum)
    drawdown = running_max - pnl_cum
    max_drawdown = np.max(drawdown)
    logger.info(f"PnL cumulé={pnl_cum[-1]:.4f}, Sharpe={sharpe:.4f}, Max drawdown={max_drawdown:.4f}")
    return {
        'pnl_cum': pnl_cum[-1],
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'pnl_cum_curve': pnl_cum if index is None else pd.Series(pnl_cum, index=index)
    }

def plot_pnl_curve(pnl_series: pd.Series) -> plt.Figure:
    """
    Trace la courbe du PnL cumulé.

    :param pnl_series: Série temporelle du PnL cumulé
    :return: Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    pnl_series.plot(ax=ax, color='blue', lw=2)
    ax.set_title('Courbe du PnL cumulé')
    ax.set_xlabel('Temps')
    ax.set_ylabel('PnL cumulé')
    ax.grid(True)
    return fig 