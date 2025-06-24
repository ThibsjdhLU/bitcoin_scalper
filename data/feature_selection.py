import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor

def permutation_importance_selection(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    n_splits: int = 5,
    threshold: float = 0.0,
    scoring: str = 'accuracy',
    random_state: Optional[int] = None
) -> List[str]:
    """
    Sélectionne les features robustes via permutation importance sur plusieurs splits temporels.
    :param X: DataFrame de features
    :param y: Série de labels
    :param model: Modèle ML (fit/predict)
    :param n_splits: Nombre de splits temporels (TimeSeriesSplit)
    :param threshold: Seuil minimal d'importance moyenne pour garder une feature
    :param scoring: Métrique de scoring (sklearn)
    :param random_state: Graine aléatoire
    :return: Liste des features sélectionnées
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    importances = pd.DataFrame(index=X.columns)
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        result = permutation_importance(model, X_test, y_test, scoring=scoring, n_repeats=5, random_state=random_state)
        importances[f'split_{i}'] = result.importances_mean
    mean_importance = importances.mean(axis=1)
    selected = mean_importance[mean_importance > threshold].index.tolist()
    return selected

def compute_vif(X: pd.DataFrame, threshold: float = 5.0) -> List[str]:
    """
    Supprime les features avec un VIF (Variance Inflation Factor) supérieur au seuil.
    :param X: DataFrame de features numériques
    :param threshold: Seuil de VIF au-delà duquel une feature est considérée comme redondante
    :return: Liste des features conservées (VIF <= threshold)
    """
    X_ = X.select_dtypes(include=[np.number]).dropna()
    features = list(X_.columns)
    while True:
        vif = pd.Series([variance_inflation_factor(X_.values, i) for i in range(X_.shape[1])], index=X_.columns)
        max_vif = vif.max()
        if max_vif > threshold:
            drop_feat = vif.idxmax()
            features.remove(drop_feat)
            X_ = X_[features]
        else:
            break
    return features 