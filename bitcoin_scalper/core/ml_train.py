import pandas as pd
import numpy as np
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, classification_report
import joblib
from typing import Optional

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except ImportError:
    _HAS_SMOTE = False


def compute_label(df: pd.DataFrame, price_col: str = "close", horizon: int = 5, up_thr: float = 0.01, down_thr: float = -0.005) -> pd.Series:
    """
    Calcule le label binaire :
    1 si le prix monte de +1% dans les N prochaines bougies sans JAMAIS baisser de -0.5% dans la même fenêtre, sinon 0.
    """
    arr = df[price_col].values
    labels = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        future = arr[i+1:i+1+horizon]
        if len(future) == 0:
            labels[i] = 0
            continue
        max_up = (future.max() - arr[i]) / arr[i]
        min_down = (future.min() - arr[i]) / arr[i]
        if max_up >= up_thr and min_down > down_thr:
            labels[i] = 1
        else:
            labels[i] = 0
    return pd.Series(labels, index=df.index)


def prepare_features(df: pd.DataFrame, drop_cols=None) -> pd.DataFrame:
    """
    Nettoie et prépare les features pour le ML (suppression colonnes inutiles, gestion NaN).
    """
    if drop_cols is None:
        drop_cols = ["timestamp", "signal", "label"]
    features = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features


def train_ml_model(
    input_csv: str,
    model_out: str = "model_rf.pkl",
    scaler_out: str = "scaler.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
    horizon: int = 5,
    up_thr: float = 0.01,
    down_thr: float = -0.005,
    use_smote: bool = False
):
    """
    Entraîne un modèle RandomForest pour prédire le label de trading.
    Peut appliquer SMOTE pour rééquilibrer le jeu de données si besoin.
    Args:
        input_csv (str): Chemin du CSV de features.
        model_out (str): Fichier de sortie du modèle.
        scaler_out (str): Fichier de sortie du scaler.
        test_size (float): Proportion du jeu de test (0.2 = 20%).
        random_state (int): Graine aléatoire.
        horizon (int): Horizon pour la génération du label (si absent).
        up_thr (float): Seuil de hausse pour label=1.
        down_thr (float): Seuil de baisse pour label=0.
        use_smote (bool): Active SMOTE pour rééquilibrer le train si True.
    Sécurité :
        - Ne modifie jamais les données d'origine.
        - Gère l'absence de SMOTE.
    """
    df = pd.read_csv(input_csv, index_col=0, parse_dates=True)
    if "label" not in df.columns:
        df["label"] = compute_label(df, price_col="close", horizon=horizon, up_thr=up_thr, down_thr=down_thr)
    df = df.dropna(subset=["label"])  # On ne garde que les lignes labellisées
    X = prepare_features(df)
    y = df.loc[X.index, "label"]
    # Split temporel (pas de shuffle)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Analyse du déséquilibre
    print("\nAnalyse du déséquilibre sur le train :")
    analyse_label_balance(pd.DataFrame({"label": y_train}), label_col="label")
    # Optionnel : SMOTE
    if use_smote:
        if not _HAS_SMOTE:
            print("[SECURITE] SMOTE non disponible : installez imblearn pour activer le rééquilibrage.")
        else:
            try:
                sm = SMOTE(random_state=random_state)
                X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
                print("[INFO] SMOTE appliqué : jeu de train rééquilibré.")
                analyse_label_balance(pd.DataFrame({"label": y_train}), label_col="label")
            except Exception as e:
                print(f"[ERREUR] SMOTE a échoué : {e}")
    # Entraînement
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)
    # Évaluation
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy: {acc:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f}")
    print("Confusion matrix:\n", cm)
    # Sharpe ratio simulé (buy=1, hold=0, on simule un trade à chaque signal)
    returns = df.loc[X_test.index, "close"].pct_change().fillna(0)
    strat_returns = returns * y_pred
    sharpe = np.mean(strat_returns) / (np.std(strat_returns) + 1e-9) * np.sqrt(252*24*12)  # annualisé (M1)
    print(f"Sharpe ratio simulé (test): {sharpe:.3f}")
    # Sauvegarde
    joblib.dump(clf, model_out)
    joblib.dump(scaler, scaler_out)
    print(f"Modèle sauvegardé sous {model_out}, scaler sous {scaler_out}")
    return clf, scaler


def analyse_label_balance(df: pd.DataFrame, label_col: str = "signal") -> pd.Series:
    """
    Affiche et retourne la répartition des labels dans le DataFrame.
    Utile pour détecter un déséquilibre (dataset de trading).
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        label_col (str): Nom de la colonne label à analyser (par défaut 'signal').
    Returns:
        pd.Series: Répartition des labels (en proportion).
    Sécurité :
        - Ne modifie pas le DataFrame.
        - Gère l'absence de la colonne label.
    """
    if label_col not in df.columns:
        raise ValueError(f"Colonne de label '{label_col}' absente du DataFrame.")
    counts = df[label_col].value_counts(normalize=True)
    print(f"\nRépartition des labels '{label_col}' :")
    print(counts)
    return counts


def main():
    parser = argparse.ArgumentParser(description="Entraînement du modèle ML de trading BTCUSD")
    parser.add_argument("--input_csv", type=str, required=True, help="Chemin du CSV de features")
    parser.add_argument("--model_out", type=str, default="model_rf.pkl", help="Fichier de sortie du modèle")
    parser.add_argument("--scaler_out", type=str, default="scaler.pkl", help="Fichier de sortie du scaler")
    parser.add_argument("--horizon", type=int, default=5, help="Horizon de labellisation (nb bougies)")
    parser.add_argument("--up_thr", type=float, default=0.01, help="Seuil hausse pour label=1")
    parser.add_argument("--down_thr", type=float, default=-0.005, help="Seuil baisse pour label=0")
    parser.add_argument("--use_smote", type=bool, default=False, help="Active SMOTE pour rééquilibrer le train")
    args = parser.parse_args()
    train_ml_model(
        input_csv=args.input_csv,
        model_out=args.model_out,
        scaler_out=args.scaler_out,
        horizon=args.horizon,
        up_thr=args.up_thr,
        down_thr=args.down_thr,
        use_smote=args.use_smote
    )

if __name__ == "__main__":
    main() 