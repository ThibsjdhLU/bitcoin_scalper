import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import json
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger("bitcoin_scalper.labeling")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

def generate_labels(
    df: pd.DataFrame,
    horizon: int = 15,
    k: float = 0.5,
    threshold_type: str = "std",
    quantile_high: float = 0.8,
    quantile_low: float = 0.2,
    spread_fee: float = 0.0,
    fee: float = 0.0,
    slippage: float = 0.0,
    spread: float = 0.0,
    n_classes: int = 3,
    neutral_policy: str = "keep",  # "keep", "nan", "drop"
    return_confidence: bool = False,
    features_for_confidence: list = None,
    knn_k: int = 10
) -> pd.Series:
    """
    Génère une colonne de labels à 3 ou 5 classes pour classification supervisée sur données financières minute.
    Le label est basé sur le mouvement futur du prix à un horizon donné, avec un seuil dynamique ou un gain net :
    - "std" : seuil = k * rolling std (par défaut)
    - "quantile" : seuils = quantiles des future_return (ex : 80e/20e percentile)
    - "spread_fee" : seuil = spread+frais (valeur absolue)
    - "actionnable" : label = 1 si gain net (après spread, fee, slippage) > 0, -1 si < 0, 0 sinon
    - n_classes=5 : labels asymétriques (+2, +1, 0, -1, -2) selon seuils dynamiques ou quantiles
    - neutral_policy : "keep" (par défaut), "nan" (remplace 0 par NaN), "drop" (supprime les 0)
    - return_confidence : si True, retourne aussi une série de confiance basée sur la densité kNN
    """
    # Recherche robuste des colonnes de prix et de log_return
    close_candidates = ['<CLOSE>', '1min_<CLOSE>', 'close', '1min_close']
    log_return_candidates = ['log_return_1m', '1min_log_return', 'log_return', '1min_log_return']
    close_col = next((col for col in close_candidates if col in df.columns), None)
    log_return_col = next((col for col in log_return_candidates if col in df.columns), None)
    if close_col is None or log_return_col is None:
        logger.error(f"Colonne de prix ou de log_return absente du DataFrame. Colonnes candidates prix : {close_candidates}, log_return : {log_return_candidates}. Colonnes présentes : {list(df.columns)}")
        raise ValueError(f"Colonne de prix ou de log_return absente du DataFrame. Colonnes candidates prix : {close_candidates}, log_return : {log_return_candidates}. Colonnes présentes : {list(df.columns)}")

    logger.info(f"Génération des labels à {n_classes} classes (horizon={horizon}, k={k}, threshold_type={threshold_type}) avec close_col={close_col}, log_return_col={log_return_col}")
    # Calcul du future_return (log(CLOSE_{t+h} / CLOSE_t))
    future_close = df[close_col].shift(-horizon)
    future_return = np.log(future_close / df[close_col])
    label = pd.Series(np.nan, index=df.index)
    if n_classes == 3:
        if threshold_type == "std":
            rolling_std = df[log_return_col].rolling(window=30, min_periods=30).std()
            seuil = k * rolling_std
            seuil_high = seuil
            seuil_low = -seuil
        elif threshold_type == "quantile":
            seuil_high = future_return.rolling(window=30, min_periods=30).quantile(quantile_high)
            seuil_low = future_return.rolling(window=30, min_periods=30).quantile(quantile_low)
        elif threshold_type == "spread_fee":
            seuil_high = pd.Series(spread_fee, index=df.index)
            seuil_low = pd.Series(-spread_fee, index=df.index)
        elif threshold_type == "actionnable":
            total_cost = 2 * (fee + slippage + spread)
            net_return = future_return - total_cost
            label[net_return > 0] = 1
            label[net_return < 0] = -1
            label[net_return == 0] = 0
            valid = ~net_return.isna()
            label = label[valid]
            logger.info(f"Distribution des labels : {label.value_counts(dropna=True).to_dict()}")
            label = label.astype(int)
            # Gestion des neutres
            if neutral_policy == "nan":
                label[label == 0] = np.nan
            elif neutral_policy == "drop":
                label = label[label != 0]
            if return_confidence:
                confidence = compute_label_confidence_knn(df, features_for_confidence, knn_k)
                confidence = confidence.loc[label.index]
                return label, confidence
            return label
        else:
            logger.error(f"Type de seuil inconnu : {threshold_type}")
            raise ValueError(f"Type de seuil inconnu : {threshold_type}")
        label[future_return > seuil_high] = 1
        label[future_return < seuil_low] = -1
        label[(future_return <= seuil_high) & (future_return >= seuil_low)] = 0
        valid = (~pd.isna(seuil_high)) & (~pd.isna(seuil_low)) & (~future_return.isna())
    elif n_classes == 5:
        # Labels asymétriques : +2, +1, 0, -1, -2
        if threshold_type == "std":
            rolling_std = df[log_return_col].rolling(window=30, min_periods=30).std()
            seuil1 = k * rolling_std
            seuil2 = 2 * k * rolling_std
            seuils = [seuil2, seuil1, -seuil1, -seuil2]
        elif threshold_type == "quantile":
            seuil2 = future_return.rolling(window=30, min_periods=30).quantile(quantile_high)
            seuil1 = future_return.rolling(window=30, min_periods=30).quantile((quantile_high + 0.5) / 2)
            seuilm2 = future_return.rolling(window=30, min_periods=30).quantile(quantile_low)
            seuilm1 = future_return.rolling(window=30, min_periods=30).quantile((quantile_low + 0.5) / 2)
            seuils = [seuil2, seuil1, seuilm1, seuilm2]
        else:
            logger.error(f"Type de seuil inconnu pour n_classes=5 : {threshold_type}")
            raise ValueError(f"Type de seuil inconnu pour n_classes=5 : {threshold_type}")
        label[future_return >= seuils[0]] = 2
        label[(future_return < seuils[0]) & (future_return >= seuils[1])] = 1
        label[(future_return < seuils[1]) & (future_return > seuils[2])] = 0
        label[(future_return <= seuils[2]) & (future_return > seuils[3])] = -1
        label[future_return <= seuils[3]] = -2
        valid = (~pd.isna(seuils[0])) & (~pd.isna(seuils[1])) & (~pd.isna(seuils[2])) & (~pd.isna(seuils[3])) & (~future_return.isna())
    else:
        logger.error(f"n_classes doit être 3 ou 5")
        raise ValueError(f"n_classes doit être 3 ou 5")
    label = label[valid]
    logger.info(f"Distribution des labels : {label.value_counts(dropna=True).to_dict()}")
    label = label.astype(int)
    # Gestion des neutres
    if neutral_policy == "nan":
        label[label == 0] = np.nan
    elif neutral_policy == "drop":
        label = label[label != 0]
    if return_confidence:
        confidence = compute_label_confidence_knn(df, features_for_confidence, knn_k)
        confidence = confidence.loc[label.index]
        return label, confidence
    return label

def generate_multi_horizon_labels(
    df: pd.DataFrame,
    horizons: list = [5, 10, 15, 30],
    k: float = 0.5,
    threshold_type: str = "std",
    quantile_high: float = 0.8,
    quantile_low: float = 0.2,
    spread_fee: float = 0.0,
    fee: float = 0.0,
    slippage: float = 0.0,
    spread: float = 0.0
) -> pd.DataFrame:
    """
    Génère des labels multi-horizon (target_5m, target_10m, etc.) et les ajoute au DataFrame.
    :param df: DataFrame enrichi, indexé par datetime UTC, contenant <CLOSE> et log_return_1m
    :param horizons: Liste des horizons de prédiction (en minutes)
    :param k: Multiplicateur du rolling_std pour le seuil dynamique
    :param threshold_type: "std", "quantile", "spread_fee", "actionnable"
    :param quantile_high: Quantile haut pour le seuil (si quantile)
    :param quantile_low: Quantile bas pour le seuil (si quantile)
    :param spread_fee: Seuil absolu (si spread_fee)
    :param fee: Frais de transaction (en proportion, ex : 0.001)
    :param slippage: Slippage (en proportion, ex : 0.0005)
    :param spread: Spread (en proportion, ex : 0.0002)
    :return: DataFrame avec les colonnes target_{horizon}m ajoutées
    """
    df = df.copy()
    for h in horizons:
        labels = generate_labels(
            df, horizon=h, k=k, threshold_type=threshold_type,
            quantile_high=quantile_high, quantile_low=quantile_low, spread_fee=spread_fee,
            fee=fee, slippage=slippage, spread=spread
        )
        col = f"target_{h}m"
        df[col] = labels.reindex(df.index)
    return df

def analyze_label_distribution(
    df: pd.DataFrame,
    label_cols: list,
    out_dir: str = "data/features",
    prefix: str = ""
) -> dict:
    """
    Analyse la distribution des labels (comptage, pourcentage) pour une ou plusieurs colonnes.
    Génère un rapport PNG (bar plot) et un rapport texte/JSON (statistiques).
    :param df: DataFrame contenant les labels
    :param label_cols: Liste des colonnes de labels à analyser
    :param out_dir: Dossier de sortie pour les rapports
    :param prefix: Préfixe pour les fichiers exportés
    :return: Dictionnaire des distributions
    """
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    for col in label_cols:
        if col not in df.columns:
            logger.warning(f"Colonne {col} absente du DataFrame pour l'analyse de distribution.")
            continue
        counts = df[col].value_counts(dropna=True).sort_index()
        percents = counts / counts.sum() * 100
        stats[col] = {"counts": counts.to_dict(), "percents": percents.round(2).to_dict()}
        # Bar plot
        plt.figure()
        counts.plot(kind="bar", color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        plt.title(f"Distribution des labels : {col}")
        plt.xlabel("Classe")
        plt.ylabel("Nombre d'occurrences")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{prefix}{col}_distribution.png")
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Distribution PNG exportée : {out_path}")
        # Alerte si >90% d'une classe
        if percents.max() > 90:
            logger.warning(f"Distribution très déséquilibrée pour {col} : {percents.idxmax()} = {percents.max()}%")
    # Export JSON
    json_path = os.path.join(out_dir, f"{prefix}labels_distribution.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Distribution des labels exportée en JSON : {json_path}")
    return stats

def compute_label_confidence_knn(df: pd.DataFrame, features: list = None, k: int = 10) -> pd.Series:
    """
    Calcule une mesure de confiance locale pour chaque ligne du DataFrame basée sur la densité kNN dans l'espace des features.
    Plus la densité locale est élevée, plus la confiance est forte (moins d'incertitude).
    :param df: DataFrame d'entrée (features déjà calculées)
    :param features: Liste des colonnes à utiliser (par défaut toutes les colonnes numériques hors labels)
    :param k: Nombre de voisins à utiliser
    :return: pd.Series de confiance (score entre 0 et 1, aligné sur df.index)
    """
    if features is None:
        features = [col for col in df.columns if df[col].dtype in [np.float32, np.float64, float, int] and not col.startswith('label')]
    X = df[features].fillna(0).values
    if len(X) < k+1:
        return pd.Series(np.nan, index=df.index)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    # On ignore la distance à soi-même (0)
    mean_dist = distances[:, 1:].mean(axis=1)
    # Score de confiance = 1 - normalisé (plus la distance moyenne est faible, plus la densité est forte)
    norm = (mean_dist - mean_dist.min()) / (mean_dist.max() - mean_dist.min() + 1e-9)
    confidence = 1 - norm
    return pd.Series(confidence, index=df.index)

def generate_q_values(
    df: pd.DataFrame,
    horizon: int = 15,
    fee: float = 0.0,
    spread: float = 0.0,
    slippage: float = 0.0
) -> pd.DataFrame:
    """
    Génère les Q-values (expected return net) pour chaque action (hold, buy, sell) à partir d'un DataFrame OHLCV.
    Q_buy = log(CLOSE_{t+h}/CLOSE_t) - frais - spread - slippage
    Q_sell = log(CLOSE_t/CLOSE_{t+h}) - frais - spread - slippage
    Q_hold = 0
    :param df: DataFrame avec <CLOSE> (et index temporel)
    :param horizon: horizon de calcul (en minutes)
    :param fee: frais de transaction (proportion)
    :param spread: spread (proportion)
    :param slippage: slippage (proportion)
    :return: DataFrame avec colonnes q_buy, q_sell, q_hold
    """
    if '<CLOSE>' not in df.columns:
        raise ValueError("Colonne <CLOSE> absente du DataFrame.")
    future_close = df['<CLOSE>'].shift(-horizon)
    log_return = np.log(future_close / df['<CLOSE>'])
    q_buy = log_return - 2 * (fee + spread + slippage)
    q_sell = -log_return - 2 * (fee + spread + slippage)
    q_hold = pd.Series(0, index=df.index)
    q_df = pd.DataFrame({
        'q_buy': q_buy,
        'q_sell': q_sell,
        'q_hold': q_hold
    })
    return q_df 