#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler

# Ajouter le chemin du répertoire racine du projet pour pouvoir importer les modules locaux
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bitcoin_scalper.core.feature_engineering import FeatureEngineering

# --- Configuration ---
INPUT_CSV_PATH = "data/features/BTCUSD_M1_201812120809_202505271849.csv" # User provided path, ensure it's accessible or make it a placeholder if not.
OUTPUT_CSV_PATH = "data/features/BTCUSD_M1_features_trend_following.csv" # New output file name

def check_temporal_integrity(df: pd.DataFrame, indicator_cols=None) -> bool:
    """
    Vérifie l'absence de look-ahead bias : chaque indicateur doit être décalé d'une bougie.
    Retourne True si aucune erreur, False sinon.
    """
    if indicator_cols is None:
        indicator_cols = [
            'ema_21', 'ema_50', 'rsi', 'atr', 'supertrend',
            'close_sma_3', 'atr_sma_20'
        ]
    errors = 0
    for col in indicator_cols:
        if col in df.columns:
            shifted = df[col].shift(1)
            # On ignore la première ligne (toujours NaN après shift)
            if not (df[col].iloc[1:] == shifted.iloc[1:]).all():
                errors += 1
                print(f"[ERREUR] Look-ahead détecté sur la colonne {col}")
    if errors == 0:
        print("[OK] Intégrité temporelle validée (aucun look-ahead bias détecté)")
        return True
    else:
        print(f"[ERREUR] {errors} colonnes présentent un look-ahead bias !")
        return False

def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère la colonne 'signal' (1=long, -1=short, 0=neutre) sur la base des indicateurs décalés.
    Stratégie très permissive : seulement EMA et RSI, seuils larges, pas de supertrend ni ATR.
    """
    df = df.copy()
    df['rsi_mean'] = df['rsi'].rolling(100, min_periods=10).mean()
    long_condition = (
        (df['ema_21'].fillna(0) > df['ema_50'].fillna(0)) &
        (df['rsi'].fillna(0) > df['rsi_mean'].fillna(0) * 0.95)
    )
    short_condition = (
        (df['ema_21'].fillna(0) < df['ema_50'].fillna(0)) &
        (df['rsi'].fillna(0) < df['rsi_mean'].fillna(0) * 1.05)
    )
    df['signal'] = 0
    df.loc[long_condition, 'signal'] = 1
    df.loc[short_condition, 'signal'] = -1
    df['signal_binary'] = (df['signal'] == 1).astype(int)
    print("[DEBUG] Distribution des signaux (signal) :\n", df['signal'].value_counts(normalize=True))
    return df

def prepare_dataset(input_path: str, output_path: str):
    """
    Pipeline complet :
    1. Chargement brut
    2. Ajout indicateurs (décalés)
    3. Calcul features dérivées (décalées)
    4. Génération du signal
    5. Gestion NaN ciblée
    6. Validation intégrité temporelle
    7. Export CSV sécurisé
    8. Normalisation + pondération des classes
    """
    print(f"Chargement des données brutes depuis {input_path}...")
    try:
        df = pd.read_csv(input_path, sep='\t', engine='python')
    except FileNotFoundError:
        print(f"Erreur : Fichier non trouvé à {input_path}")
        return
    except Exception as e:
        print(f"Erreur lors du chargement du fichier : {e}")
        return

    print("Préparation des données...")
    df['timestamp'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'], errors='coerce')
    df = df.rename(columns={
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>': 'low',
        '<CLOSE>': 'close',
        '<TICKVOL>': 'tickvol',
        '<VOL>': 'volume_zero',
        '<SPREAD>': 'spread'
    })
    df.dropna(subset=['timestamp'], inplace=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    if 'volume_zero' in df.columns:
        df = df.drop(columns=['volume_zero'])

    print("Ajout des indicateurs techniques et features...")
    fe = FeatureEngineering()
    df = fe.add_indicators(df, price_col='close', high_col='high', low_col='low', volume_col='tickvol')
    # Calcul des features dérivées avec shift(1) après le rolling
    df['close_sma_3'] = df['close'].rolling(window=3, min_periods=1).mean().shift(1)
    df['atr_sma_20'] = df['atr'].rolling(window=20, min_periods=1).mean().shift(1)

    # Génération du signal robuste (stratégie assouplie)
    df = generate_signal(df)

    # Gestion NaN ciblée (seulement sur les colonnes critiques)
    indicator_cols = [
        'ema_21', 'ema_50', 'rsi', 'atr', 'supertrend',
        'close_sma_3', 'atr_sma_20', 'signal', 'signal_binary'
    ]
    initial_count = len(df)
    df.dropna(subset=indicator_cols, inplace=True)
    print(f"Suppression de {initial_count - len(df)}/{initial_count} lignes ({(initial_count - len(df)) / initial_count * 100:.2f}%)")

    # Validation intégrité temporelle
    check_temporal_integrity(df, indicator_cols=indicator_cols)

    # Statistiques sur la cible
    print("Distribution des signaux (signal) :\n", df['signal'].value_counts(normalize=True))
    print("Distribution binaire (signal_binary) :\n", df['signal_binary'].value_counts(normalize=True))
    prop_long = df['signal_binary'].mean()
    if prop_long < 0.05:
        print("[AVERTISSEMENT] Proportion de signaux LONG très faible (<5%). Le modèle risque d'être biaisé.")

    # Pondération des classes pour le ML (inverse de la fréquence)
    class_counts = df['signal_binary'].value_counts()
    total = len(df)
    weights = {k: total/v for k, v in class_counts.items()}
    df['weight'] = df['signal_binary'].map(weights)

    # Normalisation des features numériques (hors cible, hors timestamp)
    features_to_normalize = [
        'open', 'high', 'low', 'close', 'tickvol',
        'ema_21', 'ema_50', 'rsi', 'atr', 'supertrend',
        'close_sma_3', 'atr_sma_20'
    ]
    features_to_normalize = [col for col in features_to_normalize if col in df.columns]
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
    # Export version normalisée et brute
    df['timestamp_str'] = df.index.astype(str)
    df_norm['timestamp_str'] = df.index.astype(str)
    cols_utiles = [
        'open', 'high', 'low', 'close', 'tickvol',
        'ema_21', 'ema_50', 'rsi', 'atr', 'supertrend',
        'close_sma_3', 'atr_sma_20', 'signal', 'signal_binary', 'weight', 'timestamp_str'
    ]
    cols_utiles = [col for col in cols_utiles if col in df.columns]
    df_to_export = df[cols_utiles].copy()
    df_norm_to_export = df_norm[cols_utiles].copy()
    df_to_export.to_csv(output_path, index=False)
    df_norm_to_export.to_csv(output_path.replace('.csv', '_norm.csv'), index=False)
    print("Préparation terminée.")
    print(f"Dataset filtré enregistré sous {output_path} (brut) et {output_path.replace('.csv', '_norm.csv')} (normalisé).")
    print(f"Shape du dataset final : {df_to_export.shape}")
    print("Aperçu des premières lignes :")
    print(df_to_export.head())

def prepare_ml_csv(input_path: str, output_path: str):
    """
    Prépare un CSV compatible ML à partir d'un CSV brut issu de MetaTrader ou autre.
    - Renomme les colonnes pour qu'elles soient compatibles avec XGBoost/sklearn
    - Ajoute les colonnes de features techniques manquantes (valeurs NaN ou calculées)
    - Sauvegarde le CSV prêt pour l'entraînement ML
    """
    # Mapping des noms bruts vers noms ML
    col_map = {
        '<DATE>': 'date',
        '<TIME>': 'time',
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>': 'low',
        '<CLOSE>': 'close',
        '<TICKVOL>': 'tickvol',
        '<VOL>': 'vol',
        '<SPREAD>': 'spread',
    }
    df = pd.read_csv(input_path, sep='\t')
    df = df.rename(columns=col_map)

    # Colonnes attendues pour le ML (voir scripts/prepare_features.py et core/ml_train.py)
    cols_utiles = [
        'open', 'high', 'low', 'close', 'tickvol',
        'ema_21', 'ema_50', 'rsi', 'atr', 'supertrend',
        'close_sma_3', 'atr_sma_20', 'signal', 'timestamp'
    ]
    # Ajoute les colonnes manquantes avec NaN
    for col in cols_utiles:
        if col not in df.columns:
            df[col] = np.nan
    # Optionnel : générer les features techniques ici si besoin
    # fe = FeatureEngineering()
    # df = fe.transform(df)
    # Ajoute une colonne timestamp si absente
    if 'timestamp' not in df.columns:
        df['timestamp'] = df['date'].astype(str) + ' ' + df['time'].astype(str)
    # Ajoute une colonne signal par défaut (0)
    if 'signal' not in df.columns:
        df['signal'] = 0
    # Réordonne les colonnes
    df = df[cols_utiles]
    df.to_csv(output_path, index=False)
    print(f"Nouveau CSV ML prêt : {output_path}")

if __name__ == "__main__":
    # Ensure the input file exists or adjust path.
    # The user provided "data/features/BTCUSD_M1_201812120809_202505271849.csv"
    # This path implies the script is run from the project root.
    # If INPUT_CSV_PATH is not found, the script will print an error and exit.
    prepare_dataset(INPUT_CSV_PATH, OUTPUT_CSV_PATH)
