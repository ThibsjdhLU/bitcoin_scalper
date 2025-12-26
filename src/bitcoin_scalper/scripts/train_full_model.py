#!/usr/bin/env python3
"""
Script d'entraînement complet sur l'historique Binance (2020-Aujourd'hui).
Ce script :
1. Télécharge l'historique complet depuis Binance.
2. Applique le Feature Engineering (incluant le fix 5min).
3. Génère les labels (Triple Barrier Method).
4. Entraîne le Modèle Primaire (Direction) et le Meta-Modèle (Confiance).
5. Sauvegarde le modèle prêt pour la production.

Usage:
    python src/bitcoin_scalper/scripts/train_full_model.py --symbol BTC/USDT --start 2020-01-01
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Ajout du chemin src
sys.path.append(os.path.join(os.getcwd(), 'src'))

from bitcoin_scalper.connectors.binance_connector import BinanceConnector
from bitcoin_scalper.core.feature_engineering import FeatureEngineering
from bitcoin_scalper.core.modeling import ModelTrainer
from bitcoin_scalper.models.meta_model import MetaModel
from bitcoin_scalper.core.engine import TradingEngine # Pour les utilitaires

# Configuration Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger("train_full")

def fetch_full_history(symbol: str, start_date: str, api_key: str, api_secret: str, cache_file: str = "data/full_history.csv", testnet: bool = False) -> pd.DataFrame:
    """
    Télécharge l'historique complet depuis Binance ou charge depuis le cache.
    """
    # Créer le dossier data si inexistant
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # 1. Essayer de charger le cache
    if os.path.exists(cache_file):
        logger.info(f"Chargement des données depuis le cache : {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        logger.info(f"Données chargées : {len(df)} lignes. Fin : {df.index[-1]}")

        # Optionnel : Fetcher uniquement le delta manquant ?
        # Pour simplifier, on assume que si le cache existe, l'utilisateur le gère.
        # Ou on peut forcer le retéléchargement si demandé.
        return df

    logger.info(f"Démarrage du téléchargement complet pour {symbol} depuis {start_date}...")

    # Init Connector
    connector = BinanceConnector(api_key=api_key, api_secret=api_secret, testnet=testnet)

    # Calcul du start timestamp
    since_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    limit = 1000
    all_candles = []

    # Boucle de pagination
    # Binance permet de fetcher depuis un 'startTime'.
    current_start = since_ts

    while True:
        try:
            logger.info(f"Fetching depuis {pd.to_datetime(current_start, unit='ms')}...")
            # On utilise l'attribut exchange de ccxt directement pour plus de contrôle sur 'since'
            ohlcv = connector.exchange.fetch_ohlcv(symbol, timeframe='1m', since=current_start, limit=limit)

            if not ohlcv:
                logger.info("Plus de données disponibles.")
                break

            # Check for data gap on first fetch
            if len(all_candles) == 0:
                first_candle_ts = ohlcv[0][0]
                # Compare against the initial requested start timestamp (since_ts)
                if first_candle_ts > since_ts + 86400000: # > 1 day gap
                    diff_days = (first_candle_ts - since_ts) / 86400000
                    logger.warning(f"⚠️ Gap de données détecté : {diff_days:.1f} jours manquants au début.")
                    logger.warning(f"   Demandé : {pd.to_datetime(since_ts, unit='ms')}")
                    logger.warning(f"   Disponible : {pd.to_datetime(first_candle_ts, unit='ms')}")
                    logger.warning("   (Ceci est normal sur le Testnet qui a un historique limité)")

            all_candles.extend(ohlcv)
            last_ts = ohlcv[-1][0]

            # Mise à jour du curseur : start = last_candle_time + 1 minute
            # Attention : fetch_ohlcv inclut la bougie du 'since'. Donc on doit avancer.
            # Mais ccxt gère souvent ça. Vérifions si on avance.
            if last_ts == current_start:
                logger.warning("Boucle infinie détectée (timestamp n'avance pas). Arrêt.")
                break

            current_start = last_ts + 60000 # +1 minute

            # Pause pour rate limit (géré par ccxt enableRateLimit, mais soyons prudents)
            # import time; time.sleep(0.1)

            # Arrêt si on atteint le présent (ou presque)
            if last_ts > (datetime.now().timestamp() * 1000 - 60000):
                logger.info("Arrivé au présent.")
                break

        except Exception as e:
            logger.error(f"Erreur pendant le fetch : {e}")
            break

    logger.info(f"Téléchargement terminé : {len(all_candles)} bougies.")

    # Conversion DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('date')
    df = df.drop(columns=['timestamp'])

    # Sauvegarde Cache
    df.to_csv(cache_file)
    logger.info(f"Données sauvegardées dans {cache_file}")

    return df

def train_pipeline(df: pd.DataFrame, model_output_path: str):
    """
    Exécute le pipeline ML complet : FE -> Labeling -> Train -> Save
    """
    # 1. Feature Engineering (Multi-Timeframe 1m + 5m)
    logger.info("--- Étape 1 : Feature Engineering ---")

    # Renommage colonnes (Engine convention)
    rename_map = {'open': '<OPEN>', 'high': '<HIGH>', 'low': '<LOW>', 'close': '<CLOSE>', 'volume': '<TICKVOL>'}
    df = df.rename(columns=rename_map)

    fe = FeatureEngineering()

    # A. 1-minute
    logger.info("Calcul features 1-minute...")
    prefix_1m = "1min_"
    df[f'{prefix_1m}day'] = df.index.dayofweek
    df[f'{prefix_1m}hour'] = df.index.hour
    df = fe.add_indicators(df, price_col='<CLOSE>', high_col='<HIGH>', low_col='<LOW>', volume_col='<TICKVOL>', prefix=prefix_1m)
    df = fe.add_features(df, price_col='<CLOSE>', volume_col='<TICKVOL>', prefix=prefix_1m)

    # B. 5-minute (Resampling + Fix)
    logger.info("Calcul features 5-minute (avec fix dropna)...")
    prefix_5m = "5min_"
    df_5m = df[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']].resample('5min').agg({
        '<OPEN>': 'first', '<HIGH>': 'max', '<LOW>': 'min', '<CLOSE>': 'last', '<TICKVOL>': 'sum'
    }).dropna()

    # Feature Engineering sur 5m avec drop_rows=False (important pour la cohérence, même si ici on a tout l'historique)
    df_5m = fe.add_indicators(
        df_5m, price_col='<CLOSE>', high_col='<HIGH>', low_col='<LOW>', volume_col='<TICKVOL>', prefix=prefix_5m,
        drop_rows=False # Utilisation explicite du nouveau paramètre
    )
    df_5m = fe.add_features(df_5m, price_col='<CLOSE>', volume_col='<TICKVOL>', prefix=prefix_5m)

    # Merge
    logger.info("Fusion 1m + 5m...")
    cols_5m = [col for col in df_5m.columns if col.startswith(prefix_5m)]
    df = df.join(df_5m[cols_5m], how='left')
    df[cols_5m] = df[cols_5m].ffill()

    # Nettoyage final des NaNs (Warmup global)
    # On drop le début (ex: 500 premières lignes) pour être propre
    df = df.dropna()
    logger.info(f"Dataset prêt pour entraînement : {df.shape}")

    # 2. Labeling (Triple Barrier)
    logger.info("--- Étape 2 : Labeling (Triple Barrier) ---")
    # Configuration Triple Barrier
    # Volatilité dynamique
    volatility = df['1min_atr_14'] # Utilise l'ATR 1min calculé
    # Horizon : 15 bougies (15 minutes)
    t_events = df.index

    # On utilise une version simplifiée ou la classe TripleBarrierLabeling si dispo
    # Ici, on va coder une logique simple et robuste pour la démo
    # Cible : +1 si prix touche haut, -1 si bas, 0 si timeout
    # PT (Profit Taking) = 1.5 * Volatilité, SL (Stop Loss) = 1.5 * Volatilité

    horizon = 30 # minutes

    # Calcul simple vectorisé pour aller vite sur 2M de lignes
    # Label Primaire (Direction) : Prochain retour > seuil ?
    # On va utiliser le log_return futur
    future_return = df['<CLOSE>'].shift(-horizon) / df['<CLOSE>'] - 1

    # Seuils basés sur la volatilité
    threshold = df['1min_atr_14'] / df['<CLOSE>'] # ATR en %
    threshold = threshold.rolling(100).mean() # Lissage

    # Labels
    y_primary = pd.Series(0, index=df.index)
    y_primary[future_return > threshold] = 1  # Buy
    y_primary[future_return < -threshold] = -1 # Sell

    # Meta Label (Succès du trade théorique)
    # Si Primaire dit Buy (1) et Return > 0 -> Succès (1)
    # Si Primaire dit Sell (-1) et Return < 0 -> Succès (1)
    # Sinon -> Echec (0)
    y_meta = pd.Series(0, index=df.index)
    idx_buy = (y_primary == 1) & (future_return > 0) # On peut être plus strict (future_return > threshold)
    idx_sell = (y_primary == -1) & (future_return < 0)
    y_meta[idx_buy | idx_sell] = 1

    logger.info(f"Distribution Primaire : {y_primary.value_counts().to_dict()}")
    logger.info(f"Distribution Meta : {y_meta.value_counts().to_dict()}")

    # 3. Training
    logger.info("--- Étape 3 : Entraînement ---")

    # Split Train/Val (Chronologique, pas aléatoire !)
    split_idx = int(len(df) * 0.8)
    X = df.drop(columns=['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>'] + [c for c in df.columns if 'return' in c and 'lag' not in c])
    # On garde seulement les features, on vire les prix bruts et les returns futurs (fuite)
    # Attention : '1min_return' est un shift(1), donc feature valide. 'future_return' n'est pas dans df.

    # Sécurisation features : ne garder que les numériques
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()
    logger.info(f"Features sélectionnées ({len(feature_names)}) : {feature_names[:5]} ...")

    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_p_train, y_p_val = y_primary.iloc[:split_idx], y_primary.iloc[split_idx:]
    y_m_train, y_m_val = y_meta.iloc[:split_idx], y_meta.iloc[split_idx:]

    # Modèle Primaire (CatBoost)
    logger.info("Entraînement Modèle Primaire (Direction)...")
    trainer_primary = ModelTrainer(algo='catboost', use_scaler=True)
    model_primary = trainer_primary.fit(X_train, y_p_train, X_val, y_p_val, n_trials=10) # 10 essais Optuna

    # Modèle Meta (CatBoost)
    # Le MetaModel prend en entrée X + Proba du Primaire
    # On utilise la classe MetaModel du projet qui gère ça
    logger.info("Entraînement Meta Modèle (Confiance)...")

    # On doit wrapper les modèles entraînés ou laisser MetaModel gérer ?
    # MetaModel.train entraîne tout.
    # On va instancier MetaModel avec des objets CatBoostClassifier vierges mais configurés
    from catboost import CatBoostClassifier

    # On récupère les meilleurs params du trainer si possible, ou on laisse MetaModel faire son fit standard
    # Pour ce script, on va faire simple : on utilise l'objet MetaModel qui orchestre tout

    primary_base = CatBoostClassifier(iterations=500, depth=6, verbose=0, allow_writing_files=False)
    meta_base = CatBoostClassifier(iterations=500, depth=4, verbose=0, allow_writing_files=False)

    final_meta_model = MetaModel(primary_base, meta_base, meta_threshold=0.55)

    # Entraînement conjoint
    final_meta_model.train(X_train, y_p_train, y_m_train, eval_set=(X_val, y_p_val, y_m_val))

    # 4. Sauvegarde
    logger.info(f"--- Étape 4 : Sauvegarde vers {model_output_path} ---")
    import joblib
    joblib.dump(final_meta_model, model_output_path)
    logger.info("✅ Modèle sauvegardé avec succès.")

def main():
    parser = argparse.ArgumentParser(description="Train Full Bitcoin Scalper Model")
    parser.add_argument("--api_key", type=str, required=False, help="Binance API Key", default=os.environ.get("BINANCE_API_KEY"))
    parser.add_argument("--api_secret", type=str, required=False, help="Binance Secret", default=os.environ.get("BINANCE_API_SECRET"))
    parser.add_argument("--start", type=str, default="2020-01-01", help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--out", type=str, default="models/meta_model_full_history.pkl", help="Chemin de sortie du modèle")
    parser.add_argument("--testnet", action="store_true", help="Utiliser le Testnet Binance au lieu du Mainnet")

    args = parser.parse_args()

    if not args.api_key or not args.api_secret:
        logger.error("API Key/Secret manquants. Définissez BINANCE_API_KEY/SECRET ou passez les en arguments.")
        return

    # 1. Fetch
    df = fetch_full_history("BTC/USDT", args.start, args.api_key, args.api_secret, testnet=args.testnet)

    # 2. Train
    train_pipeline(df, args.out)

if __name__ == "__main__":
    main()
