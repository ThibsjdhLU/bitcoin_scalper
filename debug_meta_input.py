#!/usr/bin/env python3
"""
Script de diagnostic pour le problème de confiance faible du MetaModel.
À exécuter depuis la racine du repo : python debug_meta_input.py
"""

import sys
import os
import yaml
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("debug_meta")

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Mock optional dependencies if missing
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    class CatBoostClassifier: pass
    class CatBoostRegressor: pass

try:
    from bitcoin_scalper.connectors.binance_connector import BinanceConnector
    from bitcoin_scalper.core.feature_engineering import FeatureEngineering
    from bitcoin_scalper.models.meta_model import MetaModel
except ImportError as e:
    logger.error(f"Erreur d'import : {e}")
    logger.error("Assurez-vous d'être dans le venv (.venv) et à la racine du repo.")
    sys.exit(1)

# Configuration
CONFIG_PATH = "config/engine_config.yaml"
# Fallback model path if not in config
DEFAULT_MODEL_PATH = "/Users/thibault/bitcoin_scalper/models/meta_model_production.pkl"

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    return {}

def generate_mock_data(n_rows=5000):
    """Generate realistic random walk data ALIGNED to minutes"""
    # Use fixed start time to ensure minute alignment (00 seconds)
    dates = pd.date_range(end=pd.Timestamp("2025-01-01 12:00:00"), periods=n_rows, freq='1min')
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, n_rows)
    price = 40000 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'open': price,
        'high': price * 1.001,
        'low': price * 0.999,
        'close': price,
        'volume': np.random.randint(100, 1000, n_rows).astype(float)
    }, index=dates)
    df.index.name = 'date'
    return df

def main():
    print("="*60)
    print("DIAGNOSTIC BITCOIN SCALPER - META MODEL CONFIDENCE")
    print("="*60)

    # 1. Configuration
    config = load_config()
    api_key = config.get('api_key', '')
    api_secret = config.get('api_secret', '')
    if api_key == "ENV_VAR": api_key = os.environ.get("BINANCE_API_KEY", "")
    if api_secret == "ENV_VAR": api_secret = os.environ.get("BINANCE_API_SECRET", "")

    model_path = config.get('trading', {}).get('model_path', DEFAULT_MODEL_PATH)
    print(f"Modèle cible : {model_path}")

    # 2. Data Fetching
    print("\n[1/6] Récupération des données (5000 bougies BTC/USDT)...")
    try:
        connector = BinanceConnector(api_key=api_key, api_secret=api_secret, testnet=True)
        try:
            df = connector.fetch_ohlcv("BTC/USDT", "1m", limit=5000)
            print(f"Données récupérées : {df.shape}")
        except Exception as e:
            print(f"Erreur fetch (attendu si hors ligne/geo-block): {e}")
            raise
    except Exception as e:
        print(f"Mode hors-ligne. Utilisation de données simulées ALIGNÉES.")
        df = generate_mock_data(5000)
        print(f"Données simulées : {df.shape}")
        print(f"Index sample: {df.index[-3:]}")

    # 3. Feature Engineering
    print("\n[2/6] Application Feature Engineering (simulation Engine)...")

    rename_map = {
        'open': '<OPEN>', 'high': '<HIGH>', 'low': '<LOW>',
        'close': '<CLOSE>', 'volume': '<TICKVOL>'
    }
    df = df.rename(columns=rename_map)

    fe = FeatureEngineering()

    # A. 1-minute
    prefix_1m = "1min_"
    df[f'{prefix_1m}day'] = df.index.dayofweek
    df[f'{prefix_1m}hour'] = df.index.hour
    df[f'{prefix_1m}minute'] = df.index.minute
    df[f'{prefix_1m}hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df[f'{prefix_1m}hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df[f'{prefix_1m}minute_sin'] = np.sin(2 * np.pi * df.index.minute / 60)
    df[f'{prefix_1m}minute_cos'] = np.cos(2 * np.pi * df.index.minute / 60)

    df = fe.add_indicators(df, price_col='<CLOSE>', high_col='<HIGH>', low_col='<LOW>', volume_col='<TICKVOL>', prefix=prefix_1m)
    df = fe.add_features(df, price_col='<CLOSE>', volume_col='<TICKVOL>', prefix=prefix_1m)

    print(f"1min DF shape: {df.shape}")

    # B. 5-minute Resampling
    prefix_5m = "5min_"
    df_5m = df[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']].resample('5min').agg({
        '<OPEN>': 'first', '<HIGH>': 'max', '<LOW>': 'min', '<CLOSE>': 'last', '<TICKVOL>': 'sum'
    }).dropna()

    print(f"5min DF (resampled) shape: {df_5m.shape}")

    if not df_5m.empty:
        df_5m[f'{prefix_5m}day'] = df_5m.index.dayofweek
        df_5m[f'{prefix_5m}hour'] = df_5m.index.hour
        df_5m[f'{prefix_5m}minute'] = df_5m.index.minute
        df_5m[f'{prefix_5m}hour_sin'] = np.sin(2 * np.pi * df_5m.index.hour / 24)
        df_5m[f'{prefix_5m}hour_cos'] = np.cos(2 * np.pi * df_5m.index.hour / 24)
        df_5m[f'{prefix_5m}minute_sin'] = np.sin(2 * np.pi * df_5m.index.minute / 60)
        df_5m[f'{prefix_5m}minute_cos'] = np.cos(2 * np.pi * df_5m.index.minute / 60)

        # USE PATCHED LOGIC IF AVAILABLE
        try:
            df_5m = fe.add_indicators(df_5m, price_col='<CLOSE>', high_col='<HIGH>', low_col='<LOW>', volume_col='<TICKVOL>', prefix=prefix_5m, drop_rows=False)
            print("Utilisation de add_indicators avec drop_rows=False (PATCH ACTIF)")
        except TypeError:
            print("Utilisation de add_indicators standard (PATCH INACTIF)")
            df_5m = fe.add_indicators(df_5m, price_col='<CLOSE>', high_col='<HIGH>', low_col='<LOW>', volume_col='<TICKVOL>', prefix=prefix_5m)

        print(f"5min DF (after indicators) shape: {df_5m.shape}")

        df_5m = fe.add_features(df_5m, price_col='<CLOSE>', volume_col='<TICKVOL>', prefix=prefix_5m)

        cols_5m = [col for col in df_5m.columns if col.startswith(prefix_5m)]
        df_5m_features = df_5m[cols_5m]

        # Join
        df = df.join(df_5m_features, how='left')
        df[cols_5m] = df[cols_5m].ffill()
    else:
        print("⚠️  df_5m est vide après resample !")

    # ANALYSE LAST ROW
    X_tail1 = df.tail(1)

    print(f"\nX.tail(1).columns (Top 10): {X_tail1.columns[:10].tolist()}")

    # Check for NaNs specifically in 5min columns
    cols_5m_in_df = [c for c in df.columns if c.startswith('5min_')]
    nan_5m = X_tail1[cols_5m_in_df].isna().sum()
    nan_5m_cols = nan_5m[nan_5m > 0]

    if not nan_5m_cols.empty:
        print(f"\n⚠️  ALERTE: {len(nan_5m_cols)} colonnes 5min sont NaN dans la dernière ligne !")
        print(nan_5m_cols.head(10).to_dict())
        print("DIAGNOSTIC: Features 5min manquantes/NaN (Cause probable: dropna ou désalignement).")
    else:
        print(f"\n✅ Les colonnes 5min semblent valides (0 NaNs sur {len(cols_5m_in_df)} cols).")

    # 4. Model Prediction
    print("\n[3/6] Chargement du modèle...")
    if not os.path.exists(model_path):
        print(f"ERREUR: Fichier modèle introuvable : {model_path}")
        print("Arrêt du script.")
        sys.exit(0)

    try:
        model = joblib.load(model_path)
        print(f"Modèle chargé: {type(model)}")

        X_input = X_tail1.select_dtypes(include=[np.number])

        print("\n[4/6] Prédiction...")
        if isinstance(model, MetaModel):
            result = model.predict_meta(X_input, return_all=True)
            print(f"Final: {result.get('final_signal')}")
            print(f"Conf:  {result.get('meta_conf')}")
        else:
            try:
                probs = model.predict_proba(X_input)
                print(f"Probs: {probs}")
            except: pass

    except Exception as e:
        print(f"Erreur prédiction: {e}")

if __name__ == "__main__":
    main()
