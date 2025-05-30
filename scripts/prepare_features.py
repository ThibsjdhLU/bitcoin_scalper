#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys

# Ajouter le chemin du répertoire racine du projet pour pouvoir importer les modules locaux
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bitcoin_scalper.core.feature_engineering import FeatureEngineering

# --- Configuration ---
INPUT_CSV_PATH = "data/features/BTCUSD_M1_201812120809_202505271849.csv" # User provided path, ensure it's accessible or make it a placeholder if not.
OUTPUT_CSV_PATH = "data/features/BTCUSD_M1_features_trend_following.csv" # New output file name

# --- Pipeline de Préparation des Features et du Label ---
def prepare_dataset(input_path: str, output_path: str):
    """
    Charge les données brutes, ajoute les features et indicateurs, génère le label 'signal',
    et exporte le dataset complet dans un nouveau fichier CSV.
    """
    print(f"Chargement des données brutes depuis {input_path}...")
    try:
        # 1. Chargement du CSV brut
        # Changed separator to comma to match dummy data creation and subtask description for dummy CSV.
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
    # --- OPTIMIZATION PLACEHOLDER ---
    # The parameters for the indicators used below are candidates for future optimization.
    # Techniques like Grid Search, Random Search, or Bayesian Optimization can be applied
    # within a backtesting framework to find optimal values.
    # Key parameters to consider for optimization:
    #   - EMA fast period (currently 21, e.g., range 10-30)
    #   - EMA slow period (currently 50, e.g., range 40-100)
    #   - RSI period (currently 14, e.g., range 7-21)
    #   - Supertrend period (currently 7, e.g., range 5-15)
    #   - Supertrend multiplier (currently 3.0, e.g., range 1.5-4.0)
    #   - ATR period for ATR threshold calculation (currently 14 for ATR, 20 for its SMA, e.g., range 7-30 for both)
    #   - The logic for 'atr > atr_sma_20' itself could be optimized (e.g., different SMA period, or ATR > fixed_value * ATR.rolling(n).std())
    fe = FeatureEngineering()
    # Use the correct column names as per the CSV structure for volume
    df = fe.add_indicators(df, price_col='close', high_col='high', low_col='low', volume_col='tickvol')
    # df = fe.add_features(df, price_col='close', volume_col='tickvol') # Keep or remove based on strategy needs

    # --- START OF MODIFICATIONS FOR NEW STRATEGY ---

    # 1. Ensure all required indicators are present (from FeatureEngineering class)
    # EMA 21, EMA 50, RSI 14, ATR 14, Supertrend (7,3)
    # These should be named: 'ema_21', 'ema_50', 'rsi', 'atr', 'supertrend' (or 'SUPERT_7_3.0' if that's the actual name from pandas-ta, then rename it to 'supertrend')
    # If 'supertrend' is not the direct output name from pandas-ta, add a line to rename it:
    # For example: if df.ta.supertrend outputs 'SUPERT_7_3.0', 'SUPERTd_7_3.0', etc.
    # We need the main trend line. Let's assume it's 'SUPERT_7_3.0'.
    # df['supertrend'] = df['SUPERT_7_3.0'] # Add this if 'supertrend' is not the direct name
    # The worker reported assigning the main supertrend line to df[f"{prefix}supertrend"], so it should be 'supertrend' already.

    # 2. Calculate additional required features
    df['close_sma_3'] = df['close'].rolling(window=3).mean()
    # Placeholder for ATR threshold - for now, let's use ATR > its rolling mean or a fixed small value.
    # This needs to be defined more concretely or optimized later.
    df['atr_sma_20'] = df['atr'].rolling(window=20).mean() # Example: ATR compared to its 20-period SMA

    # 3. Remove or comment out the "Phoenix" strategy logic for 'signal' generation.
    # This includes all 'buy_condition_*' and 'sell_condition_*' for Phoenix.

    # 4. Implement new "Trend-Following Intelligent" strategy logic
    print("Génération du label 'signal' pour la stratégie Trend-Following Intelligent...")
    df['signal'] = 0 # Neutral signal by default

    # Conditions for Long (achat)
    long_condition_ema = df['ema_21'] > df['ema_50']
    long_condition_rsi = df['rsi'] > 50
    long_condition_supertrend = df['close'] > df['supertrend'] # Assumes 'supertrend' column exists
    long_condition_close_avg = df['close'] > df['close_sma_3']
    long_condition_atr = df['atr'] > df['atr_sma_20'] # Example: ATR above its 20-period SMA as a minimum threshold

    combined_long_conditions = (
        long_condition_ema &
        long_condition_rsi &
        long_condition_supertrend &
        long_condition_close_avg &
        long_condition_atr
    )
    df.loc[combined_long_conditions, 'signal'] = 1

    # Conditions for Short (vente à découvert)
    short_condition_ema = df['ema_21'] < df['ema_50']
    short_condition_rsi = df['rsi'] < 50
    short_condition_supertrend = df['close'] < df['supertrend'] # Assumes 'supertrend' column exists
    short_condition_close_avg = df['close'] < df['close_sma_3']
    # For "ATR élevé", let's use the same condition as for long for now (ATR > ATR_SMA_20)
    # This should be refined based on what "élevé" (high) means in context.
    short_condition_atr = df['atr'] > df['atr_sma_20']

    combined_short_conditions = (
        short_condition_ema &
        short_condition_rsi &
        short_condition_supertrend &
        short_condition_close_avg &
        short_condition_atr
    )
    df.loc[combined_short_conditions, 'signal'] = -1


    # 5. Handle NaNs introduced by rolling windows or shifts AFTER all calculations.
    # The original script had a good approach for this.
    # df.replace([np.inf, -np.inf], np.nan, inplace=True) # Already in original if needed
    initial_rows_with_nan = df.isnull().any(axis=1).sum()
    if initial_rows_with_nan > 0:
        print(f"Suppression des lignes contenant des NaNs après calculs de stratégie ({initial_rows_with_nan} lignes)...")
        df.dropna(inplace=True)
        if df.empty:
            print("Erreur: Plus de données restantes après suppression des NaNs post-stratégie.")
            return
    else:
        print("Aucun NaN détecté dans les features calculées pour la stratégie.")

    # 6. Exit Conditions (Placeholders / Notes)
    # As discussed, full implementation of dynamic exits is complex here.
    # This script focuses on generating entry signals.
    # Stop-loss, take-profit, and alternative exits (RSI to 50, EMA crossover)
    # would typically be handled by a backtesting engine or execution bot that tracks individual trades.
    # For now, we are only generating the primary entry signal (1 for buy, -1 for sell, 0 for hold).
    # We can add columns for SL/TP levels if needed for external systems, but the logic to *act* on them is separate.
    # df['stop_loss_price'] = np.nan
    # df['take_profit_price'] = np.nan
    # if 'atr' in df.columns:
    #    df.loc[df['signal'] == 1, 'stop_loss_price'] = df['close'] - (df['atr'] * 1.5)
    #    df.loc[df['signal'] == 1, 'take_profit_price'] = df['close'] + (df['atr'] * 1.5 * 2) # SL * 2
    #    df.loc[df['signal'] == -1, 'stop_loss_price'] = df['close'] + (df['atr'] * 1.5)
    #    df.loc[df['signal'] == -1, 'take_profit_price'] = df['close'] - (df['atr'] * 1.5 * 2) # SL * 2

    # --- END OF MODIFICATIONS FOR NEW STRATEGY ---

    # --- RISK MANAGEMENT PLACEHOLDER ---
    # The 'signal' column generated by this script is intended for consumption by a separate
    # trading execution system or backtesting engine. That system would be responsible for
    # implementing actual risk management rules, such as:
    #   1. Position Sizing: e.g., Max 2% of total capital engaged per trade.
    #   2. Daily Drawdown Limits: e.g., If daily drawdown (realized or unrealized) exceeds 5%,
    #      pause all new trading activity for the bot until the next trading day/session.
    #   3. Circuit Breakers: e.g., If 3 consecutive losing trades occur, temporarily pause
    #      the strategy or specific asset trading.
    #   4. Stop-Loss / Take-Profit: While SL/TP levels can be calculated here (as shown in commented
    #      out section above), their actual execution and management during a trade's lifecycle
    #      are handled by the execution engine.
    # This script focuses on generating the potential entry signals based on technical analysis.

    print(f"Export du dataset complet vers {output_path}...")
    # Colonnes utiles pour le ML
    cols_utiles = [
        'open', 'high', 'low', 'close', 'tickvol',
        'ema_21', 'ema_50', 'rsi', 'atr', 'supertrend',
        'close_sma_3', 'atr_sma_20', 'signal', 'timestamp'
    ]
    # On garde seulement les colonnes présentes dans le DataFrame
    cols_utiles = [col for col in cols_utiles if col in df.columns]
    df_to_export = df[cols_utiles].copy()
    df_to_export.to_csv(output_path, index=True)
    print("Préparation terminée.")
    print(f"Dataset filtré enregistré sous {output_path}.")
    print(f"Shape du dataset final : {df_to_export.shape}")
    print("Aperçu des premières lignes :")
    print(df_to_export.head())

if __name__ == "__main__":
    # Ensure the input file exists or adjust path.
    # The user provided "data/features/BTCUSD_M1_201812120809_202505271849.csv"
    # This path implies the script is run from the project root.
    # If INPUT_CSV_PATH is not found, the script will print an error and exit.
    prepare_dataset(INPUT_CSV_PATH, OUTPUT_CSV_PATH)
