#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys

# Ajouter le chemin du répertoire racine du projet pour pouvoir importer les modules locaux
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bitcoin_scalper.core.feature_engineering import FeatureEngineering

# --- Configuration ---
INPUT_CSV_PATH = "data/features/BTCUSD_M1_201812120809_202505271849.csv"
OUTPUT_CSV_PATH = "data/features/BTCUSD_M1_features_phoenix.csv"

# --- Pipeline de Préparation des Features et du Label ---
def prepare_dataset(input_path: str, output_path: str):
    """
    Charge les données brutes, ajoute les features et indicateurs, génère le label 'signal',
    et exporte le dataset complet dans un nouveau fichier CSV.
    """
    print(f"Chargement des données brutes depuis {input_path}...")
    try:
        # 1. Chargement du CSV brut
        # Utilisation de engine='python' pour gérer le séparateur tabulation
        df = pd.read_csv(input_path, sep='\t', engine='python')
    except FileNotFoundError:
        print(f"Erreur : Fichier non trouvé à {input_path}")
        return
    except Exception as e:
        print(f"Erreur lors du chargement du fichier : {e}")
        return

    print("Préparation des données...")
    # 2. Fusion Date/Heure et renommage des colonnes
    # Utilisation de errors='coerce' pour gérer d'éventuelles erreurs de format de date/heure
    df['timestamp'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'], errors='coerce')
    df = df.rename(columns={
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>': 'low',
        '<CLOSE>': 'close',
        '<TICKVOL>': 'tickvol', # Renommer TICKVOL car VOL semble être 0
        '<VOL>': 'volume_zero', # Conserver mais noter qu'il est probablement inutile
        '<SPREAD>': 'spread'
    })
    
    # Supprimer les lignes avec erreur de parsing de date/heure
    df.dropna(subset=['timestamp'], inplace=True)
    df.set_index('timestamp', inplace=True)
    
    # Assurer l'ordre chronologique (important pour les indicateurs rolling)
    df.sort_index(inplace=True)
    
    # Supprimer la colonne inutile
    if 'volume_zero' in df.columns:
        df = df.drop(columns=['volume_zero'])

    # 3. Ajout des indicateurs techniques et features dérivées
    print("Ajout des indicateurs techniques et features...")
    fe = FeatureEngineering()
    # Passer 'tickvol' comme colonne de volume car '<VOL>' est à 0
    df = fe.add_indicators(df, price_col='close', high_col='high', low_col='low', volume_col='tickvol')
    df = fe.add_features(df, price_col='close', volume_col='tickvol')

    # 4. Calcul des moyennes mobiles supplémentaires pour la stratégie Phoenix
    print("Calcul des features spécifiques à la stratégie Phoenix...")
    # Gérer la division par zéro potentielle dans vol_price_ratio_sma_5 si tickvol est 0 pour une période
    # On remplace les infinis potentiels par NaN puis 0, ou on filtre. Faisons-le après le calcul.
    df['atr_sma_10'] = df['atr'].rolling(window=10).mean()
    df['tickvol_sma_5'] = df['tickvol'].rolling(window=5).mean()
    df['vol_price_ratio_sma_5'] = df['vol_price_ratio'].rolling(window=5).mean()

    # Remplacer les infinis par NaN pour une gestion cohérente
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 5. Gestion des valeurs NaN (supprimer les premières lignes impactées par les calculs rolling/shift)
    # Cette étape doit venir *avant* la génération du signal pour que les conditions soient évaluées sur des données propres.
    initial_rows_with_nan = df.isnull().any(axis=1).sum()
    if initial_rows_with_nan > 0:
        print(f"Suppression des {initial_rows_with_nan} premières lignes contenant des NaNs...")
        df.dropna(inplace=True)
        # S'assurer qu'il reste des données après suppression
        if df.empty:
            print("Erreur: Plus de données restantes après suppression des NaNs.")
            return
    else:
        print("Aucun NaN détecté dans les features calculées.")

    # 6. Génération du label 'signal' (Stratégie Phoenix)
    # Cette étape vient *après* la suppression des NaNs
    print("Génération du label 'signal'...")
    df['signal'] = 0 # Signal neutre par défaut

    # Calcul des conditions pour Achat (signal = 1)
    # Les conditions utilisent maintenant des colonnes qui ne contiennent plus de NaNs initiaux
    buy_condition_trend = (df['close'] > df['ema_20']) & (df['ema_20'] > df['sma_20'])
    # Pour le croisement MACD, on compare la valeur actuelle à la valeur précédente. Le shift(1) introduira temporairement un NaN sur la 1ère ligne restante, qui sera géré par le masque booléen.
    buy_condition_momentum_macd_cross = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    buy_condition_momentum_rsi = (df['rsi'] > 50) & (df['rsi'] < 70)
    buy_condition_momentum = buy_condition_momentum_macd_cross & buy_condition_momentum_rsi

    buy_condition_volatility = (df['close'] > df['sma_20']) & (df['close'] <= df['bb_high']) & (df['atr'] > df['atr_sma_10'])
    
    buy_condition_volume = (df['tickvol'] > df['tickvol_sma_5']) & (df['vol_price_ratio'] > df['vol_price_ratio_sma_5'])
    buy_condition_return = df['return'] > 0

    # Conditions d'achat complètes (combinaison logique)
    # S'assurer que les conditions sont bien de type booléen (en cas de NaNs résiduels)
    buy_conditions = buy_condition_trend & buy_condition_momentum & buy_condition_volatility & buy_condition_volume & buy_condition_return
    
    # Ajouter les colonnes booléennes pour l'analyse
    df['buy_cond_trend'] = buy_condition_trend
    df['buy_cond_momentum'] = buy_condition_momentum
    df['buy_cond_volatility'] = buy_condition_volatility
    df['buy_cond_volume'] = buy_condition_volume
    df['buy_cond_return'] = buy_condition_return

    df.loc[buy_conditions, 'signal'] = 1

    # Calcul des conditions pour Vente (signal = -1)
    sell_condition_trend = (df['close'] < df['ema_20']) & (df['ema_20'] < df['sma_20'])
    # Pour le croisement MACD, on compare la valeur actuelle à la valeur précédente. Le shift(1) introduira temporairement un NaN sur la 1ère ligne restante, qui sera géré par le masque booléen.
    sell_condition_momentum_macd_cross = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    sell_condition_momentum_rsi = (df['rsi'] < 50) & (df['rsi'] > 30)
    sell_condition_momentum = sell_condition_momentum_macd_cross & sell_condition_momentum_rsi

    sell_condition_volatility = (df['close'] < df['sma_20']) & (df['close'] >= df['bb_low']) & (df['atr'] > df['atr_sma_10'])
    
    sell_condition_volume = (df['tickvol'] > df['tickvol_sma_5']) & (df['vol_price_ratio'] > df['vol_price_ratio_sma_5'])
    sell_condition_return = df['return'] < 0
    
    # Conditions de vente complètes (combinaison logique)
    # S'assurer que les conditions sont bien de type booléen
    sell_conditions = sell_condition_trend & sell_condition_momentum & sell_condition_volatility & sell_condition_volume & sell_condition_return

    # Ajouter les colonnes booléennes pour l'analyse
    df['sell_cond_trend'] = sell_condition_trend
    df['sell_cond_momentum'] = sell_condition_momentum
    df['sell_cond_volatility'] = sell_condition_volatility
    df['sell_cond_volume'] = sell_condition_volume
    df['sell_cond_return'] = sell_condition_return

    df.loc[sell_conditions, 'signal'] = -1

    # 7. Export du CSV complet
    print(f"Export du dataset complet vers {output_path}...")
    # Supprimer les colonnes brutes si elles ne sont plus nécessaires pour l'entraînement
    # Conservation de 'open', 'high', 'low', 'close', 'tickvol', 'spread' car elles font partie des features ou peuvent être utiles pour l'analyse.
    cols_to_drop_before_export = ['<DATE>', '<TIME>'] # Supprimer les colonnes brutes de date/heure
    df = df.drop(columns=cols_to_drop_before_export, errors='ignore')
    
    df.to_csv(output_path, index=True) # Inclure le timestamp comme colonne

    print("Préparation terminée.")
    print(f"Dataset complet enregistré sous {output_path}.")
    print(f"Shape du dataset final : {df.shape}")
    print("Aperçu des premières lignes (après suppression NaNs) :\n", df.head())

# --- Point d'entrée du script ---
if __name__ == "__main__":
    prepare_dataset(INPUT_CSV_PATH, OUTPUT_CSV_PATH) 