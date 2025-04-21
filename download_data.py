#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour télécharger les données historiques de Binance
"""

import os
import sys
import ccxt
import pandas as pd
from datetime import datetime, timedelta

def download_historical_data(symbol, timeframe, start_date, end_date, output_file):
    """
    Télécharge les données historiques de Binance
    
    Args:
        symbol (str): Symbole de trading (ex: BTC/USDT)
        timeframe (str): Intervalle de temps (ex: 1m, 5m, 1h)
        start_date (datetime): Date de début
        end_date (datetime): Date de fin
        output_file (str): Fichier de sortie
    """
    # Initialisation de l'exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future'
        }
    })
    
    # Conversion des dates en timestamps
    since = int(start_date.timestamp() * 1000)
    end = int(end_date.timestamp() * 1000)
    
    # Récupération des données
    all_candles = []
    while since < end:
        print(f"Téléchargement des données pour {datetime.fromtimestamp(since/1000)}")
        
        try:
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=1000  # Maximum par requête
            )
            
            if not candles:
                break
                
            all_candles.extend(candles)
            
            # Mise à jour du timestamp pour la prochaine requête
            since = candles[-1][0] + 1
            
        except Exception as e:
            print(f"Erreur: {e}")
            break
    
    # Conversion en DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Sauvegarde des données
    df.to_csv(output_file, index=False)
    print(f"Données sauvegardées dans {output_file}")

def main():
    # Configuration
    symbol = "BTC/USDT"
    timeframe = "1m"
    start_date = datetime(2024, 1, 1)
    end_date = datetime.now()
    
    # Création du dossier de données si nécessaire
    os.makedirs("data", exist_ok=True)
    
    # Nom du fichier de sortie
    output_file = f"data/{symbol.replace('/', '')}_1m_2024.csv"
    
    # Téléchargement des données
    download_historical_data(symbol, timeframe, start_date, end_date, output_file)

if __name__ == "__main__":
    main() 