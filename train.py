#!/usr/bin/env python3
"""
Script simplifié pour lancer l'entraînement du modèle ML.
Utilise le fichier CSV dans /data pour entraîner le modèle.

Usage:
    python train.py
    ou
    python train.py --csv data/BTCUSD_M1_202301010000_202512011647.csv
"""
import sys
import os

# Ajouter le chemin du projet au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bitcoin_scalper.core.orchestrator import main

if __name__ == '__main__':
    # Si aucun argument n'est fourni, utiliser le CSV par défaut dans /data
    if len(sys.argv) == 1:
        csv_file = 'data/BTCUSD_M1_202301010000_202512011647.csv'
        if not os.path.exists(csv_file):
            print(f"Erreur: Le fichier {csv_file} n'existe pas.")
            print("Veuillez spécifier un fichier CSV avec --csv <chemin>")
            sys.exit(1)
        
        print(f"Utilisation du fichier CSV par défaut: {csv_file}")
        sys.argv.extend([
            '--csv', csv_file,
            '--fill_missing',  # Combler les trous temporels
            '--export',  # Sauvegarder le modèle
            '--model_prefix', 'model_model',  # Préfixe pour les fichiers de modèle
            '--pipeline', 'ml'  # Pipeline ML classique
        ])
    
    # Lancer l'orchestrateur principal
    main()
