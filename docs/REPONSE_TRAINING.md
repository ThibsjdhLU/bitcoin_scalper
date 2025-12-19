# Réponse rapide : Comment lancer l'entraînement ML ?

## Commande pour lancer l'entraînement

### Option 1 : Commande la plus simple (recommandée)
```bash
python train.py
```

### Option 2 : Via Makefile
```bash
make train
```

### Option 3 : Commande complète avec l'orchestrateur
```bash
python -m bitcoin_scalper.core.orchestrator \
    --csv data/BTCUSD_M1_202301010000_202512011647.csv \
    --fill_missing \
    --export \
    --pipeline ml
```

## Est-ce que le fichier CSV dans /data doit être utilisé ?

**OUI**, le fichier CSV dans `/data` doit être utilisé pour l'entraînement.

Le fichier `data/BTCUSD_M1_202301010000_202512011647.csv` contient les données historiques OHLCV (Open, High, Low, Close, Volume) de Bitcoin en résolution 1 minute, couvrant la période de janvier 2023 à décembre 2025.

### Utilisation automatique du CSV
Le script `train.py` est configuré pour utiliser **automatiquement** ce fichier CSV si aucun autre fichier n'est spécifié :

```python
# Dans train.py
csv_file = 'data/BTCUSD_M1_202301010000_202512011647.csv'
```

### Vérifier que le fichier existe
```bash
ls -lh data/BTCUSD_M1_202301010000_202512011647.csv
```

Le fichier fait environ **98 MB** et contient les données nécessaires pour entraîner le modèle de Machine Learning.

## Résumé

1. **Commande** : `python train.py` ou `make train`
2. **Fichier CSV** : Oui, utilise automatiquement `data/BTCUSD_M1_202301010000_202512011647.csv`
3. **Sortie** : Le modèle sera sauvegardé dans `model_model.cbm`

Pour plus de détails, consultez [README_TRAINING.md](README_TRAINING.md).
