# RÉPONSE : Commande Training ML + Utilisation CSV

## Question 1 : Quelle est la commande pour lancer le training de la ML ?

### Réponse directe :
```bash
python train.py
```

### Alternatives :
```bash
# Via Makefile
make train

# Via l'orchestrateur directement (commande complète)
python -m bitcoin_scalper.core.orchestrator \
    --csv data/BTCUSD_M1_202301010000_202512011647.csv \
    --fill_missing \
    --export \
    --pipeline ml
```

## Question 2 : Doit-elle utiliser le fichier CSV dans /data ?

### Réponse directe :
**OUI, absolument.**

Le fichier `data/BTCUSD_M1_202301010000_202512011647.csv` contient les données historiques BTCUSD en résolution 1 minute (M1) nécessaires pour entraîner le modèle.

### Configuration :
Le script `train.py` est **déjà configuré** pour utiliser automatiquement ce fichier CSV. Vous n'avez rien à modifier.

### Détails du fichier :
- **Chemin** : `data/BTCUSD_M1_202301010000_202512011647.csv`
- **Taille** : ~98 MB (102,601,283 octets)
- **Format** : OHLCV (Open, High, Low, Close, TickVolume)
- **Période** : Janvier 2023 - Décembre 2025
- **Résolution** : 1 minute

## Résumé en 3 points :

1. **Commande** : `python train.py`
2. **CSV** : Oui, utilise automatiquement `/data/BTCUSD_M1_202301010000_202512011647.csv`
3. **Sortie** : Modèle sauvegardé dans `model_model.cbm`

---

**Documentation complète** : Voir [README_TRAINING.md](README_TRAINING.md) pour tous les détails et options avancées.
