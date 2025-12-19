# 🚀 Guide Rapide - Entraînement du Modèle ML

## 📋 Question posée
> "Donne moi la commande pour lancer le training de la ML. Doit elle utiliser le fichier csv dans /data ?"

## ✅ Réponses

### 1️⃣ Commande pour lancer le training

**Méthode la plus simple** :
```bash
python train.py
```

**Ou via Makefile** :
```bash
make train
```

**Ou commande complète** :
```bash
python -m bitcoin_scalper.core.orchestrator \
    --csv data/BTCUSD_M1_202301010000_202512011647.csv \
    --fill_missing \
    --export \
    --pipeline ml
```

### 2️⃣ Utilisation du fichier CSV dans /data

**OUI**, le fichier CSV dans `/data` **DOIT** être utilisé pour l'entraînement.

#### 📁 Fichier concerné
```
data/BTCUSD_M1_202301010000_202512011647.csv
```

#### 📊 Caractéristiques
- **Taille** : ~98 MB
- **Période** : Janvier 2023 → Décembre 2025
- **Résolution** : 1 minute (M1)
- **Contenu** : Données OHLCV (Open, High, Low, Close, Volume) de BTC/USD

#### 🤖 Configuration automatique
Le script `train.py` utilise **automatiquement** ce fichier CSV par défaut.
Vous n'avez rien à configurer !

## 🎯 Workflow complet

```
1. Vérifier les prérequis
   └─ pip install -r requirements.txt

2. Lancer l'entraînement
   └─ python train.py
   
3. Le modèle est sauvegardé
   └─ model_model.cbm

4. Utiliser le modèle dans le bot
   └─ Configurer ML_MODEL_PATH dans config.json
```

## 📚 Documentation détaillée

- **Guide complet** : [README_TRAINING.md](README_TRAINING.md)
- **Réponse rapide** : [REPONSE_TRAINING.md](REPONSE_TRAINING.md)
- **Roadmap ML** : [docs/roadmap_ml_training.md](docs/roadmap_ml_training.md)

## 🔍 Vérifications

### Vérifier que le CSV existe
```bash
ls -lh data/BTCUSD_M1_202301010000_202512011647.csv
```

### Vérifier le modèle après entraînement
```bash
ls -lh model_model.cbm
```

## ⚙️ Paramètres avancés

Le script `train.py` utilise ces paramètres par défaut :
- `--fill_missing` : Comble les trous temporels dans les données
- `--export` : Sauvegarde le modèle après entraînement
- `--model_prefix model_model` : Nom du fichier de sortie
- `--pipeline ml` : Utilise le pipeline ML classique

Pour personnaliser, voir [README_TRAINING.md](README_TRAINING.md).

## 🎉 C'est tout !

**Commande unique** : `python train.py`

**Fichier CSV** : Utilisé automatiquement depuis `/data`

Le modèle sera prêt à être utilisé par le bot de trading ! 🤖📈
