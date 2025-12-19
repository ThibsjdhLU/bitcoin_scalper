# Guide de Migration - Restructuration du Projet

Ce document explique les changements apportés à la structure du projet `bitcoin_scalper` et les étapes pour migrer votre configuration locale.

## 🎯 Objectif

Réorganisation complète de la structure du projet pour améliorer la lisibilité, la maintenabilité et suivre les meilleures pratiques de développement Python.

## 📋 Changements Principaux

### Structure des Dossiers

#### Ancienne Structure → Nouvelle Structure

```
Ancien                              →  Nouveau
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
bitcoin_scalper/                    →  src/bitcoin_scalper/
├── core/                           →  src/bitcoin_scalper/core/
├── connectors/                     →  src/bitcoin_scalper/connectors/
├── threads/                        →  src/bitcoin_scalper/threads/
├── ui/                             →  src/bitcoin_scalper/ui/
├── web/                            →  src/bitcoin_scalper/web/
├── utils/                          →  src/bitcoin_scalper/utils/
└── main.py                         →  src/bitcoin_scalper/main.py

Scripts à la racine                 →  scripts/
├── train.py                        →  scripts/train.py
├── encrypt_config.py               →  scripts/encrypt_config.py
├── decrypt_config.py               →  scripts/decrypt_config.py
└── check_password_key.py           →  scripts/check_password_key.py

data/                               →  data/
├── *.csv                           →  data/raw/*.csv
├── augmentation.py                 →  data/features/augmentation.py
├── synthetic_ohlcv.py              →  data/features/synthetic_ohlcv.py
└── feature_selection.py            →  data/features/feature_selection.py

model_model.cbm                     →  models/model_model.cbm

backtest_reports/                   →  reports/backtest/
ml_reports/                         →  reports/ml/
catboost_info/                      →  reports/logs/catboost_info/

config.json                         →  config/config.json
config.enc                          →  config/config.enc
                                       config/.env.example (nouveau)

resources/*.svg                     →  resources/icons/*.svg

README_TRAINING.md                  →  docs/README_TRAINING.md
GUIDE_RAPIDE_TRAINING.md            →  docs/GUIDE_RAPIDE_TRAINING.md
REPONSE_TRAINING.md                 →  docs/REPONSE_TRAINING.md
```

## 🔧 Migration Étape par Étape

### 1. Mettre à Jour votre Environnement Git

```sh
# Mettre à jour depuis la branche
git pull origin <branch_name>

# Vérifier que tous les fichiers sont bien en place
ls -la src/bitcoin_scalper/
ls -la scripts/
ls -la config/
ls -la models/
```

### 2. Mettre à Jour vos Chemins de Configuration

Si vous aviez des fichiers de configuration locaux :

**Ancien :**
```sh
config.json
config.enc
```

**Nouveau :**
```sh
config/config.json
config/config.enc
```

**Action :** Déplacer vos fichiers de configuration :
```sh
# Si vous avez des configs locales
mv config.json config/config.json 2>/dev/null || true
mv config.enc config/config.enc 2>/dev/null || true
```

### 3. Mettre à Jour vos Scripts et Commandes

#### Entraînement ML

**Ancien :**
```sh
python train.py
```

**Nouveau :**
```sh
python scripts/train.py
```

#### Lancement du Bot

**Ancien :**
```sh
python -m bitcoin_scalper.main
```

**Nouveau (Option 1 - PYTHONPATH) :**
```sh
PYTHONPATH=src python -m bitcoin_scalper.main
```

**Nouveau (Option 2 - Installation en mode dev) :**
```sh
pip install -e .
python -m bitcoin_scalper.main
```

#### Scripts de Configuration

**Ancien :**
```sh
python encrypt_config.py config.json config.enc <key>
python decrypt_config.py config.enc <key>
python check_password_key.py <password>
```

**Nouveau :**
```sh
python scripts/encrypt_config.py config/config.json config/config.enc <key>
python scripts/decrypt_config.py config/config.enc <key>
python scripts/check_password_key.py <password>
```

### 4. Mettre à Jour les Chemins dans vos Fichiers de Configuration

Si vous avez personnalisé `config.json`, mettez à jour les chemins :

**Ancien :**
```json
{
  "ML_MODEL_PATH": "model_rf.pkl"
}
```

**Nouveau :**
```json
{
  "ML_MODEL_PATH": "models/model"
}
```

### 5. Vérifier les Données

Vos fichiers CSV doivent maintenant être dans `data/raw/` :

```sh
ls -la data/raw/
# Devrait afficher: BTCUSD_M1_202301010000_202512011647.csv
```

## 🔄 Changements dans le Code

### Imports

Les imports des modules n'ont pas changé grâce à la structure `src/` :

```python
# Ces imports fonctionnent toujours
from bitcoin_scalper.core.config import SecureConfig
from bitcoin_scalper.core.modeling import predict
```

### Chemins de Fichiers

Les chemins hardcodés ont été mis à jour pour utiliser des chemins relatifs au projet :

**Ancien :**
```python
config = SecureConfig("config.enc", aes_key)
ml_model_path = "model_rf.pkl"
features_path = "data/features/BTCUSD_M1.csv"
```

**Nouveau :**
```python
config = SecureConfig("config/config.enc", aes_key)
ml_model_path = "models/model"
features_path = "data/features/BTCUSD_M1.csv"
```

## 📝 Nouveaux Fichiers

### config/.env.example

Un template de configuration a été ajouté. Vous pouvez le copier et le personnaliser :

```sh
cp config/.env.example config/.env
# Éditer config/.env avec vos valeurs
```

### src/bitcoin_scalper/ui/positions_model.py

Un nouveau module `PositionsModel` a été créé pour gérer l'affichage des positions dans l'interface PyQt.

## 🐛 Résolution de Problèmes

### Erreur : "ModuleNotFoundError: No module named 'bitcoin_scalper'"

**Solution :**
```sh
# Option 1: Utiliser PYTHONPATH
PYTHONPATH=src python -m bitcoin_scalper.main

# Option 2: Installer en mode développement
pip install -e .
```

### Erreur : "FileNotFoundError: config.enc"

**Solution :**
```sh
# Vérifier que votre config est dans le bon dossier
ls config/config.enc

# Si nécessaire, recréer depuis config.json
python scripts/encrypt_config.py config/config.json config/config.enc <key>
```

### Erreur : "FileNotFoundError: model_model.cbm"

**Solution :**
```sh
# Vérifier que le modèle est dans le bon dossier
ls models/model_model.cbm

# Si nécessaire, réentraîner le modèle
python scripts/train.py
```

### Erreur : "FileNotFoundError: data/raw/BTCUSD_M1_*.csv"

**Solution :**
```sh
# Vérifier que les données sont dans le bon dossier
ls data/raw/

# Si les fichiers sont ailleurs, les déplacer
mv data/*.csv data/raw/
```

## ✅ Checklist de Migration

- [ ] Code mis à jour depuis Git
- [ ] Fichiers de configuration déplacés vers `config/`
- [ ] Commandes mises à jour dans les scripts/CI/CD
- [ ] Chemins de modèles vérifiés dans `models/`
- [ ] Données CSV déplacées vers `data/raw/`
- [ ] Tests d'import Python réussis
- [ ] Bot démarre correctement avec `PYTHONPATH=src python -m bitcoin_scalper.main`
- [ ] Scripts d'entraînement fonctionnent avec `python scripts/train.py`

## 📚 Avantages de la Nouvelle Structure

1. **Séparation claire** : Code source dans `src/`, scripts autonomes dans `scripts/`
2. **Conformité aux standards** : Structure conforme au PEP 517/518
3. **Meilleure organisation** : Données, modèles, rapports et configs dans des dossiers dédiés
4. **Packaging facilité** : Structure compatible avec `pip install`
5. **Documentation centralisée** : Tous les docs dans `docs/`

## 🆘 Besoin d'Aide ?

Si vous rencontrez des problèmes lors de la migration :

1. Vérifiez que vous êtes sur la bonne branche
2. Consultez les logs d'erreur pour identifier les chemins incorrects
3. Référez-vous aux exemples dans le README.md mis à jour
4. Assurez-vous que votre environnement virtuel est à jour

## 🔗 Liens Utiles

- [README principal](README.md)
- [Guide d'entraînement](docs/README_TRAINING.md)
- [Guide rapide](docs/GUIDE_RAPIDE_TRAINING.md)
