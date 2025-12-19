# Guide d'entraînement du modèle ML

Ce guide explique comment entraîner le modèle de Machine Learning pour le bot de trading Bitcoin.

## Commande rapide (recommandée)

La commande la plus simple pour lancer l'entraînement avec les paramètres par défaut :

```bash
python train.py
```

Cette commande utilise automatiquement le fichier CSV dans `/data/BTCUSD_M1_202301010000_202512011647.csv`.

## Ou via Makefile

```bash
make train
```

## Commande complète avec tous les paramètres

Si vous souhaitez personnaliser l'entraînement, utilisez l'orchestrateur directement :

```bash
python -m bitcoin_scalper.core.orchestrator \
    --csv data/BTCUSD_M1_202301010000_202512011647.csv \
    --model_prefix model_model \
    --label_horizon 15 \
    --label_k 0.5 \
    --block_duration 1D \
    --min_block_size 100 \
    --split_method fixed \
    --train_frac 0.7 \
    --val_frac 0.15 \
    --test_frac 0.15 \
    --tuning optuna \
    --early_stopping_rounds 20 \
    --fill_missing \
    --export \
    --pipeline ml
```

## Description des paramètres principaux

- `--csv` : Chemin vers le fichier CSV contenant les données OHLCV (obligatoire)
  - **Utilise le fichier dans `/data` par défaut**
- `--model_prefix` : Préfixe pour les fichiers du modèle sauvegardé (défaut: `model`)
- `--label_horizon` : Horizon de prédiction en minutes (défaut: 15)
- `--label_k` : Multiplicateur du seuil dynamique pour les labels (défaut: 0.5)
- `--block_duration` : Durée des blocs pour l'équilibrage (défaut: `1D`)
- `--min_block_size` : Taille minimale d'un bloc (défaut: 100)
- `--split_method` : Méthode de split (`fixed` ou `purged_kfold`, défaut: `fixed`)
- `--train_frac` : Fraction des données pour l'entraînement (défaut: 0.7)
- `--val_frac` : Fraction des données pour la validation (défaut: 0.15)
- `--test_frac` : Fraction des données pour le test (défaut: 0.15)
- `--tuning` : Méthode de tuning des hyperparamètres (`optuna` ou `grid`, défaut: `optuna`)
- `--early_stopping_rounds` : Nombre de rounds pour l'early stopping (défaut: 20)
- `--fill_missing` : Active le comblement automatique des trous temporels (recommandé)
- `--export` : Sauvegarde le modèle entraîné (recommandé)
- `--pipeline` : Pipeline à exécuter (`ml`, `tuning`, `backtest`, `rl`, `stacking`, `hybrid`, défaut: `ml`)

## Pipelines disponibles

### Pipeline ML (recommandé pour commencer)
```bash
python train.py --pipeline ml
```
Pipeline de Machine Learning classique avec feature engineering, labeling, balancing, et entraînement.

### Pipeline Tuning
```bash
python train.py --pipeline tuning
```
Se concentre sur l'optimisation des hyperparamètres.

### Pipeline Backtest
```bash
python train.py --pipeline backtest
```
Exécute un backtest sur les données historiques.

### Pipeline RL (Reinforcement Learning)
```bash
python train.py --pipeline rl
```
Utilise l'apprentissage par renforcement.

### Pipeline Stacking
```bash
python train.py --pipeline stacking
```
Combine plusieurs modèles via stacking.

### Pipeline Hybrid
```bash
python train.py --pipeline hybrid
```
Combine stratégies algorithmiques et ML.

## Fichiers de sortie

Après l'entraînement, les fichiers suivants seront créés :

- `model_model.cbm` : Modèle CatBoost entraîné
- `artf_*.pkl` : Fichiers d'artefacts intermédiaires (données nettoyées, features, etc.)
- `ml_reports/` : Rapports d'évaluation du modèle
- `artf_pnl_curve.png` : Courbe de PnL (si `--plot` est activé)

## Utilisation du CSV dans /data

**Oui, le fichier CSV dans `/data` doit être utilisé pour l'entraînement.**

Le fichier `data/BTCUSD_M1_202301010000_202512011647.csv` contient les données historiques OHLCV de Bitcoin en résolution 1 minute. Le script `train.py` utilise automatiquement ce fichier si aucun autre n'est spécifié.

Format attendu du CSV :
- Données OHLCV en résolution 1 minute
- Colonnes requises : `<OPEN>`, `<HIGH>`, `<LOW>`, `<CLOSE>`, `<TICKVOL>`
- Index temporel (timestamp)

## Exemples d'utilisation avancée

### Entraînement avec Q-values (régression)
```bash
python -m bitcoin_scalper.core.orchestrator \
    --csv data/BTCUSD_M1_202301010000_202512011647.csv \
    --qvalue \
    --export \
    --pipeline ml
```

### Entraînement avec cross-validation temporelle
```bash
python -m bitcoin_scalper.core.orchestrator \
    --csv data/BTCUSD_M1_202301010000_202512011647.csv \
    --split_method purged_kfold \
    --purge_window 60 \
    --export \
    --pipeline ml
```

### Entraînement avec tracé de la courbe PnL
```bash
python -m bitcoin_scalper.core.orchestrator \
    --csv data/BTCUSD_M1_202301010000_202512011647.csv \
    --export \
    --plot \
    --pipeline ml
```

## Vérification après l'entraînement

Pour vérifier que le modèle a bien été entraîné et sauvegardé :

```bash
ls -lh model_model.cbm
```

Pour utiliser le modèle entraîné dans le bot de trading, assurez-vous que le chemin dans `config.json` ou `config.enc` pointe vers le bon fichier :

```json
{
  "ML_MODEL_PATH": "model_model.cbm"
}
```

## Troubleshooting

### Erreur : "Des trous temporels ont été détectés"
Solution : Ajoutez l'option `--fill_missing` à votre commande :
```bash
python train.py --fill_missing
```

### Erreur : "Fichier CSV introuvable"
Solution : Vérifiez que le fichier existe dans le répertoire `/data` :
```bash
ls -lh data/BTCUSD_M1_202301010000_202512011647.csv
```

### Manque de mémoire
Solution : Réduisez la taille des données ou utilisez un subset :
```bash
python -m bitcoin_scalper.core.orchestrator \
    --csv data/subset.csv \
    --export
```

## Support

Pour plus d'informations, consultez :
- `docs/roadmap_ml_training.md` : Roadmap détaillée du pipeline ML
- `bitcoin_scalper/core/orchestrator.py` : Code source de l'orchestrateur
- `bitcoin_scalper/core/modeling.py` : Implémentation du ModelTrainer
