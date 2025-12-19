# Bitcoin Scalper

Bot de trading algorithmique BTC/USD avec Machine Learning, gestion du risque, et interface PyQt.

## Fonctionnalités

- **Trading algorithmique** :
  - Stratégie de scalping BTC/USD avec signaux ML
  - Gestion automatique des positions (Stop Loss / Take Profit)
  - Exécution d'ordres avancée (Iceberg, VWAP, TWAP)
  
- **Machine Learning** :
  - Pipeline ML complet : feature engineering, entraînement, backtesting
  - Modèles CatBoost/LightGBM/XGBoost avec calibration des probabilités
  - Prédiction temps réel et évaluation continue
  
- **Gestion du risque** :
  - Risk management avec calcul ATR pour SL/TP dynamiques
  - Validation des positions avant exécution
  - Monitoring du drawdown et PnL
  
- **Interface utilisateur** :
  - Dashboard PyQt avec graphiques en temps réel
  - Visualisation des positions et métriques
  - API FastAPI pour supervision à distance

## Structure du projet

```
bitcoin_scalper/
├── bitcoin_scalper/     # Code source principal
│   ├── core/            # ML, risk, ingestion, backtesting
│   ├── connectors/      # MT5 REST client
│   ├── threads/         # Trading worker
│   ├── ui/              # Dashboard PyQt
│   ├── web/             # API FastAPI
│   ├── scripts/         # Utilitaires
│   └── main.py          # Point d'entrée
├── data/                # Données historiques CSV
├── config.json          # Configuration (ou config.enc chiffrée)
├── model_model.cbm      # Modèle ML entraîné
└── requirements.txt     # Dépendances Python
```

## Installation

Python 3.11.x recommandé

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

### Entraîner le modèle ML

```sh
python train.py
```

Le script utilise automatiquement les données dans `/data/BTCUSD_M1_202301010000_202512011647.csv`.

Pour plus de détails : [README_TRAINING.md](README_TRAINING.md)

### Lancer le bot de trading

```sh
python -m bitcoin_scalper.main
```

Le bot lance automatiquement :
- Le dashboard PyQt
- L'ingestion de données temps réel
- L'exécution des stratégies de trading
- L'API FastAPI (optionnel)

### Configuration

Éditez `config.json` ou utilisez `config.enc` (chiffré avec AES-256) :

```json
{
  "MT5_REST_URL": "http://localhost:8000",
  "MT5_REST_API_KEY": "your_api_key",
  "DEFAULT_SL_PCT": 0.01,
  "DEFAULT_TP_PCT": 0.02,
  "SL_ATR_MULT": 2.0,
  "TP_ATR_MULT": 3.0
}
```

Pour chiffrer la configuration :
```sh
python encrypt_config.py
```

## Sécurité

- Configuration chiffrée avec AES-256 et dérivation PBKDF2
- Pas de secrets en clair dans le code
- Mot de passe demandé au démarrage pour déchiffrer la config 