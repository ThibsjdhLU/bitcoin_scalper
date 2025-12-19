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
├── 📁 src/                          # Code source
│   └── bitcoin_scalper/
│       ├── core/                    # ML, risk, backtesting
│       ├── connectors/              # MT5 REST
│       ├── threads/                 # Trading workers
│       ├── ui/                      # Interface PyQt
│       ├── web/                     # API FastAPI
│       ├── utils/                   # Utilitaires
│       └── main.py
│
├── 📁 scripts/                      # Scripts autonomes
│   ├── train.py                     # Entraînement ML
│   ├── encrypt_config.py            # Chiffrement config
│   ├── decrypt_config.py            # Déchiffrement config
│   └── check_password_key.py        # Vérification password
│
├── 📁 data/                         # Données du projet
│   ├── raw/                         # Données brutes (CSV historiques)
│   └── features/                    # Features engineering
│
├── 📁 models/                       # Modèles ML entraînés
│   └── model_model.cbm              # Modèle CatBoost
│
├── 📁 reports/                      # Rapports et métriques
│   ├── backtest/                    # Résultats backtests
│   ├── ml/                          # Métriques ML
│   └── logs/                        # Logs temporaires
│
├── 📁 config/                       # Configuration
│   ├── config.json                  # Config en clair (dev)
│   ├── config.enc                   # Config chiffrée (prod)
│   └── .env.example                 # Template variables d'environnement
│
├── 📁 resources/                    # Ressources statiques
│   └── icons/                       # SVG pour l'UI
│
├── 📁 docs/                         # Documentation
│   ├── README_TRAINING.md           
│   ├── GUIDE_RAPIDE_TRAINING.md     
│   └── REPONSE_TRAINING.md          
│
├── .gitignore
├── README.md                        # Documentation principale
├── requirements.txt
└── pyproject.toml
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
python scripts/train.py
```

Le script utilise automatiquement les données dans `data/raw/BTCUSD_M1_202301010000_202512011647.csv`.

Pour plus de détails : [docs/README_TRAINING.md](docs/README_TRAINING.md)

### Lancer le bot de trading

Option 1 (avec PYTHONPATH) :
```sh
PYTHONPATH=src python -m bitcoin_scalper.main
```

Option 2 (avec installation en mode développement) :
```sh
pip install -e .
python -m bitcoin_scalper.main
```

Le bot lance automatiquement :
- Le dashboard PyQt
- L'ingestion de données temps réel
- L'exécution des stratégies de trading
- L'API FastAPI (optionnel)

### Configuration

Éditez `config/config.json` ou utilisez `config/config.enc` (chiffré avec AES-256) :

```json
{
  "MT5_REST_URL": "http://localhost:8000",
  "MT5_REST_API_KEY": "your_api_key",
  "DEFAULT_SL_PCT": 0.01,
  "DEFAULT_TP_PCT": 0.02,
  "SL_ATR_MULT": 2.0,
  "TP_ATR_MULT": 3.0,
  "ML_MODEL_PATH": "models/model"
}
```

Pour chiffrer la configuration :
```sh
python scripts/encrypt_config.py config/config.json config/config.enc <clé_hex>
```

Pour déchiffrer la configuration :
```sh
python scripts/decrypt_config.py config/config.enc <clé_hex>
```

Pour générer une clé depuis un mot de passe :
```sh
python scripts/check_password_key.py <mot_de_passe>
```

## Sécurité

- Configuration chiffrée avec AES-256 et dérivation PBKDF2
- Pas de secrets en clair dans le code
- Mot de passe demandé au démarrage pour déchiffrer la config

## Documentation

- [Guide de migration](MIGRATION.md) - Instructions pour migrer depuis l'ancienne structure
- [Guide d'entraînement](docs/README_TRAINING.md) - Documentation complète sur le ML pipeline
- [Guide rapide](docs/GUIDE_RAPIDE_TRAINING.md) - Démarrage rapide pour l'entraînement
- [Réponses FAQ](docs/REPONSE_TRAINING.md) - Questions fréquentes sur le training 