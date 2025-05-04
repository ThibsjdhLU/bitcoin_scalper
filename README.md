# Bot de Trading Crypto (AvaTrade via MT5)

Bot de trading crypto automatisé connecté à AvaTrade via MetaTrader 5.

## 🚀 Fonctionnalités

- Connexion à AvaTrade via MT5
- Stratégies de trading basées sur des indicateurs techniques
- Gestion des risques avancée
- Backtesting des stratégies
- Logging complet des opérations

## 📋 Prérequis

- Python 3.11+
- MetaTrader 5 installé
- Compte AvaTrade (demo ou réel)

## 🛠 Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd bitcoin_scalper
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Configurer le fichier `config/config.json` :
- Ajouter vos identifiants MT5
- Ajuster les paramètres de trading
- Configurer les stratégies

## 🏗 Structure du Projet

```
/trading_bot/
├── main.py                 # Point d'entrée
├── config/                 # Configuration
├── core/                   # Composants principaux
├── strategies/            # Stratégies de trading
├── backtest/              # Outils de backtesting
├── utils/                 # Utilitaires
├── tests/                 # Tests unitaires
└── logs/                  # Fichiers de logs
```

## 🧪 Tests

Exécuter les tests unitaires :
```bash
pytest tests/
```

## 📝 Logging

Les logs sont stockés dans le dossier `logs/` avec :
- Rotation automatique des fichiers
- Différents niveaux de log (DEBUG, INFO, WARNING, ERROR)
- Format détaillé avec timestamp et contexte

## 🔒 Sécurité

- Les credentials sont stockés dans le fichier de configuration
- Validation des paramètres de trading
- Gestion des erreurs robuste

## 📈 Roadmap

Voir le fichier `roadmap.md` pour les détails de l'évolution du projet.

## 📄 Licence

[À DÉFINIR] 