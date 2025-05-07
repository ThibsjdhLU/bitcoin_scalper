# Bitcoin Scalper

Bot de trading algorithmique pour le scalping de Bitcoin sur MetaTrader 5.

## 🚀 Fonctionnalités

- Interface graphique avec PySide6
- Connexion à MetaTrader 5
- Stratégies de trading basées sur des indicateurs techniques
- Gestion des risques avancée
- Backtesting des stratégies
- Logging complet des opérations
- API REST pour le monitoring

## 📋 Prérequis

- Python 3.11+
- MetaTrader 5 installé
- Compte AvaTrade (demo ou réel)

## 🛠 Installation

1. Cloner le repository :
```bash
git clone https://github.com/mat0192/bitcoin_scalper.git
cd bitcoin_scalper
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Configurer les variables d'environnement dans le fichier `.env`:
```
MT5_LOGIN=votre_login
MT5_PASSWORD=votre_mot_de_passe
MT5_SERVER=votre_serveur
```

## 🏗 Structure du Projet

```
/bitcoin_scalper/
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

- Les credentials sont stockés dans le fichier `.env`
- Validation des paramètres de trading
- Gestion des erreurs robuste

## 📈 Roadmap

Voir le fichier `roadmap.md` pour les détails de l'évolution du projet.

## 📄 Licence

MIT License
