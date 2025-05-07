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

# Bitcoin Scalper

Bot de trading automatique pour Bitcoin avec API REST.

## Installation

1. Cloner le repository
2. Installer les dépendances : `pip install -r requirements.txt`
3. Configurer le fichier `config/config.json`

## Configuration

1. Configurer les paramètres de trading dans `config/config.json` :
   - Paramètres MT5 (broker, login, password)
   - Stratégies de trading
   - Gestion des risques
   - Configuration de l'API :
     ```json
     "api": {
         "key": "votre_token_secret_ici",
         "host": "0.0.0.0",
         "port": 8000
     }
     ```

## Utilisation

### Démarrer le bot

```bash
python main.py
```

### API REST

Le bot expose une API REST sur le port 8000 par défaut.

#### Routes disponibles

- `GET /status` : État du bot
- `GET /logs` : Derniers logs
- `POST /start` : Démarrer le bot
- `POST /stop` : Arrêter le bot

#### Sécurité

Toutes les routes nécessitent un token API dans l'en-tête `X-API-Key`.

Exemple avec curl :
```bash
curl -H "X-API-Key: votre_token_secret_ici" http://localhost:8000/status
```

### Exposition de l'API

Pour exposer l'API sur internet, vous pouvez utiliser :

#### Avec ngrok

1. Installer ngrok : `pip install ngrok`
2. Créer un tunnel : `ngrok http 8000`
3. Utiliser l'URL fournie par ngrok

#### Avec Cloudflare Tunnel

1. Installer cloudflared : https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/
2. Créer un tunnel : `cloudflared tunnel --url http://localhost:8000`
3. Utiliser l'URL fournie par cloudflared

## Documentation

- `docs/` : Documentation technique
- `docs/components.md` : Architecture des composants
- `docs/scalability.md` : Considérations de scalabilité 