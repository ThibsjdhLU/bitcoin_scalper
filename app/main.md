# Script principal du bot de trading BTCUSD (`app/main.py`)

## Description
Ce script orchestre le bot de trading BTCUSD côté macOS : il récupère les données via le serveur REST MT5 (Windows), applique une stratégie simple (RSI), gère le risque, envoie les ordres, et boucle en temps réel.

## Prérequis
- Python 3.11+
- Modules installés (voir `pyproject.toml` ou `requirements.txt`)
- Serveur REST MT5 opérationnel sur Windows (voir `windows/mt5_rest_server.py`)
- Configuration sécurisée (voir ci-dessous)

## Configuration
Les secrets et paramètres sont lus via `SecureConfig` :
- `MT5_REST_URL` : URL du serveur REST MT5 (ex: http://192.168.1.10:8000)
- `MT5_REST_API_KEY` : clé API pour authentification

## Fonctionnement
1. Charge la configuration sécurisée
2. Initialise les modules (client REST, nettoyage, features, risk)
3. Boucle :
   - Récupère les données OHLCV
   - Nettoie les données
   - Calcule le RSI
   - Génère un signal (buy/sell) selon RSI
   - Vérifie la gestion du risque
   - Envoie l'ordre si signal valide
   - Loggue toutes les actions

## Extension
- Remplacer la stratégie RSI par toute stratégie avancée (ML, deep learning, etc.)
- Ajouter des indicateurs, des signaux, ou des modules de monitoring
- Intégrer la supervision, le reporting, ou l'interface web

## Sécurité
- Les clés API ne doivent jamais être hardcodées : utiliser `SecureConfig` et chiffrement AES-256
- Le serveur REST doit être protégé (firewall, VPN, clé API forte)
- Les logs ne doivent pas contenir d'informations sensibles

## Exemple d'usage
```bash
python app/main.py
```

## Dépannage
- Vérifier que le serveur REST MT5 est accessible
- Vérifier la configuration (`MT5_REST_URL`, `MT5_REST_API_KEY`)
- Consulter les logs pour tout message d'erreur

## Pour aller plus loin
- Intégrer des modèles ML (voir `app/core/ml_pipeline.py`)
- Ajouter la supervision Prometheus/Grafana
- Étendre la gestion du risque (VaR, drawdown, etc.) 