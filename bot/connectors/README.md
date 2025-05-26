# Connecteur MetaTrader5 (`MT5Connector`)

Ce module gère la connexion sécurisée et robuste à MetaTrader5 : initialisation, reconnexion automatique, gestion des erreurs, logs détaillés.

## Fonctionnalités
- Connexion/déconnexion à MT5
- Reconnexion automatique en cas de perte
- Gestion des erreurs réseau/MT5
- Logging détaillé
- Dépendance à `SecureConfig` pour les identifiants (login, password, server). Si un identifiant est manquant, une exception explicite est levée.

## Utilisation
```python
from app.core.config import SecureConfig
from bot.connectors.mt5_connector import MT5Connector
import os
config = SecureConfig("/chemin/vers/config.enc", os.environ["CONFIG_AES_KEY"])
mt5c = MT5Connector(config)
mt5c.connect()
# ...
mt5c.disconnect()
```

## Tests
Voir `tests/connectors/test_mt5_connector.py` pour des exemples de tests unitaires et de mock.

# Client REST MT5 multiplateforme (`MT5RestClient`)

Ce module permet d'interagir avec un serveur MT5 distant via API REST, sans dépendance MetaTrader5 locale. Compatible macOS, Linux, Windows, CI/CD.

## Fonctionnalités
- Récupération ticks, OHLCV, statut serveur
- Exécution d'ordres (buy/sell, market/limit)
- Gestion reconnexions, erreurs réseau, authentification (clé API)
- Interface compatible avec le reste du bot

## Endpoints attendus côté serveur
- `GET /ticks?symbol=BTCUSD&limit=100`
- `GET /ohlcv?symbol=BTCUSD&timeframe=M1&limit=100`
- `POST /order` (payload : symbol, action, volume, ...)
- `GET /status`

## Utilisation
```python
from bot.connectors.mt5_rest_client import MT5RestClient
client = MT5RestClient("https://mt5-server.example.com/api", api_key="votre_cle_api")
ticks = client.get_ticks("BTCUSD", limit=200)
res = client.send_order("BTCUSD", action="buy", volume=0.01)
```

## Sécurité
- Clé API obligatoire (header Authorization)
- HTTPS recommandé

## Tests
Voir `tests/connectors/test_mt5_rest_client.py` (mock HTTP, >95% couverture)

## Intégration CI/CD
- Aucun import MetaTrader5 requis côté client
- Compatible Docker, macOS, Linux, Windows 