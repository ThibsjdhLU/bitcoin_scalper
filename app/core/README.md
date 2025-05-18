# Module Configuration & Connexion MT5

## Configuration sécurisée
- Les identifiants sensibles (login, mot de passe) sont chiffrés AES-256 dans le fichier YAML.
- Utilisez `SecureConfig` pour charger et accéder à la configuration.

## Exemple d'utilisation
```python
from app.core.config import SecureConfig
from app.core.mt5_connector import MT5Connector

# Clé AES-256 (32 bytes) à stocker de façon sécurisée (ex: variable d'environnement)
ENCRYPTION_KEY = b"0123456789abcdef0123456789abcdef"
config = SecureConfig("config/config.yaml", ENCRYPTION_KEY)

mt5_conn = MT5Connector(config)
if mt5_conn.connect():
    print("Connexion MT5 OK")
else:
    print("Erreur connexion MT5")
```

## Sécurité
- Ne jamais versionner les clés non chiffrées.
- Stocker la clé AES dans une variable d'environnement ou un HSM.
- Les logs ne contiennent jamais d'informations sensibles.

## Tests
- Lancer les tests unitaires :
```bash
pytest tests/core/
``` 