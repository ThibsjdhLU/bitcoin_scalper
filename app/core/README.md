# Module de configuration sécurisée (`SecureConfig`)

Ce module gère le chargement sécurisé des identifiants et secrets nécessaires à la connexion MetaTrader5 et autres services sensibles.

## Fonctionnalités
- Stockage des secrets dans un fichier chiffré AES-256 (CBC, padding PKCS7)
- Déchiffrement à la volée via une clé fournie par variable d'environnement (`CONFIG_AES_KEY`)
- Interface simple pour accéder aux paramètres
- Gestion des erreurs et logs

## Utilisation
```python
from app.core.config import SecureConfig
import os
config = SecureConfig("/chemin/vers/config.enc", os.environ["CONFIG_AES_KEY"])
mt5_login = config.get("mt5_login")
```

## Sécurité
- **Ne jamais** versionner le fichier de secrets en clair
- La clé AES-256 doit être fournie par variable d'environnement (`CONFIG_AES_KEY`) et doit faire exactement 32 bytes (256 bits), sinon une exception explicite est levée.
- Le fichier chiffré doit commencer par 16 bytes d'IV, suivis des données chiffrées (base64)

## Format du fichier chiffré
- IV (16 bytes) + données chiffrées (base64)
- Le JSON original doit contenir :
```json
{
  "mt5_login": 123456,
  "mt5_password": "motdepasse",
  "mt5_server": "nom_serveur"
}
```

## Tests
Voir `tests/core/test_config.py` pour des exemples de tests unitaires et de mock.

# Client TimescaleDB (`TimescaleDBClient`)

Ce module gère la connexion, la création du schéma et l'insertion performante de données ticks/OHLCV dans TimescaleDB (PostgreSQL).

## Fonctionnalités
- Connexion sécurisée (SSL par défaut)
- Création automatique des tables et index (ticks, ohlcv)
- Insertion batch performante (execute_batch)
- Gestion reconnexions, erreurs, logs
- Schéma optimisé pour requêtes temporelles

## Schéma des tables
- `ticks(symbol, timestamp, bid, ask, volume)`
- `ohlcv(symbol, timestamp, open, high, low, close, volume, timeframe)`
- Index temporels et hypertables TimescaleDB

## Utilisation
```python
from app.core.timescaledb_client import TimescaleDBClient
db = TimescaleDBClient(host, port, dbname, user, password)
db.create_schema()
db.insert_ticks([{...}])
db.insert_ohlcv([{...}])
db.close()
```

## Sécurité
- Connexion SSL par défaut
- Jamais de credentials en clair dans le code (utiliser SecureConfig)

## Tests
Voir `tests/core/test_timescaledb_client.py` pour des exemples de tests unitaires (mock complet, >95% couverture).

# Pipeline d'ingestion temps réel (`DataIngestor`)

Ce module gère l'ingestion continue des données ticks et OHLCV depuis un serveur MT5 REST, avec insertion batch dans TimescaleDB.

## Fonctionnalités
- Récupération temps réel via MT5RestClient (REST)
- Batching, filtrage par timestamp, robustesse réseau
- Insertion optimisée dans TimescaleDB
- Interface start/stop, logs détaillés

## Utilisation
```python
from bot.connectors.mt5_rest_client import MT5RestClient
from app.core.timescaledb_client import TimescaleDBClient
from app.core.data_ingestor import DataIngestor
mt5_client = MT5RestClient("https://mt5-server/api", api_key="cle")
db_client = TimescaleDBClient(...)
ingestor = DataIngestor(mt5_client, db_client)
ingestor.start()
# ...
ingestor.stop()
```

## Robustesse
- Gestion reconnexions, erreurs réseau, logs
- Thread dédié, arrêt propre

## Tests
Voir `tests/core/test_data_ingestor.py` (mock complet, >95% couverture)

## Intégration
- Compatible avec le client REST multiplateforme et TimescaleDB
- Nettoyage avancé à intégrer dans l'étape suivante 

# Nettoyage avancé des données (`DataCleaner`)

Ce module fournit un nettoyage avancé des données ticks et OHLCV :
- Suppression des outliers (z-score, IQR)
- Gestion des valeurs manquantes
- Détection d'anomalies (Isolation Forest)

## Fonctionnalités
- Méthodes : `clean_ticks`, `clean_ohlcv`
- Paramétrage du seuil z-score, IQR, contamination Isolation Forest
- Purement transformation de données (pas d'IO)

## Utilisation
```python
from app.core.data_cleaner import DataCleaner
cleaner = DataCleaner()
ticks_clean = cleaner.clean_ticks(ticks)
ohlcv_clean = cleaner.clean_ohlcv(ohlcv)
```

## Robustesse
- Suppression automatique des valeurs aberrantes et lignes incomplètes
- Détection d'anomalies non supervisée (Isolation Forest)

## Tests
Voir `tests/core/test_data_cleaner.py` (outliers, missing, anomalies, edge cases, >95% couverture)

## Intégration
- À utiliser dans le pipeline d'ingestion avant stockage TimescaleDB 

# Feature engineering pour ML (`FeatureEngineering`)

Ce module calcule les indicateurs techniques principaux et extrait des features dérivées pour l'entraînement ML.

## Indicateurs calculés
- RSI, MACD, EMA, SMA, Bollinger Bands, ATR, VWAP
- Support multi-timeframe (1min, 5min, 15min, 1h…)

## Features dérivées
- Retours simples et log, volatilité rolling, ratio volume/prix

## Utilisation
```python
from app.core.feature_engineering import FeatureEngineering
fe = FeatureEngineering(["1min", "5min"])
df_1m = fe.add_indicators(df_1m)
df_1m = fe.add_features(df_1m)
features = fe.multi_timeframe({"1min": df_1m, "5min": df_5m})
```

## Robustesse
- Calcul vectorisé, gestion edge cases, pas d'IO

## Tests
Voir `tests/core/test_feature_engineering.py` (couverture >95%, edge cases, multi-timeframe) 

# Module order_execution.py

## Fonctionnalité principale

Ce module fournit une fonction unique `send_order` permettant d'envoyer des ordres d'achat/vente sur MetaTrader 5 via un client REST multiplateforme (`MT5RestClient`). Il est conçu pour fonctionner nativement sous macOS, sans dépendance MetaTrader5 native.

## Fonction : `send_order`

```python
from app.core.order_execution import send_order
from bot.connectors.mt5_rest_client import MT5RestClient

client = MT5RestClient(base_url="https://mt5-server.example.com/api", api_key="votre_cle_api")
result = send_order(
    symbol="BTCUSD",
    volume=0.01,
    order_type="buy",  # ou "sell"
    price=None,         # None pour market, sinon float pour limit
    client=client
)
```

### Structure de retour
- Succès :
```python
{
    "success": True,
    "data": { ...réponse JSON du serveur MT5... }
}
```
- Échec (erreur MT5, réseau, authentification) :
```python
{
    "success": False,
    "error": "Message d'erreur explicite"
}
```

### Gestion des erreurs
- **Erreur réseau ou serveur** : message explicite, pas d'exception non gérée.
- **Échec d'authentification** : message clair, pas de fuite de clé API.
- **Erreur inattendue** : message générique, log détaillé côté serveur.
- **Client non fourni** : lève `ValueError`.

## Importance de l'isolation via MT5RestClient
- **Compatibilité macOS** : aucune dépendance native MetaTrader5 requise, tout passe par REST.
- **Sécurité** : pas d'exposition directe des clés ou de la logique d'exécution locale.
- **Testabilité** : le client est mockable, permettant des tests unitaires CI/CD sans accès réel à MT5.

## Exemples d'appel
```python
# Ordre market
send_order("BTCUSD", 0.01, "buy", client=client)

# Ordre limit
send_order("BTCUSD", 0.01, "sell", price=65000.0, client=client)
```

## Couverture de tests
- Succès, échec MT5, erreur réseau, crash inattendu, absence de client : tous testés et mockés.
- Compatible CI/CD macOS/Linux. 

# Module risk_management.py

## Fonctionnalité principale

Ce module fournit une classe `RiskManager` pour la gestion du risque (drawdown, perte max, taille position, PnL) en trading BTC/USD, en s'appuyant exclusivement sur le client REST `MT5RestClient` (aucune dépendance native MT5, compatible macOS).

## Classe : `RiskManager`

### Initialisation
```python
from app.core.risk_management import RiskManager
from bot.connectors.mt5_rest_client import MT5RestClient
client = MT5RestClient(base_url="https://mt5-server/api", api_key="cle")
risk = RiskManager(client, max_drawdown=0.05, max_daily_loss=0.05, risk_per_trade=0.01, max_position_size=1.0)
```

### Vérifier l'ouverture d'une position
```python
res = risk.can_open_position("BTCUSD", 0.5)
if res["allowed"]:
    # Ouvrir la position
else:
    print(res["reason"])
```

### Calculer la taille de position optimale
```python
size = risk.calculate_position_size("BTCUSD", stop_loss=50)  # 50 points/pips
```

### Mise à jour après un trade
```python
risk.update_after_trade(profit=120.0)
```

### Obtenir les métriques de risque
```python
metrics = risk.get_risk_metrics()
print(metrics)
# {'drawdown': ..., 'daily_pnl': ..., 'peak_balance': ..., 'last_balance': ...}
```

## Structure de retour
- `can_open_position` : `{ 'allowed': bool, 'reason': str }`
- `calculate_position_size` : `float` (lots)
- `get_risk_metrics` : `dict` (drawdown, PnL, peak, last balance)

## Gestion des erreurs
- **Erreur réseau/REST** : message explicite, pas d'exception non gérée.
- **Paramètres invalides** : message clair dans le champ `reason`.
- **Robustesse** : logs détaillés, aucune dépendance native MT5.

## Importance de l'isolation via MT5RestClient
- **Compatibilité macOS** : tout passe par REST, aucune dépendance native.
- **Sécurité** : pas d'accès direct aux credentials ou à l'API MT5 locale.
- **Testabilité** : client mockable, tests unitaires CI/CD sans accès réel à MT5.

## Couverture de tests
- Succès, refus (drawdown, perte, taille), erreurs réseau, calculs, update, métriques : tous testés et mockés.
- Compatible CI/CD macOS/Linux. 

# Gestion des datasets et versioning (DVC)

Les datasets utilisés pour l'entraînement ML, le backtesting et la reproductibilité sont stockés dans le dossier `data/` à la racine du projet :
- `data/raw` : données brutes extraites
- `data/clean` : données nettoyées (après DataCleaner)
- `data/features` : datasets enrichis (après FeatureEngineering)

Ces dossiers sont suivis par DVC pour garantir la reproductibilité parfaite des expériences ML et des backtests. Les fichiers `.dvc` sont versionnés dans git, mais les données elles-mêmes ne le sont pas.

Pour ajouter ou mettre à jour un dataset :
```sh
# Ajouter un nouveau fichier ou dossier de données
cp mon_fichier.csv data/raw/
dvc add data/raw
# Committer le .dvc
git add data/raw.dvc
```

Pour synchroniser les datasets sur un autre poste :
```sh
dvc pull
```

# MLPipeline avancé (ML & Deep Learning)

Ce module fournit un pipeline unifié pour l'entraînement, la validation, le tuning et la prédiction de modèles ML supervisés et deep learning séquentiel.

## Modèles supportés
- RandomForest, XGBoost (tabulaire)
- DNN, LSTM, Transformer, CNN1D (PyTorch, séquentiel)

## API principale
```python
from app.core.ml_pipeline import MLPipeline

# Tabulaire (RandomForest)
pipe = MLPipeline(model_type="random_forest")
metrics = pipe.fit(X, y, val_split=0.2, cv=3)
preds = pipe.predict(X_test)
probas = pipe.predict_proba(X_test)
pipe.save("model_rf.pkl")
pipe2 = MLPipeline(model_type="random_forest")
pipe2.load("model_rf.pkl")

# Séquentiel (LSTM)
pipe = MLPipeline(model_type="lstm", params={"input_dim":8, "output_dim":2})
metrics = pipe.fit(X_seq, y, epochs=5)
preds = pipe.predict(X_seq)

# Tuning hyperparamètres (GridSearch)
metrics = pipe.tune(X, y, param_grid={"n_estimators": [50, 100]}, cv=3)

# Explicabilité (SHAP)
shap_vals = pipe.explain(X, method="shap")
```

## Intégration DVC
- Activez `dvc_track=True` pour versionner automatiquement les modèles sauvegardés.
- Les artefacts sont ajoutés, commités et poussés via DVCManager.

## Conseils & limitations
- Pour les modèles séquentiels, X doit être de shape `[n, seq, features]`.
- Les modèles deep learning sont CPU par défaut (PyTorch), adaptez le paramètre `device` pour GPU.
- L'explicabilité LIME pour séquentiel est à compléter.
- Les callbacks permettent d'ajouter du logging ou de l'early stopping custom.

## Tests unitaires
- Voir `tests/core/test_ml_pipeline.py` (couverture >95%, mocks torch/DVC, tous modèles)

## Troubleshooting
- Vérifiez que torch, xgboost, shap sont installés.
- Pour DVC, voir la section DVCManager.

# DVCManager

Ce module permet de piloter DVC (Data Version Control) en Python pour le versioning des datasets, features, modèles et artefacts ML.

## Fonctions principales
- Initialisation DVC (`init`)
- Ajout de fichiers/dossiers (`add`)
- Commit DVC (`commit`)
- Push/pull vers/depuis remote
- Reproduction pipeline (`repro`)
- Statut, diff, nettoyage (`status`, `diff`, `gc`)
- Gestion des remotes

## Exemple d'utilisation Python
```python
from app.core.dvc_manager import DVCManager

dvc = DVCManager()
dvc.init()
dvc.add('data/raw/btcusd.csv')
dvc.commit()
dvc.push()
print(dvc.status())
```

## Exemple CLI
```bash
python scripts/dvc_utils.py init
python scripts/dvc_utils.py add --path data/raw/btcusd.csv
python scripts/dvc_utils.py commit --path data/raw/btcusd.csv
python scripts/dvc_utils.py push
python scripts/dvc_utils.py status
```

## Intégration pipeline
- Utiliser DVCManager dans les scripts d'ingestion, nettoyage, feature engineering, ML, etc.
- Automatiser la synchronisation des données et artefacts à chaque étape critique.

## Dépannage
- Vérifier que DVC est installé (`pip install dvc`)
- Les erreurs CLI sont capturées dans le retour de chaque méthode (stdout/stderr)
- Pour les remotes, configurer d'abord `dvc remote add` puis `dvc push` 