# 🧩 Composants Principaux

## MT5Connector

Le connecteur MT5 gère toutes les interactions avec MetaTrader 5.

### Fonctionnalités

- Connexion/déconnexion automatique
- Gestion des symboles
- Récupération des données OHLCV
- Passage d'ordres

### Exemple d'utilisation

```python
from core.mt5_connector import MT5Connector

# Initialisation
connector = MT5Connector("config/config.json")

# Connexion
connector.connect()

# Récupération d'informations
symbol_info = connector.get_symbol_info("BTCUSD")
print(f"Spread actuel: {symbol_info['spread']} points")

# Déconnexion propre
connector.disconnect()
```

### Gestion de la reconnexion

```python
# La méthode ensure_connection() gère automatiquement la reconnexion
connector.ensure_connection()

# Utilisation avec le context manager
with MT5Connector("config/config.json") as connector:
    symbol_info = connector.get_symbol_info("BTCUSD")
```

## OrderExecutor

L'exécuteur d'ordres gère le passage et le suivi des ordres.

### Fonctionnalités

- Ordres Market/Limit/Stop
- Gestion des ordres partiels
- Modification/Annulation d'ordres
- Suivi des positions

### Exemple d'utilisation

```python
from core.order_executor import OrderExecutor, OrderType, OrderSide

# Initialisation
executor = OrderExecutor(connector)

# Ordre market
success, order_id = executor.execute_market_order(
    symbol="BTCUSD",
    volume=0.1,
    side=OrderSide.BUY,
    sl=49000,
    tp=51000
)

# Ordre limit
success, order_id = executor.execute_limit_order(
    symbol="BTCUSD",
    volume=0.1,
    side=OrderSide.BUY,
    price=50000,
    sl=49000,
    tp=51000
)

# Suivi des ordres partiels
status = executor.check_order_status(order_id)
print(f"Volume exécuté: {status.filled_volume}/{status.volume}")
```

## RiskManager

Le gestionnaire de risques contrôle l'exposition et les limites.

### Fonctionnalités

- Protection contre le drawdown
- Limites journalières
- Taille de position dynamique
- Restrictions par stratégie

### Exemple d'utilisation

```python
from core.risk_manager import RiskManager

# Initialisation
risk_manager = RiskManager("config/risk_config.json")

# Vérification avant ordre
if risk_manager.can_open_position(
    strategy="ema_crossover",
    symbol="BTCUSD",
    side="long",
    price=50000,
    stop_loss=49000
):
    # Calculer la taille optimale
    size = risk_manager.calculate_position_size(
        strategy="ema_crossover",
        symbol="BTCUSD",
        price=50000,
        stop_loss=49000
    )
    # Placer l'ordre...

# Mise à jour après trade
risk_manager.on_trade(
    strategy="ema_crossover",
    symbol="BTCUSD",
    pnl=100
)

# Vérification des métriques
metrics = risk_manager.get_risk_metrics()
print(f"Drawdown actuel: {metrics['current_drawdown']:.2%}")
```

## StrategyEngine

Le moteur de stratégie gère l'exécution des stratégies.

### Fonctionnalités

- Chargement dynamique des stratégies
- Exécution parallèle
- Gestion des signaux
- Logging des décisions

### Exemple d'utilisation

```python
from core.strategy_engine import StrategyEngine
from strategies.ema_crossover import EMACrossoverStrategy

# Création des stratégies
strategy = EMACrossoverStrategy(
    data_fetcher=connector,
    order_executor=executor,
    params={
        "fast_period": 9,
        "slow_period": 21
    }
)

# Initialisation du moteur
engine = StrategyEngine(
    strategies=[strategy],
    risk_manager=risk_manager
)

# Démarrage
engine.start()

# Arrêt propre
engine.stop()
```

## Monitoring

L'interface de monitoring affiche l'état du système en temps réel.

### Fonctionnalités

- Positions ouvertes
- P&L en temps réel
- Drawdown
- Alertes importantes

### Exemple d'utilisation

```python
from monitor import Monitor

# Initialisation
monitor = Monitor(
    risk_manager=risk_manager,
    mt5_connector=connector,
    refresh_rate=1.0
)

# Lancement
monitor.run()
```

## Bonnes pratiques

1. **Gestion des erreurs**
```python
try:
    connector.ensure_connection()
    # Opérations...
except ConnectionError as e:
    logger.error(f"Erreur de connexion: {e}")
    # Gestion de l'erreur...
```

2. **Nettoyage des ressources**
```python
def cleanup():
    executor.cancel_all_orders()
    connector.disconnect()
    logger.info("Nettoyage effectué")

# Enregistrer pour l'arrêt
import atexit
atexit.register(cleanup)
```

3. **Logging**
```python
from loguru import logger

logger.add("logs/trading.log", rotation="1 day")
logger.info("Démarrage du bot")
```

4. **Configuration**
```python
# Toujours valider la configuration
def validate_config(config: dict) -> bool:
    required = ["server", "login", "password"]
    return all(key in config for key in required)
``` 