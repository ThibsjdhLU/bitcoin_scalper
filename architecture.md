# Architecture du Bot de Trading

## Diagramme de Composants

```mermaid
graph TD
    A[main.py] --> B[Core]
    A --> C[Strategies]
    A --> D[Utils]
    A --> E[Tests]
    
    B --> B1[broker_interface.py]
    B --> B2[order_manager.py]
    B --> B3[strategy_engine.py]
    B --> B4[risk_manager.py]
    
    C --> C1[base_strategy.py]
    C --> C2[ema_crossover.py]
    
    D --> D1[logger.py]
    D --> D2[indicators.py]
    D --> D3[data_fetcher.py]
    
    E --> E1[test_strategy_engine.py]
    
    B1 --> F[MetaTrader5]
    B2 --> F
    D3 --> F
```

## Flow de Données

```mermaid
sequenceDiagram
    participant M as main.py
    participant BI as broker_interface
    participant SE as strategy_engine
    participant OM as order_manager
    participant RM as risk_manager
    
    M->>BI: Initialize MT5 Connection
    BI-->>M: Connection Status
    
    loop Trading Cycle
        BI->>SE: Market Data
        SE->>SE: Process Strategy
        SE->>RM: Check Risk Parameters
        RM-->>SE: Risk Status
        SE->>OM: Trading Signal
        OM->>BI: Execute Order
        BI-->>OM: Order Status
        OM-->>M: Update Status
    end
```

## Description des Composants

### Core
- **broker_interface.py**: Gère la connexion MT5 et les interactions avec le broker
- **order_manager.py**: Gère l'exécution et le suivi des ordres
- **strategy_engine.py**: Orchestre l'exécution des stratégies
- **risk_manager.py**: Applique les règles de gestion des risques

### Strategies
- **base_strategy.py**: Classe abstraite définissant l'interface des stratégies
- **ema_crossover.py**: Implémentation de la stratégie EMA Crossover

### Utils
- **logger.py**: Système de logging centralisé
- **indicators.py**: Calculs des indicateurs techniques
- **data_fetcher.py**: Récupération et stockage des données

### Tests
- **test_strategy_engine.py**: Tests unitaires du moteur de stratégie 