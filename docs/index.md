# 🤖 Bot de Trading Crypto (AvaTrade via MT5)

## Vue d'ensemble

Bot de trading crypto automatisé, modulaire et robuste, conçu pour trader sur AvaTrade via MetaTrader 5. Le bot intègre :

- 📈 Multiples stratégies de trading
- 🛡️ Gestion avancée des risques
- 📊 Backtesting et optimisation
- 🔄 Reconnexion automatique
- 📱 Interface de monitoring

## 🚀 Démarrage rapide

1. **Prérequis**
```bash
# Python 3.11+
python -m pip install -r requirements.txt

# MetaTrader 5
# Télécharger et installer depuis le site officiel
```

2. **Configuration**
```bash
# Copier et éditer les fichiers de configuration
cp config/config.example.json config/config.json
cp config/risk_config.example.json config/risk_config.json
```

3. **Lancement**
```bash
# Démarrer le bot
python main.py

# Lancer le moniteur
python monitor.py
```

## 📁 Structure du Projet

```
trading_bot/
├── main.py                 # Point d'entrée principal
├── monitor.py             # Interface de monitoring
├── config/               # Configuration
│   ├── config.json
│   └── risk_config.json
├── core/                # Composants principaux
│   ├── mt5_connector.py
│   ├── order_executor.py
│   ├── risk_manager.py
│   └── strategy_engine.py
├── strategies/          # Stratégies de trading
│   ├── base_strategy.py
│   ├── ema_crossover.py
│   └── rsi_strategy.py
├── backtest/           # Backtesting
│   └── backtest_engine.py
└── utils/             # Utilitaires
    ├── logger.py
    └── indicators.py
```

## 🔧 Configuration

Le bot utilise deux fichiers de configuration principaux :

1. `config/config.json` : Configuration générale
```json
{
    "broker": {
        "mt5": {
            "server": "AvaTrade-Demo",
            "login": "YOUR_LOGIN",
            "password": "YOUR_PASSWORD",
            "symbols": ["BTCUSD", "ETHUSD"]
        }
    }
}
```

2. `config/risk_config.json` : Gestion des risques
```json
{
    "general": {
        "initial_capital": 10000.0,
        "max_drawdown": 0.15,
        "daily_loss_limit": 0.05
    }
}
```

## 📚 Documentation détaillée

- [Guide d'installation](installation.md)
- [Configuration](configuration.md)
- [Composants principaux](components.md)
- [Stratégies de trading](strategies.md)
- [Backtesting](backtesting.md)
- [Monitoring](monitoring.md)
- [API Reference](api/index.md)

## 🤝 Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amazing_feature`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push la branche (`git push origin feature/amazing_feature`)
5. Ouvrir une Pull Request

## 📝 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails. 