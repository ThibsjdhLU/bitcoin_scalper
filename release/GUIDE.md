# 📖 Guide d'Utilisation Personnel

## 🚀 Lancement du Bot

1. **Vérification de l'environnement**
```bash
# Vérifier Python
python --version  # Doit être 3.11+

# Vérifier MT5
# MetaTrader 5 doit être installé et connecté à AvaTrade
```

2. **Configuration**
```bash
# Éditer les fichiers de configuration
notepad config/config.json
notepad config/risk_config.json

# Vérifier les paramètres :
# - Credentials MT5
# - Symboles
# - Limites de risque
```

3. **Démarrage**
```bash
# Lancer le bot
python main.py

# Dans un autre terminal, lancer le moniteur
python monitor.py
```

4. **Arrêt propre**
```bash
# Dans le terminal du bot
Ctrl+C  # Le bot fermera proprement les positions

# Dans le terminal du moniteur
Ctrl+C
```

## 🧪 Exécution des Tests

1. **Tests Unitaires**
```bash
# Tous les tests
python -m pytest tests/

# Tests spécifiques
python -m pytest tests/test_risk_manager.py
python -m pytest tests/test_strategy.py

# Avec couverture
python -m pytest --cov=core tests/
```

2. **Tests de Stress**
```bash
# Tests de charge
python -m pytest tests/test_stress.py

# Tests de connexion
python -m pytest tests/test_stress.py -k "test_connection_loss"
```

3. **Backtests**
```bash
# Backtest simple
python backtest/run_backtest.py

# Backtest avec paramètres
python backtest/run_backtest.py --start-date 2023-01-01 --end-date 2023-12-31
```

## 📈 Ajout/Modification de Stratégies

1. **Créer une nouvelle stratégie**
```python
# strategies/my_strategy.py

from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        # Votre logique ici
        signals = []
        return signals
```

2. **Configurer la stratégie**
```json
// config/strategies.json
{
    "my_strategy": {
        "enabled": true,
        "params": {
            "param1": 10,
            "param2": 20
        }
    }
}
```

3. **Tester la stratégie**
```bash
# Créer les tests
touch tests/test_my_strategy.py

# Lancer les tests
python -m pytest tests/test_my_strategy.py
```

4. **Backtest**
```bash
# Ajouter aux backtests
python backtest/run_backtest.py --strategy my_strategy
```

## 📊 Logs et Résultats

1. **Structure des logs**
```
logs/
├── trading/               # Logs de trading
│   ├── YYYYMMDD.log      # Logs journaliers
│   └── errors.log        # Erreurs critiques
├── backtest/             # Résultats des backtests
│   └── YYYYMMDD_HHMMSS/  # Par run
└── monitoring/           # Logs du moniteur
```

2. **Sauvegarde des logs**
```bash
# Sauvegarde manuelle
cp -r logs/ backup/logs_YYYYMMDD/

# Les logs sont automatiquement archivés après 7 jours
```

3. **Analyse des résultats**
```bash
# Visualiser les résultats
python tools/analyze_results.py logs/backtest/YYYYMMDD_HHMMSS/

# Exporter en Excel
python tools/export_results.py --format excel
```

4. **Nettoyage**
```bash
# Nettoyer les vieux logs
python tools/clean_logs.py --older-than 30d

# Archiver les résultats
python tools/archive_results.py
```

## ⚠️ Points Importants

1. **Sécurité**
- Ne jamais commiter les credentials
- Toujours utiliser les fichiers .env
- Vérifier les permissions des fichiers de log

2. **Maintenance**
- Vérifier les logs quotidiennement
- Nettoyer les vieux fichiers régulièrement
- Sauvegarder la configuration

3. **Dépannage**
- Vérifier `logs/errors.log`
- Utiliser `--debug` pour plus de détails
- Consulter la documentation des composants

4. **Support**
- Documentation dans `docs/`
- Changelog dans `CHANGELOG.md`
- Tests comme exemples d'utilisation 