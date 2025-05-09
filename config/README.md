# Configuration du Bitcoin Scalper

Ce dossier contient la configuration centralisée pour le système Bitcoin Scalper.

## Configuration unifiée

Tous les paramètres de configuration ont été centralisés dans un fichier unique: `unified_config.py`.

Cette approche apporte plusieurs avantages:
- Une source unique de vérité pour la configuration
- Une interface cohérente et simple pour accéder aux paramètres
- Un système de configuration évolutif et maintenable

## Utilisation

Pour utiliser la configuration unifiée dans vos modules:

```python
from config.unified_config import config

# Accéder aux valeurs
login = config.get("exchange.login")
demo_mode = config.get("trading.demo_mode", False)  # Valeur par défaut si non trouvée

# Modifier une valeur
config.set("trading.risk_per_trade", 2.0)
config.save()  # Sauvegarder les modifications

# Charger depuis les variables d'environnement
config.load_env()

# Réinitialiser aux valeurs par défaut
config.reset_to_default()
```

## Structure de la configuration

La configuration est organisée en sections thématiques:

- `broker`: Paramètres de connexion aux courtiers (MT5)
- `api`: Configuration des API externes
- `risk`: Paramètres de gestion des risques
- `strategies`: Configuration des stratégies de trading
- `trading`: Paramètres généraux de trading
- `logging`: Configuration des journaux
- `backtest`: Paramètres pour les backtests
- `optimization`: Configuration de l'optimisation des stratégies
- `indicators`: Paramètres des indicateurs techniques
- `ml_strategy`: Configuration des stratégies basées sur l'apprentissage automatique
- `notifications`: Configuration des notifications (email, Telegram)
- `exchange`: Paramètres de connexion à la bourse
- `interface`: Configuration de l'interface utilisateur

## Anciens fichiers

Les anciens fichiers de configuration ont été déplacés dans le dossier `archive/` pour référence. Veuillez ne plus les utiliser directement. 