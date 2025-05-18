# Bitcoin Trading Bot

Bot de trading Bitcoin avec interface utilisateur NiceGUI.

## Fonctionnalités

- Interface utilisateur moderne et réactive avec NiceGUI
- Connexion en temps réel à MetaTrader 5
- Graphiques de prix en temps réel
- Gestion des positions et des ordres
- Suivi du PnL et des statistiques
- Console de logs en direct
- Support de plusieurs stratégies de trading

## Installation

1. Cloner le repository :
```bash
git clone https://github.com/votre-username/bitcoin-trading-bot.git
cd bitcoin-trading-bot
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Configurer MetaTrader 5 :
- Installer MetaTrader 5
- Configurer les identifiants dans le fichier `.env`

## Utilisation

1. Lancer l'application :
```bash
python app/main.py
```

2. Ouvrir votre navigateur à l'adresse : `http://localhost:8080`

3. Utiliser l'interface pour :
- Démarrer/Arrêter le bot
- Sélectionner une stratégie
- Ajuster les paramètres
- Surveiller les performances

## Structure du Projet

```
bitcoin-trading-bot/
├── app/
│   ├── core/
│   │   ├── app_state.py
│   │   └── refresh_manager.py
│   ├── services/
│   │   └── trading_service.py
│   ├── ui/
│   │   └── components.py
│   └── main.py
├── config/
│   └── unified_config.py
├── logs/
├── requirements.txt
└── README.md
```

## Développement

### Architecture

- `app/core/` : Gestion de l'état et des tâches en arrière-plan
- `app/services/` : Services de trading et de données
- `app/ui/` : Composants d'interface utilisateur
- `config/` : Configuration de l'application

### Ajouter une Nouvelle Stratégie

1. Créer une nouvelle classe de stratégie dans `app/strategies/`
2. Implémenter les méthodes requises
3. Ajouter la stratégie dans le sélecteur de l'interface

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## Licence

MIT
