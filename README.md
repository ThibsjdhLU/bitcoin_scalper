# Bitcoin Scalper

Un outil de backtesting pour les stratégies de trading sur le Bitcoin, avec une interface utilisateur interactive.

## Fonctionnalités

- Backtesting de stratégies de trading
- Interface utilisateur avec Streamlit
- Visualisation des résultats avec Plotly
- Sauvegarde et chargement des résultats
- Stratégies disponibles :
  - Bollinger Bands Reversal

## Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/votre-username/bitcoin-scalper.git
cd bitcoin-scalper
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Installer le package en mode développement :
```bash
pip install -e .
```

## Structure du projet

```
bitcoin-scalper/
├── data/                    # Données de marché et résultats
├── src/
│   ├── strategies/         # Stratégies de trading
│   ├── services/          # Services (backtest, stockage)
│   ├── utils/             # Utilitaires
│   └── ui/                # Interface utilisateur
├── tests/                 # Tests unitaires
├── setup.py              # Configuration du package
└── requirements.txt      # Dépendances
```

## Utilisation

1. Lancer l'interface utilisateur :
```bash
streamlit run src/ui/app.py
```

2. Dans l'interface :
   - Sélectionner une stratégie
   - Configurer les paramètres
   - Charger les données
   - Lancer le backtest
   - Visualiser les résultats

## Tests

Pour lancer les tests :
```bash
python -m pytest tests/
```

## Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit les changements (`git commit -am 'Ajout d'une nouvelle fonctionnalité'`)
4. Push la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
