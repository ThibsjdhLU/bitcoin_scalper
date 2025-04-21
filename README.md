# Bitcoin Scalper

Un bot de trading automatique pour le Bitcoin utilisant MetaTrader 5.

## Fonctionnalités

- Connexion automatique à MetaTrader 5
- Analyse technique en temps réel
- Trading automatique basé sur des stratégies personnalisables
- Interface graphique pour le suivi des performances
- Gestion des risques intégrée

## Prérequis

- Python 3.8 ou supérieur
- MetaTrader 5 installé et configuré
- Compte de trading actif

## Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/votre-username/bitcoin_scalper.git
cd bitcoin_scalper
```

2. Créez un environnement virtuel et activez-le :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Configurez vos variables d'environnement :
Créez un fichier `.env` à la racine du projet avec les informations suivantes :
```
MT5_LOGIN=votre_login
MT5_PASSWORD=votre_mot_de_passe
MT5_SERVER=votre_serveur
```

## Utilisation

1. Lancez le bot :
```bash
python src/main.py
```

2. L'interface graphique s'ouvrira automatiquement.

## Structure du projet

```
bitcoin_scalper/
├── src/
│   ├── main.py
│   ├── config/
│   ├── core/
│   ├── strategies/
│   └── ui/
├── tests/
├── requirements.txt
├── .env
└── README.md
```

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails. 