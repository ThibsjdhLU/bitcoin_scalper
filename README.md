# Bitcoin Scalper

Bot de trading algorithmique pour le scalping de Bitcoin sur MetaTrader 5.

## Installation

```bash
pip install -e .
```

## Configuration

1. Installer MetaTrader 5
2. Configurer les variables d'environnement dans le fichier `.env`:
   ```
   MT5_LOGIN=votre_login
   MT5_PASSWORD=votre_mot_de_passe
   MT5_SERVER=votre_serveur
   ```

## Utilisation

```bash
python main.py
```

## Fonctionnalités

- Interface graphique avec PySide6
- Connexion à MetaTrader 5
- Indicateurs techniques
- Gestion des risques
- Backtesting

## Licence

MIT License 