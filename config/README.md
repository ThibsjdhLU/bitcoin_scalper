# Configuration du Bot de Trading

Ce dossier contient les fichiers de configuration pour le bot de trading.

## Configuration des Notifications (`notifier_config.json`)

### Configuration Email
```json
{
    "email": {
        "smtp_server": "smtp.gmail.com",  // Serveur SMTP à utiliser
        "smtp_port": 587,                 // Port du serveur SMTP
        "sender_email": "",               // Email d'envoi des alertes
        "sender_password": "",            // Mot de passe d'application Gmail
        "recipient_email": ""             // Email de réception des alertes
    }
}
```

#### Configuration Gmail
Pour utiliser Gmail comme serveur SMTP :
1. Activer la validation en deux étapes sur votre compte Gmail
2. Générer un mot de passe d'application :
   - Aller dans les paramètres du compte Google
   - Sécurité > Validation en deux étapes > Mots de passe d'application
   - Créer un nouveau mot de passe d'application pour "Bot de Trading"
3. Utiliser ce mot de passe dans la configuration (`sender_password`)

### Configuration des Alertes
```json
{
    "alerts": {
        "signal": {
            "enabled": true,              // Activer/désactiver les alertes de signal
            "min_strength": 0.5           // Force minimale du signal pour envoyer une alerte (0-1)
        },
        "order": {
            "enabled": true,              // Activer/désactiver les alertes d'ordre
            "min_volume": 0.01            // Volume minimum pour envoyer une alerte
        },
        "risk": {
            "enabled": true,              // Activer/désactiver les alertes de risque
            "drawdown_threshold": 10.0,    // Seuil de drawdown en pourcentage
            "daily_loss_threshold": 5.0    // Seuil de perte journalière en pourcentage
        }
    }
}
```

### Types d'Alertes

1. **Alertes de Signal** (`signal`)
   - Envoyées lorsqu'un signal de trading est détecté
   - Inclut les détails du signal (symbole, type, prix, indicateurs)
   - La force minimale permet de filtrer les signaux faibles

2. **Alertes d'Ordre** (`order`)
   - Envoyées lors de l'exécution d'un ordre
   - Inclut les détails de l'ordre (type, prix, volume, statut)
   - Le volume minimum permet de filtrer les petits trades

3. **Alertes de Risque** (`risk`)
   - Envoyées lorsque les seuils de risque sont dépassés
   - Drawdown : pourcentage de perte par rapport au plus haut
   - Perte journalière : pourcentage de perte sur la journée

### Format des Messages

Les messages sont formatés de manière claire et incluent :
- Emoji pour une identification rapide du type d'alerte
- Horodatage précis
- Données pertinentes selon le type d'alerte
- Métadonnées additionnelles si disponibles

## Configuration de l'Optimisation (`optimizer_config.json`)

### Méthodes d'Optimisation

1. **Grid Search**
   ```json
   {
       "grid_search": {
           "enabled": true,
           "param_grid": {
               "strategy_name": {
                   "param1": [val1, val2, val3],
                   "param2": [val1, val2, val3]
               }
           }
       }
   }
   ```
   - Recherche exhaustive sur une grille de paramètres
   - Teste toutes les combinaisons possibles
   - Plus précis mais plus lent que les autres méthodes

2. **Random Search**
   ```json
   {
       "random_search": {
           "enabled": true,
           "n_iter": 100,
           "param_ranges": {
               "strategy_name": {
                   "param1": [min, max],
                   "param2": [min, max]
               }
           }
       }
   }
   ```
   - Recherche aléatoire dans l'espace des paramètres
   - Plus rapide que Grid Search
   - Bon compromis entre exploration et temps de calcul

3. **Differential Evolution**
   ```json
   {
       "differential_evolution": {
           "enabled": true,
           "max_iter": 100,
           "popsize": 15,
           "mutation": 0.8,
           "recombination": 0.7
       }
   }
   ```
   - Algorithme évolutionnaire pour l'optimisation globale
   - Très efficace pour les espaces de paramètres complexes
   - Converge généralement vers de meilleures solutions

### Configuration ML

1. **Modèles Disponibles**
   ```json
   {
       "models": {
           "random_forest": {
               "enabled": true,
               "params": {
                   "n_estimators": 100,
                   "max_depth": 10
               }
           }
       }
   }
   ```
   - Random Forest : robuste et facile à interpréter
   - XGBoost : performances élevées, nécessite plus de tuning
   - LightGBM : rapide et efficace en mémoire
   - SVM : bon pour les données non linéaires

2. **Features**
   ```json
   {
       "features": {
           "price": ["returns", "log_returns"],
           "moving_averages": [5, 10, 20, 50],
           "volatility": [5, 10, 20],
           "volume": [5, 20],
           "momentum": ["rsi", "macd"]
       }
   }
   ```
   - Prix : rendements et log-rendements
   - Moyennes mobiles : différentes périodes
   - Volatilité : fenêtres glissantes
   - Volume : moyennes mobiles
   - Momentum : indicateurs techniques

3. **Prédiction**
   ```json
   {
       "prediction": {
           "feature_window": 20,
           "prediction_window": 5,
           "signal_threshold": 0.6
       }
   }
   ```
   - `feature_window` : nombre de bougies pour les features
   - `prediction_window` : horizon de prédiction
   - `signal_threshold` : seuil de confiance pour les signaux

4. **Entraînement**
   ```json
   {
       "training": {
           "train_size": 0.8,
           "shuffle": false,
           "cv_splits": 5
       }
   }
   ```
   - `train_size` : proportion des données d'entraînement
   - `shuffle` : mélanger les données (false pour séries temporelles)
   - `cv_splits` : nombre de folds pour la validation croisée 