### Roadmap : Implémentation complète de l'entraînement ML (`--mode ml`)

Cette roadmap suit les étapes logiques d'un pipeline ML, de la donnée au modèle versionné et évaluable.

**Phase 1 : Préparation des données et Structure de base**

1.  **Clarification du Problème ML**
    *   **Tâche :** Définir précisément ce que le modèle ML doit prédire (le "label `y`"). S'agit-il de prédire le mouvement de prix futur ? Un signal d'achat/vente/garde basé sur une autre logique ? La probabilité d'un événement ?
    *   **Livrable :** Documentation interne ou commentaire clair dans le code spécifiant le label cible et sa source/méthode de calcul.
    *   **Dépendance :** Nécessite de savoir comment les labels sont ou seront générés dans les données de features (`data/features/BTCUSD_M1.csv`).

2.  **Identification et Chargement des Features/Labels dans `run_ml`**
    *   **Tâche :** Compléter le `TODO` #3 dans `bitcoin_scalper/main.py`. Identifier les colonnes qui seront utilisées comme features (`X`) et la colonne pour le label (`y`) dans le DataFrame `df` chargé. Supprimer les colonnes non pertinentes (timestamp, etc.).
    *   **Livrable :** Code fonctionnel pour séparer `df` en `features` (DataFrame pandas) et `labels` (Series pandas).
    *   **Dépendance :** Dépend de la structure et du contenu final du fichier `data/features/BTCUSD_M1.csv`.

3.  **Séparation Temporelle Train/Validation/Test dans `run_ml`**
    *   **Tâche :** Compléter le `TODO` #4 dans `bitcoin_scalper/main.py`. Implémenter une séparation des données en ensembles d'entraînement, de validation et éventuellement de test **basée sur l'ordre chronologique** (`iloc`). **Crucial pour éviter le *look-ahead bias* dans les séries temporelles.** Ne jamais mélanger les données avant le split.
    *   **Livrable :** Variables `X_train`, `y_train`, `X_val`, `y_val` (et `X_test`, `y_test` si un ensemble de test séparé est utilisé) contenant les données correctement séparées.
    *   **Dépendance :** Dépend du `TODO` précédent (avoir les DataFrames `features` et `labels`).

**Phase 2 : Développement et Intégration du Pipeline ML**

4.  **Finalisation de la Méthode `fit` dans `MLPipeline`**
    *   **Tâche :** Revoir et compléter l'implémentation de la méthode `fit` dans `bitcoin_scalper/core/ml_pipeline.py` pour le(s) type(s) de modèle(s) choisi(s) (ex: "random_forest"). S'assurer qu'elle utilise `X_train`, `y_train` pour l'entraînement et qu'elle puisse optionnellement évaluer sur `X_val`, `y_val` pour le suivi pendant l'entraînement ou l'arrêt anticipé (pour les modèles PyTorch).
    *   **Livrable :** Méthode `fit` fonctionnelle dans `MLPipeline` qui entraîne le modèle interne (`self.model`) et retourne des métriques d'entraînement/validation.
    *   **Dépendance :** Dépend de la définition des features/labels et du split.

5.  **Intégration de l'Entraînement dans `run_ml`**
    *   **Tâche :** Compléter le `TODO` #6 dans `bitcoin_scalper/main.py`. Appeler la méthode `ml_pipe.fit()` en lui passant les ensembles d'entraînement et de validation (`X_train`, `y_train`, `X_val`, `y_val`) et les paramètres d'entraînement nécessaires.
    *   **Livrable :** Ligne de code qui lance l'entraînement et capture les métriques retournées.
    *   **Dépendance :** Dépend des `TODO` 3 et 4 (données préparées, `fit` implémentée).

6.  **Implémentation et Intégration de l'Évaluation du Modèle**
    *   **Tâche :** Compléter le `TODO` #7 dans `bitcoin_scalper/main.py`. Utiliser la méthode `ml_pipe.predict` ou `ml_pipe.predict_proba` sur l'ensemble de validation (`X_val`). Calculer des métriques pertinentes pour un problème de classification (Accuracy, F1-score, AUC, Precision, Recall). **Important :** Utiliser des métriques adaptées aux séries temporelles et aux données déséquilibrées si c'est le cas.
    *   **Tâche :** Ajouter les imports nécessaires (`from sklearn.metrics import ...`).
    *   **Livrable :** Code qui calcule et logue les métriques de performance sur l'ensemble de validation.
    *   **Dépendance :** Dépend du `TODO` 5 (modèle entraîné) et de la méthode `predict` dans `MLPipeline`.

7.  **Implémentation et Intégration de la Sauvegarde/Chargement du Modèle**
    *   **Tâche :** Compléter le `TODO` #8 dans `bitcoin_scalper/main.py`. Assurer que les méthodes `save` et `load` dans `MLPipeline` fonctionnent correctement pour le type de modèle choisi (ex: `joblib.dump`/`joblib.load` pour Scikit-learn/XGBoost, `torch.save`/`torch.load` pour PyTorch). Gérer les chemins de fichier.
    *   **Tâche :** Mettre à jour la fonction `run_live_trading` pour utiliser le chemin de sauvegarde/chargement défini dans `run_ml`.
    *   **Livrable :** Modèle sauvegardé sur disque après entraînement et code de chargement correspondant mis à jour dans `run_live_trading`.
    *   **Dépendance :** Dépend du `TODO` 5 (modèle entraîné).

**Phase 3 : Industrialisation, Tuning et Explicabilité**

8.  **Intégration du Versioning DVC**
    *   **Tâche :** Compléter le `TODO` #9 dans `bitcoin_scalper/main.py`. Instancier `DVCManager` et utiliser ses méthodes (`add`, `commit`, `push`) pour versionner le fichier du modèle sauvegardé et le pipeline de données associé (le fichier features).
    *   **Livrable :** Le modèle entraîné est tracké par DVC, permettant de le lier aux données et au code qui l'ont généré.
    *   **Dépendance :** DVC doit être initialisé dans le projet (`dvc init`) et un remote doit être configuré (`dvc remote add`).

9.  **Implémentation du Tuning Hyperparamètres**
    *   **Tâche :** Compléter le `TODO` #10 (partie tuning) dans `bitcoin_scalper/main.py`. Utiliser `GridSearchCV` (Scikit-learn) ou `Optuna` pour trouver les meilleurs hyperparamètres pour le modèle choisi en évaluant sur l'ensemble de validation. Cela pourrait nécessiter d'ajouter une méthode `tune` dans `MLPipeline`.
    *   **Livrable :** Code qui effectue une recherche d'hyperparamètres et sélectionne les meilleurs avant l'entraînement final.
    *   **Dépendance :** Dépend des `TODO` 3 et 4 (données préparées, méthode `fit` robuste).

10. **Implémentation de l'Explicabilité (SHAP/LIME)**
    *   **Tâche :** Compléter le `TODO` #10 (partie explicabilité) dans `bitcoin_scalper/main.py`. Utiliser SHAP ou LIME pour comprendre les features les plus importantes pour les prédictions du modèle. Cela pourrait nécessiter d'ajouter une méthode `explain` dans `MLPipeline`.
    *   **Livrable :** Code qui génère des visualisations ou des rapports d'explicabilité (Feature importance, SHAP values).
    *   **Dépendance :** Dépend du `TODO` 5 (modèle entraîné).

**Phase 4 : Tests et Documentation**

11. **Tests Unitaires pour les Nouveaux Composants**
    *   **Tâche :** Créer des tests unitaires pour la fonction `run_ml` (en mockant les dépendances externes comme la lecture de fichier, DVC, MLPipeline) et pour les nouvelles méthodes/classes ajoutées ou modifiées dans `MLPipeline`. 