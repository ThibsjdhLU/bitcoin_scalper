# Historique des actions IA et contributeurs

Ce fichier consigne chaque action majeure réalisée sur le projet (création, modification, suppression, refactoring, ajout de test, évolution de règle, etc.), par l’IA ou un contributeur humain.

## Format attendu
[Date] – [Auteur] : [Résumé de l’action et de sa raison]

---

### Exemples :
2024-06-10 – IA : Création du fichier d’historique pour assurer la traçabilité des actions et la cohérence du projet. 

- 2024-06-07 : Nettoyage massif du projet. Suppression de nombreux fichiers et dossiers inutiles : logs, caches, fichiers temporaires, données volumineuses, fichiers de configuration obsolètes, fichiers système (.DS_Store), dossiers de build Sphinx, etc. Voir l'historique des suppressions pour le détail. Conformité renforcée avec les standards Cursor.

- Complétion des TODO critiques dans main.py : instanciation effective de DataIngestor, TimescaleDBClient, DVCManager, Backtester, algos d’ordre ; ajout d’exemples d’utilisation de add_features, multi_timeframe, exécution avancée d’ordres ; intégration du reporting/backtest offline ; ajout de métriques Prometheus avancées ; intégration de la logique DVC (add, commit, push, pull). Cette action vise à rendre le pipeline principal opérationnel et traçable, conformément aux standards du projet.

- Finalisation de tous les TODO restants dans main.py : initialisation effective des algos d’exécution avancée (iceberg, TWAP, VWAP), du backtester, enrichissement du pipeline avec add_features et multi_timeframe, intégration de la logique de chargement des features/signaux/reporting dans run_backtest, intégration des scripts d’audit et export métriques dans run_audit, gestion complète DVC/datasets dans run_data. Cette action clôture la phase d’implémentation des points critiques du pipeline principal.

- Création d'un test unitaire pour utils/settings.py afin d'assurer la robustesse du chargement de configuration, la gestion des erreurs et la couverture du signal settings_reloaded.

- Ajout de docstrings et d'annotations de type dans utils/settings.py pour conformité PEP8 et respect des standards de documentation du projet.

- Ajout de docstrings et d'annotations de type dans utils/logger.py pour conformité PEP8 et respect des standards de documentation du projet.

- Ajout de tests unitaires avancés sur utils/logger.py et utils/settings.py pour valider les comportements métier, la robustesse, la sécurité (gestion d'erreur, thread-safety, signaux Qt, persistance) et garantir une couverture >95% pertinente.

- Purge complète de l'historique Git des fichiers volumineux (.pkl, .csv) avec BFG Repo-Cleaner, mise à jour du .gitignore, et synchronisation forcée avec le dépôt GitHub pour respecter les limites de taille et garantir la conformité du projet.

- Création d'un notebook Google Colab prêt à l'emploi pour entraîner le pipeline ML du projet Bitcoin Scalper, incluant montage Google Drive, installation des dépendances, préparation des features, entraînement LightGBM, sauvegarde du modèle et évaluation rapide, afin de faciliter l'entraînement distant et reproductible dans le respect des standards de sécurité et de structure du projet.

- Ajout d'un combleur automatique de trous temporels (réindexation 1min + ffill) dans orchestrator.py, activable via une option CLI, afin de garantir la continuité temporelle des données minute pour le pipeline ML.

- Correction du pipeline ML dans orchestrator.py : ajout de la création automatique de la colonne <CLOSE> et du calcul de log_return_1m dans df_feat avant le labeling, pour garantir la compatibilité avec la fonction de génération des labels et éviter les erreurs bloquantes.

- Correction du calcul de log_return_1m dans la pipeline, garantie de sa présence et de celle de 1min_log_return dans tous les splits (train/val/test), et ajout d'un test unitaire pour valider cette cohérence, afin d'assurer la fiabilité du calcul du PnL et du labeling.

- Correction de la fonction de génération des labels dans labeling.py : la recherche des colonnes <CLOSE> et log_return_1m accepte désormais aussi les variantes préfixées (1min_<CLOSE>, 1min_log_return, etc.), pour éviter les erreurs lors du pipeline ML multi-timeframe.

- 2024-06-10 – IA : Lancement du plan d'amélioration systémique du pipeline ML Bitcoin Scalper : harmonisation du feature engineering (log_return, indicateurs), ajout du balancing avancé (SMOTE), tuning multi-algo (LightGBM, XGBoost, DNN), validation croisée temporelle, backtest réaliste (PnL, Sharpe, drawdown, frais, slippage), intégration temps réel/simulation, renforcement de la sécurité, du monitoring et de l'audit, avec documentation et tests automatisés à chaque étape.

- 2024-06-10 – IA : Intégration de SMOTE (balancing avancé) dans le pipeline ML Bitcoin Scalper, avec génération de tests unitaires et documentation pour garantir la robustesse face à l'imbalance des classes.

- 2024-06-10 – IA : Ajout du support XGBoost et DNN (PyTorch/Keras) dans le module de modeling, extension du tuning d'hyperparamètres, génération de tests unitaires et documentation pour garantir la flexibilité et la performance du pipeline ML.

- 2024-06-10 – IA : Ajout/complétion de la validation croisée temporelle (TimeSeriesSplit) dans le module splitting, avec tests unitaires et documentation pour garantir l'absence de fuite de données et la robustesse de l'évaluation ML.

- 2024-06-10 – IA : Ajout du slippage et du reporting out-of-sample dans le module de backtest, avec tests unitaires et documentation pour garantir un backtest réaliste et une évaluation robuste des stratégies ML.

- 2024-06-10 – IA : Finalisation du simulateur d'ordres et automatisation de la prise de décision dans le pipeline ML, avec génération de tests unitaires et documentation pour garantir la robustesse et la traçabilité des exécutions simulées ou réelles.

- 2024-06-10 – IA : Renforcement du logging, monitoring et audit dans tout le pipeline (sécurité, alertes, auditabilité), avec documentation et tests pour garantir la traçabilité, la robustesse et la conformité aux standards de sécurité du projet.

- 2024-06-11 – IA : Ajout d’indicateurs techniques avancés (Keltner, Donchian, Chandelier Exit, Ulcer Index, MFI, OBV, Accumulation/Distribution, Chaikin, TSI, CCI, Williams %R, StochRSI, Ultimate Oscillator, ADX, DMI, Ichimoku, Parabolic SAR, PPO, ROC) dans le module de feature engineering pour renforcer la robustesse du pipeline ML et préparer l’étape 1.1 de la roadmap.

- 2024-06-11 – IA : Ajout de features de contexte (z-score multi-fenêtres, distance à des bornes, encodage temporel enrichi, position relative) dans add_features du module de feature engineering, pour l’étape 1.2 de la roadmap.

- 2024-06-11 – IA : Ajout d’une fonction d’analyse d’importance des features (gain, split, SHAP) avec export automatique de rapports PNG/HTML dans modeling.py, pour l’étape 1.3 de la roadmap.

- 2024-06-11 – IA : Ajout d’une fonction de sélection automatique des features par importance (gain/split ou SHAP) dans modeling.py, pour l’étape 1.4 de la roadmap.

- 2024-06-11 – IA : Ajout de la génération automatique de labels multi-horizon (target_5m, target_10m, target_15m, target_30m) dans labeling.py, pour l’étape 2.1 de la roadmap.

- 2024-06-11 – IA : Ajout du support des différents types de seuils (rolling std, quantile, spread+frais) dans generate_labels et generate_multi_horizon_labels, pour l’étape 2.2 de la roadmap.

- 2024-06-11 – IA : Ajout du mode actionnable (gain net après frais/slippage/spread) dans generate_labels et generate_multi_horizon_labels, pour l’étape 2.3 de la roadmap.

- 2024-06-11 – IA : Ajout d’une fonction d’analyse de la distribution des labels (PNG, JSON, alerte déséquilibre) dans labeling.py, pour l’étape 2.4 de la roadmap.

- 2024-06-11 – IA : Ajout de la méthode de correction avancée des outliers, trous temporels et erreurs de marché (winsorization, interpolation, réindexation, rapport JSON) dans DataCleaner pour l’étape 3.2 de la roadmap.

- 2024-06-11 – IA : Ajout de la méthode de vérification de la cohérence temporelle (look-ahead, doublons, désordre, gaps, timestamps futurs, rapport JSON) dans DataCleaner pour l’étape 3.3 de la roadmap.

- 2024-06-11 – IA : Implémentation de la fonction d’audit global de la qualité des données (pipeline complet, rapports JSON, logs), ajout d’un test unitaire, documentation Sphinx/MkDocs, et script CLI d’audit, pour l’étape 3.4 de la roadmap d’amélioration du pipeline ML Bitcoin Scalper.

- 2024-06-11 – IA : Ajout de la fonction de split temporel robuste (temporal_train_val_test_split), split par dates ou proportions, exclusion multi-horizon, génération de rapport JSON, test unitaire, documentation Sphinx/MkDocs, pour la phase 4.1 de la roadmap d’amélioration du pipeline ML Bitcoin Scalper.

- 2024-06-11 – IA : Ajout des fonctions de génération automatique des folds (TimeSeriesSplit, PurgedKFold), génération de rapport JSON détaillé, tests unitaires, documentation Sphinx/MkDocs, pour la phase 4.2 de la roadmap d’amélioration du pipeline ML Bitcoin Scalper.

- 2024-06-11 – IA : Ajout de la fonction d’orchestration du pipeline ML (run_ml_pipeline), split robuste, folds, entraînement LightGBM/XGBoost, reporting automatique (JSON, PNG, CSV), test unitaire end-to-end, documentation Sphinx/MkDocs, pour la phase 4.3 de la roadmap d’amélioration du pipeline ML Bitcoin Scalper.

- 2024-06-11 – IA : Ajout de la fonction de tuning avancé (tune_model_hyperparams), support grid/random/optuna, early stopping, reporting automatique (JSON, CSV, PNG), test unitaire, documentation Sphinx/MkDocs, pour la phase 5 de la roadmap d’amélioration du pipeline ML Bitcoin Scalper.

- 2024-06-11 – IA : Ajout de la classe d’exécution temps réel/simulation (RealTimeExecutor), support live/replay, gestion portefeuille, reporting automatique (CSV, PNG), test unitaire, documentation Sphinx/MkDocs, pour la phase 7 de la roadmap d’amélioration du pipeline ML Bitcoin Scalper.

- Intégration du scheduler adaptatif et du sizing dynamique dans `RealTimeExecutor` (`realtime.py`), permettant l'exécution adaptative en temps réel avec gestion du risque et sizing basé sur la confiance du modèle.

- Extension du module de labeling pour supporter les labels à 5 classes, la gestion avancée des neutres (nan, drop), le calcul d'un score de confiance kNN, et ajout des tests unitaires associés pour garantir la robustesse et la flexibilité du pipeline de labeling.

- Ajout de la sélection de features par permutation importance et VIF, création des tests unitaires associés, et installation de la dépendance statsmodels pour garantir la robustesse et la non-redondance des features dans la pipeline ML.

- Ajout de la dépendance CatBoost (pip install catboost) pour remplacer LightGBM dans les modules de modélisation, tuning et orchestration du projet.

- Migration du module bitcoin_scalper/core/modeling.py de LightGBM vers CatBoost (imports, tuning, entraînement, prédiction, gestion des features catégorielles), pour une meilleure gestion des corrélations internes et la robustesse du pipeline ML.

- Migration du pipeline principal (bitcoin_scalper/core/ml_orchestrator.py) de LightGBM vers CatBoost : remplacement des imports, adaptation de l'entraînement, du reporting, et ajout du support natif des features catégorielles, pour renforcer la robustesse et la cohérence du pipeline ML.

- Migration du module de tuning (bitcoin_scalper/core/tuning.py) de LightGBM vers CatBoost : adaptation du tuning (grid/random/optuna), de l'entraînement, du reporting et de la gestion des features catégorielles, pour garantir la cohérence et la performance du pipeline ML.

- Migration du module d'export (bitcoin_scalper/core/export.py) de LightGBM vers CatBoost : adaptation de la logique de sauvegarde/chargement, des signatures, des docstrings et des tests associés, pour garantir la compatibilité et la robustesse de l'inférence et du déploiement des modèles ML.

- Migration de tous les tests unitaires principaux (tests/core/) de LightGBM vers CatBoost : adaptation de la création, de l'entraînement, de la sauvegarde et du chargement des modèles dans les tests, pour garantir la robustesse et la cohérence de la validation du pipeline ML.

- 2024-06-12 – IA : Création du module d'environnement RL Gym (BitcoinScalperEnv) pour apprentissage par renforcement, ajout de la génération de Q-values (expected return net) pour chaque action, et création des tests unitaires associés (rl_env, Q-values), première étape de la migration du pipeline ML vers la régression Q-value et l'intégration RL, conformément à la roadmap et aux standards du projet.

- 2024-06-12 – IA : Adaptation du pipeline principal (orchestrator.py) pour supporter la régression Q-value (expected return net) en option, avec génération des Q-values, split, entraînement multi-sortie (CatBoostRegressor), et évaluation (RMSE, MAE), tout en conservant le mode classification directionnelle, conformément à la roadmap et aux standards du projet.

- Refactoring : transformation de la fonction run_backtest en classe Backtester (avec méthode run) dans core/backtesting.py, pour corriger l'import, préparer l'injection de coûts dynamiques et contraintes, et permettre l'extension future du backtest (benchmark, réalisme, etc.).

- Adaptation : modification de la méthode run de Backtester pour retourner (out_df, trades, kpis), assurant la compatibilité avec les tests unitaires existants et la rétrocompatibilité de l'API de backtest.

- Extension : ajout des paramètres de coûts dynamiques (slippage_fn, spread_series, fee_fn, orderbook_series) à la classe Backtester pour permettre l'injection de slippage, spread et frais adaptatifs dans le backtest, première étape du backtest réaliste.

- Implémentation : adaptation de la méthode run de Backtester pour appliquer la latence (latency_fn) et le rejet d'ordre (reject_fn) à chaque trade, avec reporting des ordres rejetés dans un fichier CSV dédié.

- Extension : ajout des hooks latency_fn et reject_fn à la classe Backtester pour simuler la latence d'exécution et le rejet d'ordre dans le backtest, conformément aux contraintes réalistes de l'étape 6.

- Extension : ajout du support des benchmarks naïfs (buy-and-hold, RSI2, etc.) dans la classe Backtester, avec calcul automatique des KPIs et rapport comparatif dans benchmarks.json.

- Création des modules `probability_calibration.py` (calibration des probabilités par Platt scaling et isotonic regression) et `trade_decision_filter.py` (filtrage dynamique des décisions de trade) dans `bitcoin_scalper/core/`, avec tests unitaires (>95% couverture) et documentation Sphinx/MkDocs, pour fiabiliser la post-prédiction et la prise de décision du bot selon les standards sécurité et qualité du projet.

- Création du module `adaptive_scheduler.py` pour centraliser le scheduling adaptatif, le sizing dynamique (Kelly/VaR/confiance ML) et la gestion du risque, conformément à l'étape 8 du plan d'exécution.

- Ajout de la fonction `execute_adaptive_trade` dans `order_execution.py`, permettant d'exécuter des ordres via le scheduler adaptatif, avec sizing dynamique (Kelly/VaR/confiance ML) et gestion du risque intégrée.

- Intégration du scheduler adaptatif et du sizing dynamique dans `Backtester` (`backtesting.py`), permettant l'exécution adaptative en backtest avec gestion du risque et sizing basé sur la confiance du modèle.

- Ajout de tests unitaires pour le scheduler adaptatif et l'exécution adaptative dans `test_order_execution.py`, `test_backtesting.py` et `test_trade_decision_filter.py`, validant le filtrage, le sizing dynamique et la gestion du risque.

- 2024-06-12 – IA : Ajout des métriques post-trade avancées (expectancy, taux de gain, profit factor, max losing streak) dans le module de backtest, et amélioration du logging/fallback automatique en cas de features manquantes en mode backtest ou live, pour renforcer l'auditabilité et la robustesse du bot Bitcoin Scalper.

- Création du module Python `MetaStackingClassifier` (métamodèle de stacking cross-horizon) dans `models/`, avec tests unitaires (>95% couverture) et documentation, pour agréger les prédictions multi-horizon selon les standards PEP8, sécurité et intégration projet.

- Intégration centralisée de tous les modules ML (tuning, backtest, RL, stacking, hybrid) dans ml_orchestrator.py et extension de la CLI dans orchestrator.py pour permettre le choix dynamique du pipeline, afin de garantir une orchestration modulaire et cohérente du projet.

- Ajout d'un test unitaire dans test_labeling.py pour garantir la robustesse de la détection des colonnes de prix et de log_return (préfixées ou non) dans le labeling.
- Génération de la documentation Sphinx/MkDocs après correction du pipeline.
- Correction des warnings de type (casting explicite) dans le feature engineering pour un pipeline plus propre.
- Ajout d'un contrôle explicite dans ml_orchestrator.py pour détecter et logger les colonnes non 1D dans les features avant l'entraînement ML, afin de diagnostiquer et corriger l'erreur 'Per-column arrays must each be 1-dimensional'.
- Ajout d'un diagnostic détaillé dans ml_orchestrator.py : log du type et des exemples de valeurs de chaque colonne de features, détection explicite des colonnes dtype=object ou contenant des listes/arrays, et levée d'une erreur explicite si problème avant l'entraînement ML.
- Ajout d'un diagnostic maximal et d'un filtrage automatique des colonnes non numériques dans ml_orchestrator.py : log des dtypes, détection et log des colonnes non numériques, filtrage automatique des features non numériques avant l'entraînement, log de la liste finale des features utilisées.
- Automatisation de la suppression des colonnes ayant >10% de NaN dans le train avant l'entraînement ML, avec log de la liste des colonnes supprimées et des features finales conservées.
- Automatisation du décalage du début du train à la première date où toutes les features sont valides (pas de NaN), avec log de la date de début effective et du nombre de lignes ignorées, dans ml_orchestrator.py.
- Création d'une fiche d'installation détaillée pour le projet Bitcoin Scalper sous Windows (docs/source/installation_windows.rst) et ajout à la documentation Sphinx, afin de faciliter l'onboarding des utilisateurs Windows et garantir la portabilité du projet.

- Modification de l'URL du serveur MT5 dans tous les modules clients (export_mt5_btc_history.py, trading_worker.py, main.py, web/api.py, core/data_ingestor.py) pour pointer vers http://192.168.1.157:8000, afin d'assurer la connexion correcte au serveur REST MT5 sur le réseau local.

- Correction du mapping des clés dans PositionsModel (models/positions_model.py) pour correspondre aux champs réels retournés par MT5 (ticket, symbol, volume, price_open, type) et ajout d'une gestion d'erreur pour éviter les KeyError. Raison : garantir la compatibilité avec la structure des positions et éviter les plantages lors de l'affichage des positions dans l'UI.