# 🚀 Roadmap – Bot de trading BTCUSD (macOS ↔ Windows MT5)

---

## 1️⃣ Environnement & Sécurité

- **Python ≥ 3.11** (Poetry)
- **Docker** & **Kubernetes**
- **Sécurité** :
  - Secrets chiffrés **AES-256**
  - Stockage sécurisé (Keychain, variables d'environnement)
  - **MFA (TOTP)** sur API et dashboard
  - Chiffrement disque (**FileVault**), pare-feu, audit
  - **Aucune fuite de secrets** (logs/API)

---

## 2️⃣ Intégration MT5 (serveur Windows)

- MT5 + REST server (**FastAPI**, Uvicorn)
- API sécurisée : `/ticks`, `/ohlcv`, `/order`, `/account`, `/symbol`
- Client REST multiplateforme (macOS)
- Conteneurisation & K8s
- **Tests unitaires complets**

---

## 3️⃣ Pipeline Data

- Ingestion temps réel (**DataIngestor**)
- Nettoyage (**DataCleaner**) : outliers, NA, anomalies (Isolation Forest)
- Stockage **TimescaleDB** (schéma optimisé)
- Versioning via **DVC**
- **Tests unitaires** par étape

---

## 4️⃣ Feature Engineering

- Indicateurs techniques (vectorisés) : RSI, MACD, EMA, etc.
- Support **multi-timeframes**
- Features dérivées (retours, volatilité, ratios…)
- Module modulaire **FeatureEngineering**
- **Tests unitaires complets**

---

## 5️⃣ ML & Deep Learning

- **MLPipeline** : RandomForest, XGBoost, DNN, LSTM, Transformer, CNN1D
- Split, cross-validation, tuning (**GridSearch**, **Optuna**)
- Explicabilité : **SHAP** (LIME à compléter)
- Versioning modèles (**DVC**)
- **Tests unitaires**

---

## 6️⃣ Risk Management

- **RiskManager** : drawdown, perte journalière, sizing dynamique
- Calcul stop loss, tick value, PnL, equity, peak balance
- Exposition des métriques
- **Tests unitaires**

---

## 7️⃣ Exécution des Ordres

- Envoi REST (**send_order**)
- Robustesse réseau, gestion erreurs
- Simulation locale (MT5 Python package)
- **Tests unitaires**

---

## 8️⃣ Backtesting

- **Backtester vectorisé**
  - Simulation historique (3+ ans)
  - KPIs : Sharpe, drawdown, winrate, profit factor
  - Support stratégies multiples, sizing dynamique
- **Tests unitaires**

---

## 9️⃣ Monitoring & Supervision

- **Prometheus** (metrics)
- **Grafana** (visualisation)
- **Streamlit dashboard** :
  - KPIs, PnL, drawdown, positions, alertes
  - Authentification MFA
- **Alertmanager** (email, Telegram, webhook)
- **Tests de supervision**

---

## 🔟 Sécurité Avancée

- **MFA généralisé (TOTP)**
- Chiffrement **AES-256**, gestion centralisée des secrets
- Scripts d'audit (disque, firewall)
- **Tests sécurité CI/CD**

---

## 1️⃣1️⃣ Documentation & Qualité

- Génération auto (**Sphinx/MkDocs**)
- README, docstrings, exemples API
- Linting (**ruff**, PEP8)
- Couverture tests >95% (**pytest + coverage**)

---

## 1️⃣2️⃣ CI/CD

- **GitHub Actions** : lint, tests, coverage, build, déploiement
- Déploiement progressif, rollback automatique
- Vérification reproductibilité (**DVC**)

---

## 🗺️ Schéma de l'Architecture

```
[Bot macOS Python]
      |
 REST API
      v
[Windows MT5 Server (FastAPI + MT5)]
      |
   [MT5 Terminal]

[Data] <--> [TimescaleDB, DVC]
     |
[MLPipeline] <--> [Backtester] <--> [FeatureEngineering]
     |
[RiskManager]
     |
[OrderExecution REST → MT5]

[FastAPI Web API] <--> [Dashboard Streamlit]
     |
[Prometheus] <--> [Grafana + Alertmanager]
```

---

## 🔄 Boucles Logiques

- **Trading** : ingestion → signal → risque → exécution → métriques
- **Supervision** : Prometheus → Grafana/alertes
- **CI/CD** : test → build → déploiement → rollback
- **Données** : ingestion → nettoyage → stockage → versioning

---

## 📊 Avancement par module

| Module                | État        | Tests | Doc   | Sécu | CI/CD | Monitor. |
|-----------------------|-------------|-------|-------|------|-------|----------|
| Connexion MT5 REST    | Terminé     | Oui   | Oui   | Oui  | Oui   | Oui      |
| Pipeline data         | Terminé     | Oui   | Oui   | Oui  | Oui   | Oui      |
| Feature engineering   | Terminé     | Oui   | Oui   | Oui  | Oui   | Oui      |
| ML pipeline           | À renforcer | Oui   | Part  | Oui  | Oui   | Oui      |
| Risk management       | Terminé     | Oui   | Oui   | Oui  | Oui   | Oui      |
| Backtesting           | Terminé     | Oui   | Oui   | Oui  | Oui   | Oui      |
| Monitoring            | À finaliser | Oui   | Oui   | Oui  | Oui   | Partiel  |
| Sécurité avancée      | À auditer   | Oui   | Oui   | À rev| Oui   | Oui      |
| Documentation         | Incomplète  | Oui   | Part  | Oui  | Oui   | Oui      |
| CI/CD                 | Terminé     | Oui   | Oui   | Oui  | Oui   | Oui      |

---

## 🎯 Objectifs stratégiques

### Vision

- Référence open-source trading BTC/USD (macOS ↔ Windows MT5)
- Robuste, sécurisé, performant, auditable
- Cycle automatisé data → modèle → exécution → monitoring
- Support expérimentations IA (AutoML, LLM, RL)

### Priorisation

**Quick Wins** :
- Tests +95%
- MFA/API sécurisés
- Monitoring Prometheus/Grafana de base

**Must Have** :
- Robustesse ingestion
- Backtesting vectorisé
- CI/CD complet, rollback auto

**Nice to Have** :
- Auto-ML (Optuna, HPO)
- Monitoring prédictif
- Dashboard explicabilité

---

## ✅ Actions assignables

- Vérification version Python (`pyproject.toml`)
- Audit secrets (AES-256, Keychain)
- Vérification MFA actif (API, dashboard)
- Génération doc auto (Sphinx/MkDocs)
- Rapport coverage tests >95%
- Reconnexion robuste (DataIngestor, MT5 REST)
- Configurer Alertmanager (email, Telegram, webhook)
- Rollback CI/CD auto
- Vérifier indexation TimescaleDB
- Compléter SHAP, LIME
- Exemples usage API

---

## 🏁 Critères de Validation

- Tests automatisés >95%
- Documentation API générée
- Sécurité (MFA, chiffrement, secrets)
- CI/CD automatisé (tests, rollback)
- Monitoring + alertes configurées
- Exemples concrets + guides API

---

## 💡 Axes d'Innovation

- Auto-ML, HPO, LLM pour feature selection
- Monitoring prédictif (drift, anomalies)
- Sécurité Zero Trust (Vault, audit continu)
- Explicabilité avancée (SHAP/LIME + dashboard)
- Orchestration intelligente (blue/green, multi-cloud)

---

## ⚠️ Risques & Contre-mesures

| Risque                | Mitigation                                 |
|-----------------------|--------------------------------------------|
| Fuite de secrets      | Rotation, audit, logs chiffrés             |
| Panne ingestion/data  | Reconnexion, backups, alertes              |
| Drift modèle          | Monitoring, retrain, alertes               |
| Faille MFA/API        | MFA obligatoire, pentest, logs             |
| CI/CD défaillant      | Rollback automatique, tests bloquants      |

---