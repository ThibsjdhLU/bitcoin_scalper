[![CI](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml/badge.svg)](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml)
[![Coverage Status](https://img.shields.io/badge/coverage-auto-important)](https://github.com/<OWNER>/<REPO>/actions)

## Sécurité avancée

- **MFA (TOTP)** : Authentification forte sur l'API (FastAPI) et dashboard (Streamlit à venir). Endpoints `/token`, `/verify`, `/secure-data`.
- **Chiffrement disque** : FileVault obligatoire sur macOS. Vérification via `scripts/check_filevault.sh`.
- **Pare-feu applicatif** : Activation obligatoire, vérification via `scripts/check_firewall.sh`.
- **Tests d'intégration sécurité** : Automatisés dans `tests/security/test_api_security.py` (pytest, couverture MFA, secrets, accès protégé).
- **Bonnes pratiques** : Aucun secret exposé dans les logs ou réponses, authentification forte obligatoire sur endpoints critiques.

**Scripts d'audit sécurité** :
- `bash scripts/check_filevault.sh`
- `bash scripts/check_firewall.sh`

## Monitoring & alertes

- **Export Prometheus** :
  - Bot : `/metrics` sur port 8001 (uptime, cycles, erreurs)
  - API web : `/metrics` (requêtes, erreurs, latence, uptime)
- **Dashboard Grafana** :
  - Connecter à Prometheus (voir `prometheus.yml`)
  - Panels recommandés : uptime, erreurs, latence, cycles, requêtes
- **Dashboard Streamlit** :
  - Visualisation temps réel des métriques bot/API (voir `app/web/dashboard.py`)
- **Alertes Prometheus Alertmanager** :
  - À configurer pour email, Telegram, webhook (voir doc Prometheus)
  - Exemple : `ALERT BotErrorsHigh IF increase(bot_errors_total[5m]) > 0`
- **Bonnes pratiques** :
  - Exporter toutes les métriques critiques (uptime, erreurs, latence, cycles, requêtes)
  - Surveiller en continu, alertes multi-canal

**Références** :
- `prometheus.yml`, `docker-compose.yml`, `app/web/dashboard.py`

## Interface web supervision

- **API FastAPI** :
  - Endpoints : `/pnl`, `/positions`, `/alerts`, `/kpis` (tous protégés MFA)
  - Authentification forte obligatoire (token + TOTP)
- **Dashboard Streamlit** :
  - Widgets PnL, drawdown, historique, positions, alertes, KPIs
  - Login MFA, rafraîchissement auto, gestion erreurs, responsive
- **Exemple d'utilisation (API)** :
  - `curl -X GET "http://localhost:8000/pnl?username=admin&code=123456" -H "Authorization: Bearer <token>"`
- **Exemple d'utilisation (Streamlit)** :
  - `streamlit run app/web/dashboard.py`
- **Bonnes pratiques** :
  - MFA obligatoire pour toute supervision
  - Ne jamais exposer de secrets dans les réponses

**Références** :
- `app/web/api.py`, `app/web/dashboard.py`

## DVC pipeline (versioning data & ML)

Ce projet utilise DVC pour le versioning des datasets, features, modèles et artefacts ML.
- Gestion centralisée via `app/core/dvc_manager.py` (API Python)
- Automatisation CLI via `scripts/dvc_utils.py`
- Intégration CI/CD : chaque PR vérifie le statut DVC et la reproductibilité pipeline

Voir [app/core/README.md](app/core/README.md) pour la documentation détaillée et des exemples d'usage.

## Nettoyage et structure

Ce projet applique une politique stricte de nettoyage et d'exclusion des fichiers/dossiers inutiles :
- Les caches Python (`__pycache__`), environnements virtuels (`.venv/`), logs, fichiers temporaires et dossiers d'indexation sont exclus du dépôt (voir `.gitignore` et `.dvcignore`).
- Les dossiers vides ou non utilisés sont supprimés régulièrement.
- Les scripts d'administration sont séparés des modules métiers.
- Les fichiers de configuration par défaut sont adaptés ou supprimés s'ils ne sont pas utilisés.

Pour toute contribution, merci de respecter ces règles afin de garantir la maintenabilité et la sécurité du projet.

# bitcoin_scalper

## Structure du projet

```
bitcoin_scalper/
│
├── README.md
├── LICENSE
├── setup.py / pyproject.toml
├── requirements.txt
├── .gitignore
├── .coveragerc
├── Makefile
│
├── docs/                # Documentation Sphinx/MkDocs
├── docker/              # Dockerfile, docker-compose.yml
├── k8s/                 # Manifestes Kubernetes
│
├── bitcoin_scalper/     # Code source principal (anciennement app/)
│   ├── core/
│   ├── web/
│   └── ...
├── bot/
├── data/
├── scripts/
├── tests/               # Tests unitaires et d'intégration
└── .github/
```

## Commandes utiles

- `make init` : Installer les dépendances
- `make test` : Lancer les tests avec couverture
- `make lint` : Linter le code
- `make docs` : Générer la documentation

## Contribution

Merci de respecter PEP8, d'ajouter des tests unitaires (>95% coverage) et de ne jamais exposer de secrets en clair. 