# Bitcoin Scalper

Bot de trading algorithmique BTC/USD intégrant pipeline Machine Learning, gestion avancée du risque, supervision sécurisée (API FastAPI, dashboard PyQt), monitoring Prometheus, et versioning des données avec DVC.

## Fonctionnalités principales

- **Sécurité avancée** :
  - Authentification forte (MFA TOTP) sur l'API et l'interface de supervision
  - Chiffrement des secrets (AES-256, dérivation PBKDF2)
  - Vérification automatique du chiffrement disque (FileVault) et du pare-feu (scripts shell)
  - Bonnes pratiques : aucun secret en clair, endpoints critiques protégés
- **Monitoring & alertes** :
  - Export Prometheus (bot & API) : uptime, erreurs, cycles, PnL, drawdown, latence
  - Intégration Grafana (panels recommandés : uptime, erreurs, PnL, drawdown)
  - Alertes Prometheus Alertmanager (email, Telegram, webhook)
- **Supervision** :
  - API FastAPI sécurisée (endpoints : `/pnl`, `/positions`, `/alerts`, `/kpis`, `/token`, `/verify`)
  - Dashboard PyQt (statut, PnL, capital, positions, alertes, graphique capital)
- **Pipeline ML & DVC** :
  - Ingestion, nettoyage, feature engineering, modélisation, backtesting, exécution
  - Versioning datasets, features, modèles et artefacts ML avec DVC

## Structure du projet

```
bitcoin_scalper/
├── README.md
├── pyproject.toml / requirements.txt
├── Makefile
├── .gitignore / .coveragerc
│
├── docs/                # Documentation Sphinx
├── docker/              # Dockerfile, docker-compose.yml
├── k8s/                 # Manifestes Kubernetes
│
├── bitcoin_scalper/     # Code source principal
│   ├── core/            # Ingestion, ML, risk, DVC, etc.
│   ├── web/             # API FastAPI
│   └── main.py          # Orchestration du bot
├── bot/                 # Connecteurs externes (MT5, etc.)
├── data/                # Données (non versionnées)
├── scripts/             # Scripts d'admin, migration, audit
├── tests/               # Tests unitaires et d'intégration
├── ui/                  # Dashboard PyQt
└── .github/             # CI/CD
```

## Installation & Prérequis

- Python 3.11.x recommandé
- Cloner le repo puis :

```sh
python3 -m venv .venv
source .venv/bin/activate
make init
```

- Pour la doc :

```sh
make docs
```

## Utilisation

- **Entraîner le modèle ML** :
  ```sh
  python train.py
  ```
  ou
  ```sh
  make train
  ```
  Le modèle utilise automatiquement le fichier CSV dans `/data/BTCUSD_M1_202301010000_202512011647.csv`.
  
  Pour plus de détails, consultez [README_TRAINING.md](README_TRAINING.md).

- **Lancer le bot principal** :
  ```sh
  python -m bitcoin_scalper.main
  ```
- **Lancer l'API FastAPI** :
  ```sh
  uvicorn bitcoin_scalper.web.api:app --reload
  ```
- **Dashboard PyQt** :
  Lancement automatique avec le bot, ou via les modules `ui/`.

- **Exemple de requête API protégée** :
  ```sh
  curl -X GET "http://localhost:8000/pnl?username=admin&code=123456" -H "Authorization: Bearer <token>"
  ```

## Tests & Qualité

- **Lancer les tests** :
  ```sh
  make test
  ```
- **Vérifier la couverture** :
  Couverture >95% exigée (voir rapport pytest-cov)
- **Lint** :
  ```sh
  make lint
  ```
- **Générer la documentation** :
  ```sh
  make docs
  ```

## Contribution

- Respecter les standards Cursor :
  - PEP8, docstrings, typage, sécurité (jamais de secrets en clair)
  - Ajouter des tests unitaires pour toute nouvelle fonctionnalité (>95% coverage)
  - Documenter chaque action significative dans `user_rules_history.md`
  - Nettoyer les fichiers/dossiers inutiles avant chaque PR

## Nettoyage & bonnes pratiques

- Les fichiers/dossiers suivants sont exclus du dépôt :
  - `__pycache__/`, `.pytest_cache/`, `.DS_Store`, `.coverage`, logs, fichiers temporaires, artefacts de build, données brutes
- Les dossiers vides ou non utilisés sont supprimés régulièrement
- Scripts d'audit sécurité :
  - `bash scripts/check_filevault.sh`
  - `bash scripts/check_firewall.sh`

## Références & contacts

- Documentation : dossier `docs/`
- CI/CD : `.github/workflows/`
- Pour toute question, ouvrir une issue ou contacter l'auteur.

## Sécurité opérationnelle : gestion et rotation des secrets

- **Ne jamais versionner config.enc ou tout fichier *.enc** (voir .gitignore)
- Pour changer la clé de chiffrement ou le mot de passe :
  1. Utiliser le script `scripts/migrate_config_to_password.py` pour rechiffrer la config avec un nouveau mot de passe.
  2. Mettre à jour la variable d'environnement `CONFIG_AES_KEY` (ou le mot de passe utilisé pour la dérivation) sur tous les environnements.
  3. Supprimer toute ancienne clé ou mot de passe stocké en local ou dans des scripts.
- **Rotation des tokens API** :
  - Modifier régulièrement les variables d'environnement `API_ADMIN_PASSWORD`, `API_ADMIN_TOTP`, `API_ADMIN_TOKEN`.
  - Redémarrer les services après chaque rotation.
- **Audit sécurité** :
  - Exécuter régulièrement `bash scripts/check_filevault.sh` et `bash scripts/check_firewall.sh`.
  - Documenter toute action de sécurité dans `user_rules_history.md`. 