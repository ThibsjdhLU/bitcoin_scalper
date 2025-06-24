.. _installation_mac:

Installation du projet Bitcoin Scalper sur MacOS
===============================================

Cette fiche détaille l'installation complète du projet sur un Mac récent (Apple Silicon ou Intel).

Prérequis système
-----------------
- MacOS 13+ (Ventura ou ultérieur recommandé)
- Xcode Command Line Tools :
  ``xcode-select --install``
- Homebrew (gestionnaire de paquets) :
  ``/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"``
- Python 3.11.x (via Homebrew ou pyenv recommandé)
- Git

Installation étape par étape
---------------------------
1. **Cloner le dépôt**
   ```sh
   git clone <url_du_repo>
   cd bitcoin_scalper
   ```
2. **Créer l'environnement virtuel Python**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Installer les dépendances**
   ```sh
   make init
   # ou
   pip install -r requirements.txt
   ```
4. **Initialiser la configuration chiffrée**
   - Placer le fichier `config.enc` fourni (ne jamais versionner de secrets !)
   - Définir la variable d'environnement `CONFIG_AES_KEY` (voir README)
   - Déchiffrer la config si besoin :
     ```sh
     python decrypt_config.py
     ```
5. **Lancer le bot**
   ```sh
   python -m bitcoin_scalper.main
   ```
6. **Lancer l'API**
   ```sh
   uvicorn bitcoin_scalper.web.api:app --reload
   ```
7. **Lancer le dashboard PyQt**
   - Automatique avec le bot, ou via les modules `ui/`.

Tests et qualité
----------------
- Lancer tous les tests unitaires :
  ```sh
  make test
  ```
- Vérifier la couverture (>95%) :
  ```sh
  make coverage
  ```
- Lint PEP8 :
  ```sh
  make lint
  ```

Documentation
-------------
- Générer la doc Sphinx :
  ```sh
  make docs
  open docs/_build/html/index.html
  ```

Conteneurisation (Docker)
-------------------------
- Construire l'image :
  ```sh
  docker build -t bitcoin_scalper .
  ```
- Lancer avec docker-compose :
  ```sh
  docker-compose up
  ```

Orchestration (Kubernetes)
--------------------------
- Appliquer les manifestes :
  ```sh
  kubectl apply -f k8s/
  ```

Sécurité et gestion des secrets
-------------------------------
- **Jamais de secrets en clair** : utiliser `config.enc` et variables d'environnement.
- Rotation régulière des secrets (voir README).
- Scripts d'audit sécurité :
  ```sh
  bash scripts/check_filevault.sh
  bash scripts/check_firewall.sh
  ```

Bonnes pratiques et dépannage
----------------------------
- Toujours activer l'environnement virtuel `.venv` avant toute commande Python.
- En cas d'erreur de compilation de paquets Python, vérifier que Xcode CLI Tools et Homebrew sont à jour.
- Pour les problèmes de dépendances, relancer `make init` ou `pip install -r requirements.txt`.
- Pour toute question, consulter le README ou ouvrir une issue. 