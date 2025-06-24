.. _installation_windows:

Installation du projet Bitcoin Scalper sous Windows
==================================================

Cette fiche détaille les étapes pour installer et exécuter Bitcoin Scalper sur un poste Windows. 

Prérequis système
-----------------
- Windows 10 ou 11 (64 bits)
- Accès administrateur
- Connexion internet stable
- Espace disque suffisant (>10 Go recommandé)

Prérequis logiciels
-------------------
- Python 3.11.x (https://www.python.org/downloads/windows/)
- Git (https://git-scm.com/download/win)
- Visual Studio Build Tools (pour compiler certains paquets Python, optionnel mais recommandé)
- (Optionnel) WSL2 + Ubuntu pour une compatibilité accrue avec certains scripts

Étapes d'installation
---------------------
1. **Cloner le dépôt**

   Ouvrir l'invite de commandes (cmd) ou PowerShell :

   .. code-block:: bat

      git clone https://github.com/<votre_organisation>/bitcoin_scalper.git
      cd bitcoin_scalper

2. **Créer un environnement virtuel Python**

   .. code-block:: bat

      python -m venv .venv
      .venv\Scripts\activate

3. **Installer les dépendances**

   .. code-block:: bat

      pip install --upgrade pip
      pip install -r requirements.txt

   Ou, si `make` est disponible (via WSL ou Git Bash) :

   .. code-block:: bash

      make init

4. **Configurer les secrets**

   - Placer le fichier `config.enc` fourni par l'administrateur à la racine du projet.
   - Définir la variable d'environnement `CONFIG_AES_KEY` (ou le mot de passe de déchiffrement) :

     .. code-block:: bat

        set CONFIG_AES_KEY=VotreMotDePasseUltraSecret

   - Ne jamais versionner ni exposer ce mot de passe !

5. **Lancer le bot**

   .. code-block:: bat

      python -m bitcoin_scalper.main

6. **Lancer l'API FastAPI**

   .. code-block:: bat

      .venv\Scripts\activate
      uvicorn bitcoin_scalper.web.api:app --reload

7. **Lancer les tests**

   .. code-block:: bat

      .venv\Scripts\activate
      pytest --maxfail=1 --disable-warnings --cov=bitcoin_scalper

8. **Générer la documentation**

   .. code-block:: bat

      .venv\Scripts\activate
      make docs

Notes et conseils
-----------------
- Pour les dépendances nécessitant une compilation (ex : `cryptography`, `pandas`), installer les Build Tools : https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Pour l'utilisation de DVC, installer DVC via pip (`pip install dvc[all]`) et configurer le remote si besoin.
- Les scripts bash peuvent être adaptés en PowerShell ou exécutés via WSL.
- Respecter les bonnes pratiques de sécurité : ne jamais exposer de secrets, activer le chiffrement disque BitLocker.

Support
-------
- Pour toute question, ouvrir une issue sur le dépôt ou contacter l'auteur. 