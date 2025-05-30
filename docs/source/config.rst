config module
=============

.. automodule:: config
   :members:
   :show-inheritance:
   :undoc-members:

Gestion sécurisée de la configuration
====================================

- Le fichier de configuration **doit** être chiffré (AES-256) et stocké sous ``config.enc``.
- La variable d'environnement ``CONFIG_AES_KEY`` (clé hexadécimale 64 caractères) est **obligatoire** pour tout lancement du bot.
- **Aucun fallback** sur ``config_clear.json`` n'est autorisé en production.
- En cas d'absence de la clé AES, le bot s'arrête immédiatement avec une erreur explicite.
- Il est recommandé d'injecter la clé via un gestionnaire de secrets (Vault, AWS, Azure).

Exemple d'utilisation :

.. code-block:: bash

   export CONFIG_AES_KEY="0123456789abcdef..."
   python bitcoin_scalper/main.py

Sécurité :
----------
- Ne jamais stocker la clé AES en clair dans le code, ni dans les fichiers de configuration.
- Ne jamais versionner ``config_clear.json`` ou tout fichier contenant des secrets non chiffrés.
