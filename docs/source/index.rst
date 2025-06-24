.. bitcoin_scalper documentation master file, created by
   sphinx-quickstart on Mon May 26 21:45:30 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

bitcoin_scalper documentation
=============================

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   web_api

.. toctree::
   :maxdepth: 2
   :caption: Modules

   config
   data_cleaner
   data_ingestor
   dvc_manager
   feature_engineering
   ml_pipeline
   order_algos
   risk_management
   timescaledb_client
   backtesting

Sécurité et gestion des secrets
==============================

- **Aucun secret ne doit être codé en dur ni avoir de valeur par défaut.**
- Tous les secrets (API_ADMIN_PASSWORD, API_ADMIN_TOTP, API_ADMIN_TOKEN, CONFIG_AES_KEY, etc.) doivent être injectés via variables d'environnement ou gestionnaire de secrets.
- Le bot et l'API refusent de démarrer si un secret est absent ou trop faible.
- Voir la section ``config`` pour la gestion sécurisée de la configuration.

.. toctree::
   :maxdepth=2
   :caption: Sommaire

   installation_windows
