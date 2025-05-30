Web API (FastAPI)
==================

.. automodule:: bitcoin_scalper.web.api
    :members:
    :undoc-members:
    :show-inheritance:

Endpoints principaux
--------------------

- ``POST /token`` : Authentification, retourne un token d'accès
- ``POST /verify`` : Vérification MFA (TOTP)
- ``GET /pnl`` : Récupère le PnL (protégé MFA + token)
- ``GET /positions`` : Récupère les positions (protégé MFA + token)
- ``GET /alerts`` : Récupère les alertes (protégé MFA + token)
- ``GET /kpis`` : Récupère les KPIs (protégé MFA + token)
- ``GET /healthz`` : Healthcheck

Sécurité
--------

- Authentification OAuth2 (token)
- MFA obligatoire (TOTP)
- Aucun secret exposé dans les réponses

Sécurité des secrets
-------------------

- Les variables d'environnement suivantes sont **obligatoires** pour démarrer l'API :
  - ``API_ADMIN_PASSWORD`` (>=12 caractères)
  - ``API_ADMIN_TOTP`` (>=16 caractères)
  - ``API_ADMIN_TOKEN`` (>=32 caractères)
- Aucun secret ne doit être codé en dur ni avoir de valeur par défaut.
- L'API refusera de démarrer si un secret est absent ou trop faible.

Exemple d'utilisation
---------------------

.. code-block:: bash

    curl -X POST "http://localhost:8000/token" -d "username=admin&password=admin"
    curl -X POST "http://localhost:8000/verify" -H "Content-Type: application/json" -d '{"username": "admin", "code": "123456"}'
    curl -H "Authorization: Bearer <token>" "http://localhost:8000/pnl?username=admin&code=123456" 

Connexion au cœur métier
------------------------

Depuis la version X.X.X, tous les endpoints critiques (/pnl, /positions, /alerts, /kpis) sont connectés en temps réel au cœur du bot Bitcoin Scalper :

- Les données retournées proviennent directement du RiskManager, du connecteur MT5 et des modules internes.
- Toute erreur d'accès au cœur métier (ex : indisponibilité du broker, problème de configuration) est remontée avec un code HTTP 500 et un message explicite.
- Aucun fallback factice n'est utilisé : si une donnée n'est pas disponible, une erreur explicite est retournée.

Gestion des erreurs et robustesse
----------------------------------

- Tous les endpoints critiques sont protégés contre les erreurs inattendues (try/except, logs, HTTP 500 explicite).
- Les accès aux secrets et à la configuration sont strictement contrôlés (aucun secret en clair, vérification de la présence des variables d'environnement).
- En cas d'erreur de configuration ou d'indisponibilité d'un service externe, l'API retourne une erreur explicite et ne divulgue aucune information sensible. 