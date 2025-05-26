# Module de supervision web

Ce module fournit :
- Un backend FastAPI pour exposer les métriques de supervision, PnL, positions, alertes, KPIs…
- Un dashboard Streamlit pour la visualisation temps réel
- Des tests unitaires (>95% à terme)

## Lancer l’API FastAPI
```sh
uvicorn app.web.api:app --reload
```

## Lancer le dashboard Streamlit
```sh
streamlit run app/web/dashboard.py
```

## Tester l’API
```sh
pytest app/web/test_api.py
```

## Endpoints disponibles
- `GET /status` : statut de santé du backend
- `POST /token` : authentification utilisateur (login)
- `POST /verify` : vérification MFA (TOTP)
- `GET /pnl` : PnL courant, drawdown, historique (auth MFA)
- `GET /positions` : positions ouvertes (auth MFA)
- `GET /alerts` : alertes actives (auth MFA)
- `GET /kpis` : KPIs de trading (auth MFA)
- `GET /metrics` : métriques Prometheus pour monitoring

## Utilisation du dashboard

Lancer le dashboard Streamlit :
```sh
streamlit run app/web/dashboard.py
```

- Connexion MFA requise (login + code TOTP)
- Visualisation temps réel : PnL, drawdown, positions, alertes, KPIs, métriques Prometheus

## Roadmap
- Ajout endpoints PnL, positions, alertes, KPIs
- Widgets Streamlit dynamiques
- Authentification, alertes, notifications 