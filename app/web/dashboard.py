"""
Dashboard Streamlit supervision BTC Scalper :
- Authentification MFA (login + TOTP)
- Visualisation PnL, drawdown, historique
- Positions ouvertes
- Alertes actives
- KPIs
- Monitoring Prometheus
"""
import streamlit as st
import requests
import re
import time

st.set_page_config(page_title="BTC Scalper Dashboard", layout="wide")
st.title("Supervision BTC Scalper – Dashboard Monitoring")

# --- Authentification MFA ---
if "token" not in st.session_state:
    st.session_state["token"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "mfa_ok" not in st.session_state:
    st.session_state["mfa_ok"] = False

with st.sidebar:
    st.header("Connexion sécurisée (MFA)")
    if not st.session_state["token"]:
        username = st.text_input("Nom d'utilisateur", key="login_user")
        password = st.text_input("Mot de passe", type="password", key="login_pass")
        if st.button("Connexion"):
            try:
                resp = requests.post("http://web:8000/token", data={"username": username, "password": password})
                if resp.status_code == 200:
                    st.session_state["token"] = resp.json()["access_token"]
                    st.session_state["username"] = username
                    st.success("Connexion réussie. Entrez le code TOTP.")
                else:
                    st.error("Identifiants invalides.")
            except Exception as e:
                st.error(f"Erreur connexion : {e}")
    elif not st.session_state["mfa_ok"]:
        code = st.text_input("Code TOTP", key="totp_code")
        if st.button("Valider MFA"):
            try:
                resp = requests.post("http://web:8000/verify", json={"username": st.session_state["username"], "code": code})
                if resp.status_code == 200:
                    st.session_state["mfa_ok"] = True
                    st.success("MFA validé.")
                else:
                    st.error("Code TOTP invalide.")
            except Exception as e:
                st.error(f"Erreur MFA : {e}")
    else:
        st.success(f"Connecté : {st.session_state['username']}")
        if st.button("Déconnexion"):
            st.session_state["token"] = None
            st.session_state["username"] = None
            st.session_state["mfa_ok"] = False
            st.rerun()

# --- Si authentifié, afficher supervision ---
if st.session_state["token"] and st.session_state["mfa_ok"]:
    headers = {"Authorization": f"Bearer {st.session_state['token']}"}
    params = {"username": st.session_state["username"], "code": st.text_input("Code TOTP (rafraîchir)", value="", key="refresh_totp")}
    # PnL
    try:
        pnl = requests.get("http://web:8000/pnl", headers=headers, params=params, timeout=2).json()
        st.subheader("PnL & Drawdown")
        st.metric("PnL courant", f"{pnl['pnl']:.2f} $")
        st.metric("Drawdown", f"{pnl['drawdown']} %")
        st.line_chart({x['date']: x['pnl'] for x in pnl['history']})
    except Exception as e:
        st.error(f"Erreur récupération PnL : {e}")
    # Positions
    try:
        pos = requests.get("http://web:8000/positions", headers=headers, params=params, timeout=2).json()
        st.subheader("Positions ouvertes")
        st.table(pos["positions"])
    except Exception as e:
        st.error(f"Erreur récupération positions : {e}")
    # Alertes
    try:
        alerts = requests.get("http://web:8000/alerts", headers=headers, params=params, timeout=2).json()
        for alert in alerts["alerts"]:
            st.warning(f"[{alert['level'].upper()}] {alert['type']} : {alert['message']}")
    except Exception as e:
        st.error(f"Erreur récupération alertes : {e}")
    # KPIs
    try:
        kpis = requests.get("http://web:8000/kpis", headers=headers, params=params, timeout=2).json()
        st.subheader("KPIs")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Sharpe", kpis["sharpe"])
        col2.metric("Winrate", f"{kpis['winrate']} %")
        col3.metric("Drawdown", f"{kpis['drawdown']} %")
        col4.metric("Latence (ms)", kpis["latency_ms"])
        col5.metric("Trades", kpis["trades"])
    except Exception as e:
        st.error(f"Erreur récupération KPIs : {e}")
    st.info("Actualisation automatique toutes les 10s.")
    time.sleep(10)
    st.rerun()
else:
    st.warning("Veuillez vous connecter avec MFA pour accéder à la supervision.")

# Fonction utilitaire pour parser les métriques Prometheus

def get_metric(metrics_text, metric_name):
    pattern = rf"^{metric_name}{{1}}\s+([0-9\.eE\+-]+)"
    match = re.search(pattern, metrics_text, re.MULTILINE)
    return float(match.group(1)) if match else None

# Récupérer les métriques du bot
try:
    bot_metrics = requests.get("http://localhost:8001/metrics", timeout=2).text
    api_metrics = requests.get("http://localhost:8000/metrics", timeout=2).text
except Exception as e:
    st.error(f"Erreur accès métriques Prometheus : {e}")
    bot_metrics = ""
    api_metrics = ""

st.header("Métriques Bot")
col1, col2, col3 = st.columns(3)
col1.metric("Uptime (s)", get_metric(bot_metrics, "bot_uptime_seconds"))
col2.metric("Cycles", get_metric(bot_metrics, "bot_cycles_total"))
col3.metric("Erreurs", get_metric(bot_metrics, "bot_errors_total"))

st.header("Métriques API")
col4, col5 = st.columns(2)
col4.metric("Uptime (s)", get_metric(api_metrics, "api_uptime_seconds"))
col5.metric("Requêtes", get_metric(api_metrics, "api_requests_total"))

st.info("Actualisation automatique toutes les 10s.")
st.rerun()

# Placeholder pour les prochains widgets (PnL, positions, alertes, KPIs)
st.header("Aperçu rapide")
st.write("Dashboard en construction…") 