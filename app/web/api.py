from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
import pyotp
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
from random import random, randint
from datetime import datetime, timedelta

app = FastAPI(title="BTC Scalper Supervision API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simuler un stockage utilisateur/TOTP (à remplacer par stockage sécurisé en prod)
users_db = {
    "admin": {
        "username": "admin",
        "password": "adminpass",  # À hasher en prod
        "totp_secret": pyotp.random_base32(),
    }
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

class Token(BaseModel):
    access_token: str
    token_type: str

@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or form_data.password != user["password"]:
        raise HTTPException(status_code=400, detail="Identifiants invalides")
    # Générer un token simple (à remplacer par JWT en prod)
    token = user["username"] + "-token"
    return {"access_token": token, "token_type": "bearer"}

class TOTPVerifyRequest(BaseModel):
    username: str
    code: str

@app.post("/verify")
def verify_totp(req: TOTPVerifyRequest):
    user = users_db.get(req.username)
    if not user:
        raise HTTPException(status_code=400, detail="Utilisateur inconnu")
    totp = pyotp.TOTP(user["totp_secret"])
    if not totp.verify(req.code):
        raise HTTPException(status_code=401, detail="Code TOTP invalide")
    return {"status": "ok"}

# Exemple d'endpoint protégé MFA
@app.get("/secure-data")
def secure_data(token: str = Depends(oauth2_scheme), username: Optional[str] = None, code: Optional[str] = None):
    # Vérifier token (démonstration, à remplacer par JWT + TOTP en prod)
    if not token or not token.endswith("-token"):
        raise HTTPException(status_code=401, detail="Token invalide")
    # Vérifier TOTP (en prod, stocker session MFA validée)
    if not username or not code:
        raise HTTPException(status_code=401, detail="MFA requis")
    user = users_db.get(username)
    if not user:
        raise HTTPException(status_code=400, detail="Utilisateur inconnu")
    totp = pyotp.TOTP(user["totp_secret"])
    if not totp.verify(code):
        raise HTTPException(status_code=401, detail="Code TOTP invalide")
    return {"data": "Données sensibles accessibles avec MFA"}

@app.get("/status", tags=["Supervision"])
def get_status():
    """Retourne le statut de santé du bot et du backend."""
    return {"status": "ok", "message": "API supervision opérationnelle"}

# Métriques Prometheus
REQUEST_COUNT = Counter('api_requests_total', 'Nombre total de requêtes API', ['method', 'endpoint'])
ERROR_COUNT = Counter('api_errors_total', 'Nombre total d\'erreurs API', ['endpoint'])
RESPONSE_LATENCY = Histogram('api_response_latency_seconds', 'Latence des réponses API', ['endpoint'])
UPTIME = Gauge('api_uptime_seconds', 'Uptime de l\'API en secondes')
START_TIME = time.time()

@app.middleware("http")
async def prometheus_metrics_middleware(request, call_next):
    endpoint = request.url.path
    method = request.method
    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    start = time.time()
    try:
        response = await call_next(request)
        latency = time.time() - start
        RESPONSE_LATENCY.labels(endpoint=endpoint).observe(latency)
        return response
    except Exception:
        ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise

@app.get("/metrics")
def metrics():
    """Endpoint Prometheus pour exporter les métriques custom."""
    UPTIME.set(time.time() - START_TIME)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# --- Endpoints supervision pour dashboard Streamlit ---

def check_auth(token: str, username: Optional[str], code: Optional[str]):
    if not token or not token.endswith("-token"):
        raise HTTPException(status_code=401, detail="Token invalide")
    if not username or not code:
        raise HTTPException(status_code=401, detail="MFA requis")
    user = users_db.get(username)
    if not user:
        raise HTTPException(status_code=400, detail="Utilisateur inconnu")
    totp = pyotp.TOTP(user["totp_secret"])
    if not totp.verify(code):
        raise HTTPException(status_code=401, detail="Code TOTP invalide")

@app.get("/pnl")
def get_pnl(token: str = Depends(oauth2_scheme), username: Optional[str] = None, code: Optional[str] = None):
    check_auth(token, username, code)
    # Mock PnL et drawdown
    pnl = round(1000 * (random() - 0.5), 2)
    drawdown = round(10 * random(), 2)
    # Mock historique PnL
    now = datetime.utcnow()
    history = [{"date": (now - timedelta(days=i)).strftime("%Y-%m-%d"), "pnl": round(1000 * (random() - 0.5), 2)} for i in range(30)]
    return {"pnl": pnl, "drawdown": drawdown, "history": history[::-1]}

@app.get("/positions")
def get_positions(token: str = Depends(oauth2_scheme), username: Optional[str] = None, code: Optional[str] = None):
    check_auth(token, username, code)
    # Mock positions ouvertes
    positions = [
        {"symbol": "BTCUSD", "side": "long", "qty": 0.1, "entry": 42000.0, "unrealized": round(100 * (random() - 0.5), 2)},
        {"symbol": "BTCUSD", "side": "short", "qty": 0.05, "entry": 43000.0, "unrealized": round(50 * (random() - 0.5), 2)}
    ]
    return {"positions": positions}

@app.get("/alerts")
def get_alerts(token: str = Depends(oauth2_scheme), username: Optional[str] = None, code: Optional[str] = None):
    check_auth(token, username, code)
    # Mock alertes
    alerts = [
        {"level": "warning", "type": "drawdown", "message": "Drawdown > 5%"},
        {"level": "critical", "type": "latency", "message": "Latence API > 500ms"}
    ]
    return {"alerts": alerts}

@app.get("/kpis")
def get_kpis(token: str = Depends(oauth2_scheme), username: Optional[str] = None, code: Optional[str] = None):
    check_auth(token, username, code)
    # Mock KPIs
    kpis = {
        "sharpe": round(1.5 + random(), 2),
        "winrate": round(60 + 20 * random(), 2),
        "drawdown": round(10 * random(), 2),
        "latency_ms": randint(100, 500),
        "trades": randint(100, 500)
    }
    return kpis 