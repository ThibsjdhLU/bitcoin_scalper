from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import pyotp
import os
from typing import Optional
from starlette.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED
from bitcoin_scalper.core.config import SecureConfig
from bitcoin_scalper.core.risk_management import RiskManager
from bitcoin_scalper.connectors.mt5_rest_client import MT5RestClient
import threading

# Simuler stockage utilisateurs et secrets (à remplacer par DB/HashiCorp Vault en prod)
API_ADMIN_PASSWORD = os.environ.get("API_ADMIN_PASSWORD")
API_ADMIN_TOTP = os.environ.get("API_ADMIN_TOTP")
API_ADMIN_TOKEN = os.environ.get("API_ADMIN_TOKEN")

if not API_ADMIN_PASSWORD or len(API_ADMIN_PASSWORD) < 12:
    raise RuntimeError("API_ADMIN_PASSWORD manquant ou trop faible (>=12 caractères requis). Arrêt immédiat.")
if not API_ADMIN_TOTP or len(API_ADMIN_TOTP) < 16:
    raise RuntimeError("API_ADMIN_TOTP manquant ou trop faible (>=16 caractères requis). Arrêt immédiat.")
if not API_ADMIN_TOKEN or len(API_ADMIN_TOKEN) < 32:
    raise RuntimeError("API_ADMIN_TOKEN manquant ou trop faible (>=32 caractères requis). Arrêt immédiat.")

USERS = {
    "admin": {
        "password": API_ADMIN_PASSWORD,
        "totp_secret": API_ADMIN_TOTP,
        "token": API_ADMIN_TOKEN
    }
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
app = FastAPI(title="Bitcoin Scalper API", description="API sécurisée de supervision du bot.")

# --- MFA TOTP ---
def verify_totp(username: str, code: str) -> bool:
    user = USERS.get(username)
    if not user:
        return False
    totp = pyotp.TOTP(user["totp_secret"])
    return totp.verify(code)

# --- Auth dépendance ---
def get_current_user(token: str = Depends(oauth2_scheme)):
    for username, user in USERS.items():
        if user["token"] == token:
            return username
    raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Token invalide")

# --- Endpoints ---
class TOTPRequest(BaseModel):
    username: str
    code: str

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Identifiants invalides")
    return {"access_token": user["token"], "token_type": "bearer"}

@app.post("/verify")
def verify(request: TOTPRequest):
    if verify_totp(request.username, request.code):
        return {"status": "ok"}
    raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Code TOTP invalide")

# Singleton d'accès au bot (thread-safe)
class BotState:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_bot()
            return cls._instance

    def _init_bot(self):
        if os.environ.get("TEST_MODE") == "1":
            # Mock pour les tests unitaires
            class DummyMT5:
                def _request(self, *a, **kw):
                    return {"profit": 42.0}
                def get_positions(self):
                    return [{"symbol": "BTCUSD", "volume": 0.1}]
            self.config = object()
            self.mt5_client = DummyMT5()
            class DummyRisk:
                def get_risk_metrics(self):
                    return {"drawdown": 0.01, "daily_pnl": 100.0}
            self.risk = DummyRisk()
            return
        aes_key = os.environ.get("CONFIG_AES_KEY")
        if not aes_key:
            raise RuntimeError("Clé AES manquante : variable d'environnement CONFIG_AES_KEY obligatoire.")
        self.config = SecureConfig("config.enc", bytes.fromhex(aes_key))
        mt5_url = self.config.get("MT5_REST_URL")
        mt5_api_key = self.config.get("MT5_REST_API_KEY")
        self.mt5_client = MT5RestClient(mt5_url, api_key=mt5_api_key)
        self.risk = RiskManager(self.mt5_client)

    def get_pnl(self):
        try:
            account = self.mt5_client._request("GET", "/account")
            return account.get("profit", 0.0)
        except Exception as e:
            raise RuntimeError(f"Erreur récupération PnL : {e}")

    def get_positions(self):
        try:
            return self.mt5_client.get_positions()
        except Exception as e:
            raise RuntimeError(f"Erreur récupération positions : {e}")

    def get_alerts(self):
        # À adapter si un module d'alertes existe, sinon retourne vide
        return []

    def get_kpis(self):
        try:
            metrics = self.risk.get_risk_metrics()
            return metrics
        except Exception as e:
            raise RuntimeError(f"Erreur récupération KPIs : {e}")

bot_state = BotState()

@app.get("/pnl")
def get_pnl(username: str, code: str, user: str = Depends(get_current_user)):
    if not verify_totp(username, code):
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="MFA requis")
    try:
        pnl = bot_state.get_pnl()
        return {"pnl": pnl}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions")
def get_positions(username: str, code: str, user: str = Depends(get_current_user)):
    if not verify_totp(username, code):
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="MFA requis")
    try:
        positions = bot_state.get_positions()
        return {"positions": positions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts")
def get_alerts(username: str, code: str, user: str = Depends(get_current_user)):
    if not verify_totp(username, code):
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="MFA requis")
    try:
        alerts = bot_state.get_alerts()
        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/kpis")
def get_kpis(username: str, code: str, user: str = Depends(get_current_user)):
    if not verify_totp(username, code):
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="MFA requis")
    try:
        kpis = bot_state.get_kpis()
        return {"kpis": kpis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- OpenAPI et sécurité ---
@app.get("/healthz")
def healthz():
    return {"status": "ok"} 