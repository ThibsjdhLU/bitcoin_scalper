import os
# Mocks pour les variables d'environnement et secrets (doit être fait AVANT tout import de l'app)
os.environ["API_ADMIN_PASSWORD"] = "SuperSecretPassword123"
os.environ["API_ADMIN_TOTP"] = "JBSWY3DPEHPK3PXP"
os.environ["API_ADMIN_TOKEN"] = "A" * 32
os.environ["CONFIG_AES_KEY"] = "B" * 64
os.environ["TEST_MODE"] = "1"

import pytest
from fastapi.testclient import TestClient
from bitcoin_scalper.web.api import app

client = TestClient(app)

@pytest.fixture
def token():
    # Simule un login pour obtenir un token
    response = client.post("/token", data={"username": "admin", "password": os.environ["API_ADMIN_PASSWORD"]})
    assert response.status_code == 200
    return response.json()["access_token"]

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_login():
    response = client.post("/token", data={"username": "admin", "password": os.environ["API_ADMIN_PASSWORD"]})
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_verify():
    import pyotp
    totp = pyotp.TOTP(os.environ["API_ADMIN_TOTP"])
    code = totp.now()
    response = client.post("/verify", json={"username": "admin", "code": code})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_pnl(token):
    import pyotp
    code = pyotp.TOTP(os.environ["API_ADMIN_TOTP"]).now()
    response = client.get("/pnl", params={"username": "admin", "code": code}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["pnl"] == 42.0

def test_positions(token):
    import pyotp
    code = pyotp.TOTP(os.environ["API_ADMIN_TOTP"]).now()
    response = client.get("/positions", params={"username": "admin", "code": code}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["positions"] == [{"symbol": "BTCUSD", "volume": 0.1}]

def test_alerts(token):
    import pyotp
    code = pyotp.TOTP(os.environ["API_ADMIN_TOTP"]).now()
    response = client.get("/alerts", params={"username": "admin", "code": code}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["alerts"] == []

def test_kpis(token):
    import pyotp
    code = pyotp.TOTP(os.environ["API_ADMIN_TOTP"]).now()
    response = client.get("/kpis", params={"username": "admin", "code": code}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["kpis"] == {"drawdown": 0.01, "daily_pnl": 100.0} 