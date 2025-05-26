import pytest
from fastapi.testclient import TestClient
from app.web.api import app, users_db
import pyotp

client = TestClient(app)

USERNAME = "admin"
PASSWORD = "adminpass"


def test_secure_data_no_mfa():
    # Login pour obtenir token
    resp = client.post("/token", data={"username": USERNAME, "password": PASSWORD})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    # Appel endpoint protégé sans MFA
    resp2 = client.get("/secure-data", headers={"Authorization": f"Bearer {token}"})
    assert resp2.status_code == 401
    assert "MFA requis" in resp2.text

def test_secure_data_with_valid_mfa():
    # Login pour obtenir token
    resp = client.post("/token", data={"username": USERNAME, "password": PASSWORD})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    # Générer code TOTP valide
    secret = users_db[USERNAME]["totp_secret"]
    code = pyotp.TOTP(secret).now()
    # Appel endpoint protégé avec MFA
    resp2 = client.get(
        f"/secure-data?username={USERNAME}&code={code}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp2.status_code == 200
    assert "Données sensibles" in resp2.text

def test_secure_data_with_invalid_mfa():
    # Login pour obtenir token
    resp = client.post("/token", data={"username": USERNAME, "password": PASSWORD})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    # Code TOTP invalide
    code = "000000"
    resp2 = client.get(
        f"/secure-data?username={USERNAME}&code={code}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp2.status_code == 401
    assert "Code TOTP invalide" in resp2.text

def test_no_secret_leak():
    # Vérifie qu'aucun secret n'est exposé dans la réponse
    resp = client.post("/token", data={"username": USERNAME, "password": PASSWORD})
    assert resp.status_code == 200
    assert "totp_secret" not in resp.text
    resp2 = client.get("/secure-data", headers={"Authorization": f"Bearer {resp.json()['access_token']}"})
    assert "totp_secret" not in resp2.text 