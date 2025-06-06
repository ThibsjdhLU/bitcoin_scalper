import pytest
import pyotp
import os
from fastapi.testclient import TestClient

def patch_env():
    os.environ["API_ADMIN_PASSWORD"] = "C"*16
    os.environ["API_ADMIN_TOTP"] = "A"*16
    os.environ["API_ADMIN_TOKEN"] = "B"*32
    os.environ["CONFIG_AES_KEY"] = "a"*64  # Clé hexadécimale de test

patch_env()

# Patch SecureConfig globalement pour tous les tests de ce fichier
import sys
import importlib
from unittest.mock import patch

class DummyConfig:
    def __init__(self, *a, **kw): pass
    def get(self, key, default=None):
        return {
            "MT5_REST_URL": "http://localhost:8080",
            "MT5_REST_API_KEY": "fakekey",
            "TSDB_HOST": "localhost",
            "TSDB_PORT": "5432",
            "TSDB_NAME": "btcdb",
            "TSDB_USER": "btcuser",
            "TSDB_PASSWORD": "btcpass",
            "TSDB_SSLMODE": "disable",
            "ML_MODEL_PATH": "model_rf.pkl"
        }.get(key, default)

with patch("bitcoin_scalper.core.config.SecureConfig", DummyConfig):
    from bitcoin_scalper.web.api import app, USERS
    client = TestClient(app)

    @pytest.fixture
    def admin_creds():
        username = "admin"
        password = USERS[username]["password"]
        totp_secret = USERS[username]["totp_secret"]
        token = USERS[username]["token"]
        totp = pyotp.TOTP(totp_secret)
        code = totp.now()
        return {"username": username, "password": password, "token": token, "code": code}

def test_login_token(admin_creds):
    resp = client.post("/token", data={"username": admin_creds["username"], "password": admin_creds["password"]})
    assert resp.status_code == 200
    assert "access_token" in resp.json()

def test_login_token_fail():
    resp = client.post("/token", data={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401

def test_verify_totp(admin_creds):
    resp = client.post("/verify", json={"username": admin_creds["username"], "code": admin_creds["code"]})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_verify_totp_fail(admin_creds):
    resp = client.post("/verify", json={"username": admin_creds["username"], "code": "000000"})
    assert resp.status_code == 401

def test_protected_endpoints(admin_creds):
    headers = {"Authorization": f"Bearer {admin_creds['token']}"}
    for ep in ["/pnl", "/positions", "/alerts", "/kpis"]:
        resp = client.get(ep, params={"username": admin_creds["username"], "code": admin_creds["code"]}, headers=headers)
        assert resp.status_code == 200

def test_protected_endpoints_mfa_fail(admin_creds):
    headers = {"Authorization": f"Bearer {admin_creds['token']}"}
    for ep in ["/pnl", "/positions", "/alerts", "/kpis"]:
        resp = client.get(ep, params={"username": admin_creds["username"], "code": "000000"}, headers=headers)
        assert resp.status_code == 401

def test_protected_endpoints_token_fail(admin_creds):
    headers = {"Authorization": "Bearer wrongtoken"}
    for ep in ["/pnl", "/positions", "/alerts", "/kpis"]:
        resp = client.get(ep, params={"username": admin_creds["username"], "code": admin_creds["code"]}, headers=headers)
        assert resp.status_code == 401

def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_api_refuse_if_password_missing(monkeypatch):
    monkeypatch.delenv("API_ADMIN_PASSWORD", raising=False)
    monkeypatch.setenv("API_ADMIN_TOTP", "A"*16)
    monkeypatch.setenv("API_ADMIN_TOKEN", "B"*32)
    with pytest.raises(RuntimeError):
        import importlib
        import sys
        if "bitcoin_scalper.web.api" in sys.modules:
            importlib.reload(sys.modules["bitcoin_scalper.web.api"])
        else:
            patch_env()
            import bitcoin_scalper.web.api

def test_api_refuse_if_password_too_short(monkeypatch):
    monkeypatch.setenv("API_ADMIN_PASSWORD", "short")
    monkeypatch.setenv("API_ADMIN_TOTP", "A"*16)
    monkeypatch.setenv("API_ADMIN_TOKEN", "B"*32)
    with pytest.raises(RuntimeError):
        import importlib
        import sys
        if "bitcoin_scalper.web.api" in sys.modules:
            importlib.reload(sys.modules["bitcoin_scalper.web.api"])
        else:
            patch_env()
            import bitcoin_scalper.web.api

def test_api_refuse_if_totp_missing(monkeypatch):
    monkeypatch.setenv("API_ADMIN_PASSWORD", "C"*16)
    monkeypatch.delenv("API_ADMIN_TOTP", raising=False)
    monkeypatch.setenv("API_ADMIN_TOKEN", "B"*32)
    with pytest.raises(RuntimeError):
        import importlib
        import sys
        if "bitcoin_scalper.web.api" in sys.modules:
            importlib.reload(sys.modules["bitcoin_scalper.web.api"])
        else:
            patch_env()
            import bitcoin_scalper.web.api

def test_api_refuse_if_token_too_short(monkeypatch):
    monkeypatch.setenv("API_ADMIN_PASSWORD", "C"*16)
    monkeypatch.setenv("API_ADMIN_TOTP", "A"*16)
    monkeypatch.setenv("API_ADMIN_TOKEN", "short")
    with pytest.raises(RuntimeError):
        import importlib
        import sys
        if "bitcoin_scalper.web.api" in sys.modules:
            importlib.reload(sys.modules["bitcoin_scalper.web.api"])
        else:
            patch_env()
            import bitcoin_scalper.web.api 