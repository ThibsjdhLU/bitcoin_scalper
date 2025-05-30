import pytest
from fastapi.testclient import TestClient
from bitcoin_scalper.web.api import app, USERS
import pyotp
import os

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
            import bitcoin_scalper.web.api 