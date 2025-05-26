import pytest
from fastapi.testclient import TestClient
from app.web.api import app, users_db
import pyotp

client = TestClient(app)

@pytest.fixture
def admin_user():
    user = users_db["admin"]
    user["password"] = "adminpass"
    user["totp_secret"] = pyotp.random_base32()
    return user

def test_login_success(admin_user):
    resp = client.post("/token", data={"username": "admin", "password": "adminpass"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_fail():
    resp = client.post("/token", data={"username": "admin", "password": "wrong"})
    assert resp.status_code == 400
    assert "Identifiants invalides" in resp.text

def test_login_fail_unknown_user():
    resp = client.post("/token", data={"username": "unknown", "password": "any"})
    assert resp.status_code == 400
    assert "Identifiants invalides" in resp.text

def test_verify_totp_success(admin_user):
    totp = pyotp.TOTP(admin_user["totp_secret"])
    code = totp.now()
    resp = client.post("/verify", json={"username": "admin", "code": code})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_verify_totp_fail(admin_user):
    resp = client.post("/verify", json={"username": "admin", "code": "000000"})
    assert resp.status_code == 401
    assert "Code TOTP invalide" in resp.text

def test_verify_totp_fail_unknown_user():
    resp = client.post("/verify", json={"username": "unknown", "code": "000000"})
    assert resp.status_code == 400
    assert "Utilisateur inconnu" in resp.text

def test_secure_data_success(admin_user):
    totp = pyotp.TOTP(admin_user["totp_secret"])
    code = totp.now()
    token = "admin-token"
    resp = client.get("/secure-data", params={"username": "admin", "code": code}, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert "data" in resp.json()

def test_secure_data_fail_token():
    resp = client.get("/secure-data", params={"username": "admin", "code": "000000"}, headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401
    assert "Token invalide" in resp.text

def test_secure_data_fail_mfa():
    token = "admin-token"
    resp = client.get("/secure-data", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert "MFA requis" in resp.text

def test_secure_data_fail_user():
    token = "admin-token"
    resp = client.get("/secure-data", params={"username": "unknown", "code": "000000"}, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 400
    assert "Utilisateur inconnu" in resp.text

def test_secure_data_fail_totp_invalid(admin_user):
    token = "admin-token"
    # Invalid TOTP code for secure_data endpoint
    resp = client.get("/secure-data", params={"username": "admin", "code": "000000"}, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert "Code TOTP invalide" in resp.text

def test_status():
    resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_metrics():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "api_requests_total" in resp.text

def test_pnl_endpoint(admin_user):
    totp = pyotp.TOTP(admin_user["totp_secret"])
    code = totp.now()
    token = "admin-token"
    resp = client.get("/pnl", params={"username": "admin", "code": code}, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert "pnl" in data
    assert "drawdown" in data
    assert "history" in data
    assert isinstance(data["history"], list)

def test_positions_endpoint(admin_user):
    totp = pyotp.TOTP(admin_user["totp_secret"])
    code = totp.now()
    token = "admin-token"
    resp = client.get("/positions", params={"username": "admin", "code": code}, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert "positions" in data
    assert isinstance(data["positions"], list)

def test_alerts_endpoint(admin_user):
    totp = pyotp.TOTP(admin_user["totp_secret"])
    code = totp.now()
    token = "admin-token"
    resp = client.get("/alerts", params={"username": "admin", "code": code}, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert "alerts" in data
    assert isinstance(data["alerts"], list)

def test_kpis_endpoint(admin_user):
    totp = pyotp.TOTP(admin_user["totp_secret"])
    code = totp.now()
    token = "admin-token"
    resp = client.get("/kpis", params={"username": "admin", "code": code}, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert "sharpe" in data
    assert "winrate" in data
    assert "drawdown" in data
    assert "latency_ms" in data
    assert "trades" in data

def test_middleware_exception_returns_500():
    # Create a simple synchronous endpoint that raises an exception
    @app.get("/test-error-500")
    def test_error_endpoint_sync():
        raise ValueError("Simulated sync error in endpoint")

    # Use the test client to make the request
    resp_sync = client.get("/test-error-500")

    # Assert that the status code is 500 (Internal Server Error)
    assert resp_sync.status_code == 500

    # Clean up the added route - Find the route by path and remove it
    # Iterate over a copy of the list to allow deletion
    routes_to_remove = [route for route in app.routes if hasattr(route, 'path') and route.path == '/test-error-500']
    for route in routes_to_remove:
        app.routes.remove(route)

# Note: We are not directly testing if the prometheus counter is incremented here due to complexity
# in accessing the metrics object reliably in the test client context. The 500 status code confirms
# the middleware caught and re-raised the exception, covering the relevant lines.

def test_protected_endpoint_fail_token(admin_user):
    totp = pyotp.TOTP(admin_user["totp_secret"])
    code = totp.now()
    # Use an invalid token format
    resp = client.get("/pnl", params={"username": "admin", "code": code}, headers={"Authorization": "InvalidToken admin-token"})
    # Correct assertion for FastAPI's default unauthenticated response
    assert resp.status_code == 401
    assert "Not authenticated" in resp.text

def test_protected_endpoint_fail_mfa_missing(admin_user):
    token = "admin-token"
    # Missing username and code params
    resp = client.get("/pnl", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert "MFA requis" in resp.text

def test_protected_endpoint_fail_user_unknown(admin_user):
    totp = pyotp.TOTP(admin_user["totp_secret"])
    code = totp.now()
    token = "admin-token"
    # Unknown username
    resp = client.get("/pnl", params={"username": "unknown", "code": code}, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 400
    assert "Utilisateur inconnu" in resp.text

def test_protected_endpoint_fail_totp_invalid(admin_user):
    token = "admin-token"
    # Invalid TOTP code
    resp = client.get("/pnl", params={"username": "admin", "code": "000000"}, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
    assert "Code TOTP invalide" in resp.text 