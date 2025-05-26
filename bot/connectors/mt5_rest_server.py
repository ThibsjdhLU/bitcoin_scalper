from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import APIKeyHeader
from typing import List, Dict, Any
import logging
import os

# MetaTrader5 doit être importé et utilisé uniquement sur Windows
# import MetaTrader5 as mt5

logger = logging.getLogger("mt5_rest_server")
API_KEY = os.environ.get("MT5_API_KEY", "changeme")

app = FastAPI(title="MT5 REST Server", description="API REST pour piloter MetaTrader5 à distance.")
api_key_header = APIKeyHeader(name="Authorization")

def check_api_key(api_key: str = Depends(api_key_header)):
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide")

@app.get("/ticks", response_model=List[Dict[str, Any]])
def get_ticks(symbol: str, limit: int = 100, _: str = Depends(check_api_key)):
    """Retourne les derniers ticks pour un symbole donné."""
    # ticks = mt5.copy_ticks_from(symbol, ...)  # À implémenter
    # return [dict(...)]
    return [{"symbol": symbol, "bid": 68000, "ask": 68010, "timestamp": "2024-06-01T12:00:00Z", "volume": 0.5} for _ in range(limit)]

@app.get("/ohlcv", response_model=List[Dict[str, Any]])
def get_ohlcv(symbol: str, timeframe: str = "M1", limit: int = 100, _: str = Depends(check_api_key)):
    """Retourne les dernières bougies OHLCV pour un symbole et timeframe donnés."""
    # ohlcv = mt5.copy_rates_from_pos(symbol, timeframe, 0, limit)  # À implémenter
    # return [dict(...)]
    return [{"symbol": symbol, "open": 68000, "high": 68100, "low": 67900, "close": 68050, "volume": 1.2, "timestamp": "2024-06-01T12:00:00Z", "timeframe": timeframe} for _ in range(limit)]

@app.post("/order", response_model=Dict[str, Any])
def send_order(order: Dict[str, Any], _: str = Depends(check_api_key)):
    """Exécute un ordre (buy/sell) sur MT5."""
    # result = mt5.order_send({...})  # À implémenter
    # return dict(result)
    return {"order_id": 123, "status": "filled", "details": order}

@app.get("/status", response_model=Dict[str, Any])
def get_status(_: str = Depends(check_api_key)):
    """Retourne le statut du serveur MT5 (connecté, ping, etc)."""
    # status = ...  # À implémenter
    return {"status": "ok", "mt5": "connected"}

"""
Exemple de lancement :
$ export MT5_API_KEY=ma_cle
$ uvicorn bot.connectors.mt5_rest_server:app --host 0.0.0.0 --port 8000

Endpoints :
- GET /ticks?symbol=BTCUSD&limit=100
- GET /ohlcv?symbol=BTCUSD&timeframe=M1&limit=100
- POST /order (payload JSON)
- GET /status
""" 