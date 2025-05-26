import MetaTrader5 as mt5
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import logging
import os
from typing import Optional

app = FastAPI()
logger = logging.getLogger("mt5_rest_server")

# --- Sécurité simple par clé API (à renforcer en prod) ---
API_KEY = os.environ.get("MT5_REST_API_KEY", "changeme")

def check_api_key(request: Request):
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# --- Modèles Pydantic ---
class OrderRequest(BaseModel):
    symbol: str
    action: str  # 'buy' ou 'sell'
    volume: float
    price: Optional[float] = None
    order_type: str = "market"  # 'market' ou 'limit'

# --- Endpoints REST ---
@app.on_event("startup")
def startup_event():
    if not mt5.initialize():
        logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    logger.info("MetaTrader5 initialized.")

@app.on_event("shutdown")
def shutdown_event():
    mt5.shutdown()
    logger.info("MetaTrader5 shutdown.")

@app.get("/account")
async def get_account(request: Request):
    check_api_key(request)
    acc = mt5.account_info()._asdict()
    return acc

@app.get("/symbol/{symbol}")
async def get_symbol(symbol: str, request: Request):
    check_api_key(request)
    info = mt5.symbol_info(symbol)
    if not info:
        raise HTTPException(status_code=404, detail="Symbol not found")
    return info._asdict()

@app.get("/ticks/{symbol}")
async def get_ticks(symbol: str, limit: int = 100, request: Request = None):
    check_api_key(request)
    ticks = mt5.copy_ticks_from(symbol, mt5.TIMEFRAME_M1, 0, limit)
    return [dict(t) for t in ticks]

@app.get("/ohlcv/{symbol}")
async def get_ohlcv(symbol: str, timeframe: str = "M1", limit: int = 100, request: Request = None):
    check_api_key(request)
    tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}
    tf = tf_map.get(timeframe, mt5.TIMEFRAME_M1)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, limit)
    if rates is None or len(rates) == 0:
        return []
    result = []
    for row in rates:
        d = {
            "symbol": symbol,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["tick_volume"]),  # ou "real_volume"
            "timestamp": int(row["time"]),  # ou datetime.utcfromtimestamp(row["time"]).isoformat()
            "timeframe": timeframe
        }
        result.append(d)
    return result

@app.post("/order")
async def send_order(order: OrderRequest, request: Request):
    check_api_key(request)
    price = order.price
    if price is None:
        tick = mt5.symbol_info_tick(order.symbol)
        price = tick.ask if order.action == "buy" else tick.bid
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": order.symbol,
        "volume": order.volume,
        "type": mt5.ORDER_TYPE_BUY if order.action == "buy" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "deviation": 5,
        "magic": 123456,
        "comment": "REST order"
    }
    result = mt5.order_send(req)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise HTTPException(status_code=400, detail=f"Order failed: {result.comment}")
    return {"order_id": result.order, "status": result.comment}

if __name__ == "__main__":
    uvicorn.run("windows.mt5_rest_server:app", host="0.0.0.0", port=8000, reload=False) 