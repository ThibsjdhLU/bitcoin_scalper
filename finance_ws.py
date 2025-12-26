#!/usr/bin/env python3
"""
Client WebSocket asynchrone pour Binance (kline 1m + option trade stream).
Expose un asyncio.Queue sur lequel le moteur de trading peut consommer les messages.

Dépendances:
  pip install websockets aiohttp
"""
import asyncio
import json
import logging
import traceback
import websockets
from typing import Optional

logger = logging.getLogger("binance_ws")
logging.basicConfig(level=logging.INFO)

BASE_WS = "wss://stream.binance.com:9443/ws"

class BinanceWS:
    def __init__(self, symbol="BTCUSDT", stream_type="kline_1m", queue: Optional[asyncio.Queue]=None):
        """
        stream_type: 'kline_1m' or 'trade'
        """
        self.symbol = symbol.lower()
        self.stream_type = stream_type
        self._stopped = False
        self.queue = queue or asyncio.Queue(maxsize=10000)

    def _stream_path(self):
        if self.stream_type == "kline_1m":
            return f"{self.symbol}@kline_1m"
        elif self.stream_type == "trade":
            return f"{self.symbol}@trade"
        else:
            raise ValueError("stream_type unknown")

    async def _connect_and_listen(self):
        uri = f"{BASE_WS}/{self._stream_path()}"
        backoff = 1
        while not self._stopped:
            try:
                logger.info("Connecting to %s", uri)
                async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("Connected")
                    backoff = 1
                    async for raw in ws:
                        try:
                            data = json.loads(raw)
                            # normalize kline: push only when kline closed=True for 1m, or push trade events
                            if self.stream_type == "kline_1m":
                                k = data.get("k", {})
                                if k.get("x", False):
                                    # closed candle
                                    item = {
                                        "open_time": k.get("t"),
                                        "open": k.get("o"),
                                        "high": k.get("h"),
                                        "low": k.get("l"),
                                        "close": k.get("c"),
                                        "volume": k.get("v"),
                                        "close_time": k.get("T"),
                                        "num_trades": k.get("n"),
                                    }
                                    await self._enqueue(item)
                            else:
                                # trade stream: push raw trade
                                await self._enqueue(data)
                        except Exception:
                            logger.exception("Error parsing message")
            except Exception:
                logger.warning("WS error, reconnect in %ds", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _enqueue(self, item):
        try:
            await self.queue.put(item)
        except asyncio.QueueFull:
            logger.warning("Queue full, dropping message")

    async def start(self):
        self._stopped = False
        await self._connect_and_listen()

    def stop(self):
        self._stopped = True

# Usage example
async def consumer_example():
    q = asyncio.Queue()
    client = BinanceWS(symbol="BTCUSDT", stream_type="kline_1m", queue=q)
    task = asyncio.create_task(client.start())
    try:
        while True:
            item = await q.get()
            # item is a closed 1m candle dict; adapt to your engine
            print("Received:", item)
    finally:
        client.stop()
        await task

if __name__ == "__main__":
    asyncio.run(consumer_example())
