import os
import pprint

from bitcoin_scalper.connectors.binance_connector import BinanceConnector


def main():
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if not api_key or not api_secret:
        print("Warning: BINANCE_API_KEY or BINANCE_API_SECRET not set (using empty string).")
        # Optionally return here if you prefer to fail fast:
        # return

    c = BinanceConnector(api_key=api_key, api_secret=api_secret, testnet=True)

    try:
        res = c.get_ohlcv("BTC/USDT", "1m", limit=1500)
    except Exception as e:
        print("Failed to fetch OHLCV:", e)
        return

    print("type:", type(res))

    try:
        print("len:", len(res))
        if isinstance(res, list):
            pprint.pprint(res[:1])
    except Exception as e:
        print("Inspect failed:", e)

    # If your connector requires explicit close/shutdown, call it:
    if hasattr(c, "close"):
        try:
            c.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
