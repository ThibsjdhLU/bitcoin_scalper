import os, pprint 
from bitcoin_scalper.connectors.binance_connector import BinanceConnector 
c = BinanceConnector(api_key=os.getenv('BINANCE_API_KEY',''), 
                     api_secret=os.getenv('BINANCE_API_SECRET',''), 
                     testnet=True) res = c.get_ohlcv('BTC/USDT', '1m', limit=1500) 
print("type:", type(res)) try: print("len:", len(res)) if isinstance(res, list): pprint.pprint(res[:1]) except Exception as e: print("inspect failed:", e)
