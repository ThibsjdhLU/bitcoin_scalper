import os
import pandas as pd
from bot.connectors.mt5_rest_client import MT5RestClient
from bitcoin_scalper.core.config import SecureConfig

def ticks_to_ohlcv(df_ticks, timeframe='1min'):
    """
    Convertit un DataFrame de ticks en OHLCV (par minute).
    """
    print("Colonnes disponibles dans les ticks :", df_ticks.columns)
    df_ticks['datetime'] = pd.to_datetime(df_ticks['time'], unit='s')
    df_ticks = df_ticks.set_index('datetime')
    ohlcv = df_ticks.resample(timeframe).agg({
        'bid': 'first',
        'ask': 'first',
        'volume': 'sum'
    })
    ohlcv['open'] = df_ticks['bid'].resample(timeframe).first()
    ohlcv['high'] = df_ticks['bid'].resample(timeframe).max()
    ohlcv['low'] = df_ticks['bid'].resample(timeframe).min()
    ohlcv['close'] = df_ticks['bid'].resample(timeframe).last()
    ohlcv['volume'] = df_ticks['volume'].resample(timeframe).sum()
    ohlcv = ohlcv[['open', 'high', 'low', 'close', 'volume']]
    ohlcv = ohlcv.dropna()
    ohlcv.reset_index(inplace=True)
    return ohlcv

def export_btc_ohlcv():
    """
    Exporte l'historique OHLCV du Bitcoin depuis le serveur MT5 (REST) dans un fichier CSV.
    Les secrets sont lus depuis la configuration chiffrée (aucun secret en clair).
    """
    aes_key = os.environ.get("CONFIG_AES_KEY")
    if not aes_key:
        raise RuntimeError("La variable d'environnement CONFIG_AES_KEY doit être définie (clé AES-256 hex)")
    config = SecureConfig("config.enc", bytes.fromhex(aes_key))
    mt5_url = config.get("MT5_REST_URL")
    mt5_api_key = config.get("MT5_REST_API_KEY")

    client = MT5RestClient(mt5_url, api_key=mt5_api_key)
    symbol = "BTCUSD"
    limit = 1000  # Réduit pour éviter le timeout, augmenter progressivement si besoin
    ticks = client.get_ticks(symbol, limit=limit)

    df_ticks = pd.DataFrame(ticks)
    if df_ticks.empty:
        print("Aucun tick récupéré.")
        return
    df_ohlcv = ticks_to_ohlcv(df_ticks, timeframe='1min')
    output_path = f"data/features/{symbol}_M1_history.csv"
    df_ohlcv.to_csv(output_path, index=False)
    print(f"Historique OHLCV (M1) exporté : {output_path}")

if __name__ == "__main__":
    export_btc_ohlcv() 