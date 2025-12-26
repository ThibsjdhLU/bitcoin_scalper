#!/usr/bin/env python3
"""
Télécharge les fichiers 1m depuis data.binance.vision, les concatène et produit un fichier Parquet
optimisé pour l'entraînement ML.

Usage:
  python scripts/download_binance_1m_from_data_vision.py --symbol BTCUSDT --start 2020-01-01 --end 2025-12-01 --out data/btc_1m.parquet

Dépendances:
  pip install requests pandas pyarrow tqdm python-dateutil
"""
import os
import io
import zipfile
import argparse
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
from dateutil import parser as dateparser
from tqdm import tqdm

BASE = "https://data.binance.vision/data/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date}.zip"
INTERVAL = "1m"

def daterange(start_date, end_date):
    cur = start_date
    while cur <= end_date:
        yield cur
        cur += timedelta(days=1)

def download_zip_for_date(symbol, date_str, session, out_dir, timeout=60):
    url = BASE.format(symbol=symbol, interval=INTERVAL, date=date_str)
    local_zip = os.path.join(out_dir, f"{symbol}-{INTERVAL}-{date_str}.zip")
    if os.path.exists(local_zip) and os.path.getsize(local_zip) > 0:
        return local_zip  # already downloaded
    r = session.get(url, stream=True, timeout=timeout)
    if r.status_code == 200:
        os.makedirs(out_dir, exist_ok=True)
        with open(local_zip + ".tmp", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        os.replace(local_zip + ".tmp", local_zip)
        return local_zip
    else:
        # file may not exist for that date
        return None

def extract_csv_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        # Typical zip contains one CSV named like SYMBOL-1m-YYYY-MM-DD.csv
        for name in z.namelist():
            if name.endswith(".csv"):
                with z.open(name) as fh:
                    return fh.read()  # bytes
    return None

def csv_bytes_to_df(bts):
    # CSV columns: open_time,open,high,low,close,volume,close_time,quote_asset_volume,num_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume,ignore
    df = pd.read_csv(io.BytesIO(bts), header=None)
    # enforce names
    df.columns = ["open_time","open","high","low","close","volume","close_time","quote_asset_volume","num_trades","taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    numeric = ["open","high","low","close","volume","quote_asset_volume","taker_buy_base_asset_volume","taker_buy_quote_asset_volume"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce").astype("Int64")
    return df

def append_to_parquet(out_path, df_new):
    if os.path.exists(out_path):
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, df_new], ignore_index=True).drop_duplicates(subset=["open_time"]).sort_values("open_time")
        combined.to_parquet(out_path, index=False)
    else:
        df_new.drop_duplicates(subset=["open_time"]).sort_values("open_time").to_parquet(out_path, index=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out", default="data/btc_1m.parquet")
    p.add_argument("--download-dir", default="data/raw_zips")
    args = p.parse_args()

    start_date = dateparser.parse(args.start).date()
    end_date = dateparser.parse(args.end).date()
    session = requests.Session()
    out_parquet = args.out

    # determine already-downloaded dates from parquet to skip
    existing_dates = set()
    if os.path.exists(out_parquet):
        try:
            ex = pd.read_parquet(out_parquet, columns=["open_time"])
            if not ex.empty:
                existing_dates = set(dt.date() for dt in pd.to_datetime(ex["open_time"]))
        except Exception:
            existing_dates = set()

    for d in tqdm(list(daterange(start_date, end_date)), desc="dates"):
        date_str = d.strftime("%Y-%m-%d")
        if d in existing_dates:
            continue
        try:
            zip_local = download_zip_for_date(args.symbol, date_str, session, args.download_dir)
            if not zip_local:
                # pas de fichier pour cette date, skip
                continue
            bts = extract_csv_from_zip(zip_local)
            if not bts:
                continue
            df = csv_bytes_to_df(bts)
            append_to_parquet(out_parquet, df)
        except Exception as e:
            print(f"Erreur pour {date_str}: {e}")
            # continue pour ne pas tout casser
            continue

    print("Terminé. Parquet:", out_parquet)

if __name__ == "__main__":
    main()
