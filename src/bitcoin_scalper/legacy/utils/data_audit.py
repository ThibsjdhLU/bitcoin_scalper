#!/usr/bin/env python3
"""
Script CLI d'audit global de la qualité des données pour Bitcoin Scalper.
Usage :
  python scripts/audit_data_quality.py --input data.csv --labels target_5m,target_10m --outdir reports/
"""
import argparse
import pandas as pd
import os
from bitcoin_scalper.core.data_cleaner import DataCleaner

def main():
    parser = argparse.ArgumentParser(description="Audit global de la qualité des données OHLCV/ML.")
    parser.add_argument("--input", required=True, help="Fichier d'entrée OHLCV (CSV ou pickle)")
    parser.add_argument("--labels", default="", help="Colonnes de labels séparées par des virgules (optionnel)")
    parser.add_argument("--outdir", default="reports/", help="Dossier de sortie des rapports")
    args = parser.parse_args()
    # Chargement des données
    ext = os.path.splitext(args.input)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(args.input, index_col=0, parse_dates=True)
    elif ext in [".pkl", ".pickle"]:
        df = pd.read_pickle(args.input)
    else:
        raise ValueError("Format de fichier non supporté (CSV ou pickle uniquement)")
    label_cols = [c.strip() for c in args.labels.split(",") if c.strip()] if args.labels else None
    cleaner = DataCleaner()
    report = cleaner.audit_data_quality(df, label_cols=label_cols, report_dir=args.outdir, prefix="cli_")
    print(f"Rapport global d'audit généré : {report['global_audit'] if 'global_audit' in report else os.path.join(args.outdir, 'cli_global_audit.json')}")
    # Affiche les alertes critiques
    import json
    global_path = report.get('global_audit', os.path.join(args.outdir, 'cli_global_audit.json'))
    with open(global_path, "r") as f:
        global_report = json.load(f)
    for key in ["temporal", "clean_market", "correction"]:
        if key in global_report:
            with open(global_report[key], "r") as f2:
                sub = json.load(f2)
                for subkey, val in sub.items():
                    if isinstance(val, dict):
                        for k, v in val.items():
                            if v:
                                print(f"[ALERTE] {key}.{subkey}.{k} : {v}")
                    elif isinstance(val, list) and val:
                        print(f"[ALERTE] {key}.{subkey} : {val}")
if __name__ == "__main__":
    main() 