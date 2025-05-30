import argparse
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("fix_missing_minutes")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

def fix_missing_minutes(input_csv: str, output_csv: str) -> None:
    """
    Corrige un CSV minute BTC en ajoutant les minutes manquantes (lignes NaN sauf datetime).

    :param input_csv: Chemin du CSV d'origine
    :param output_csv: Chemin du CSV corrigé
    """
    logger.info(f"Chargement du CSV : {input_csv}")
    df = pd.read_csv(input_csv, dtype=str, sep=None, engine='python')
    if '<DATE>' not in df.columns or '<TIME>' not in df.columns:
        logger.error("Colonnes <DATE> et/ou <TIME> absentes du CSV.")
        raise ValueError("Colonnes <DATE> et/ou <TIME> absentes du CSV.")
    dt_str = df['<DATE>'].astype(str) + ' ' + df['<TIME>'].astype(str)
    df['datetime'] = pd.to_datetime(dt_str, utc=True, errors='raise')
    df = df.set_index('datetime')
    df = df.sort_index()
    # Génère l'index complet minute à minute
    full_index = pd.date_range(df.index.min(), df.index.max(), freq='T', tz='UTC')
    n_missing = len(full_index) - len(df)
    logger.info(f"Nombre de minutes manquantes à combler : {n_missing}")
    df_full = df.reindex(full_index)
    # Remettre <DATE> et <TIME> à partir de l'index
    df_full['<DATE>'] = df_full.index.strftime('%Y.%m.%d')
    df_full['<TIME>'] = df_full.index.strftime('%H:%M:%S')
    # Réorganise les colonnes pour garder l'ordre d'origine
    cols = ['<DATE>', '<TIME>'] + [c for c in df_full.columns if c not in ['<DATE>', '<TIME>']]
    df_full = df_full[cols]
    logger.info(f"Sauvegarde du CSV corrigé : {output_csv}")
    df_full.to_csv(output_csv, index=False)
    logger.info("Correction terminée.")

def main():
    parser = argparse.ArgumentParser(description="Corrige un CSV minute BTC en ajoutant les minutes manquantes (lignes NaN sauf datetime)")
    parser.add_argument('--input', required=True, help='Chemin du CSV minute d\'origine')
    parser.add_argument('--output', required=True, help='Chemin du CSV corrigé')
    args = parser.parse_args()
    fix_missing_minutes(args.input, args.output)

if __name__ == '__main__':
    main() 