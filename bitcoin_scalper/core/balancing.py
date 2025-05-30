import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("bitcoin_scalper.balancing")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

def balance_by_block(
    df: pd.DataFrame,
    label_col: str = 'label',
    block_duration: str = '1D',
    min_block_size: int = 100,
    shuffle: bool = False
) -> pd.DataFrame:
    """
    Équilibre un DataFrame multi-classes par sous-échantillonnage local dans des blocs temporels non superposés.

    - Découpe le dataset en blocs temporels (par défaut 1 jour)
    - Dans chaque bloc, réduit les classes majoritaires à la taille de la minoritaire
    - Concatène les blocs équilibrés
    - Logging structuré des pertes par classe et bloc

    :param df: DataFrame d'entrée, indexé par datetime UTC, avec colonne label à 3 classes
    :param label_col: Nom de la colonne label (par défaut 'label')
    :param block_duration: Durée d'un bloc temporel (ex: '1D', '4H')
    :param min_block_size: Taille minimale d'un bloc pour être conservé
    :param shuffle: Mélange les lignes dans chaque bloc avant sous-échantillonnage (par défaut False)
    :return: DataFrame équilibré, même colonnes, même index (sous-ensemble)
    """
    if label_col not in df.columns:
        logger.error(f"Colonne label '{label_col}' absente du DataFrame.")
        raise ValueError(f"Colonne label '{label_col}' absente du DataFrame.")
    logger.info(f"Équilibrage par bloc temporel : block_duration={block_duration}, min_block_size={min_block_size}, shuffle={shuffle}")
    # Découpage en blocs temporels non superposés
    df = df.copy()
    df = df.sort_index()
    blocks = []
    for block_start, block in df.groupby(pd.Grouper(freq=block_duration)):
        if len(block) < min_block_size:
            logger.info(f"Bloc {block_start} ignoré (taille {len(block)} < {min_block_size})")
            continue
        class_counts = block[label_col].value_counts()
        if class_counts.min() == 0 or class_counts.shape[0] < 2:
            logger.warning(f"Bloc {block_start} ignoré (classes absentes ou non équilibrables) : {class_counts.to_dict()}")
            continue
        n_min = class_counts.min()
        block_balanced = []
        for c in class_counts.index:
            sub = block[block[label_col] == c]
            if shuffle:
                sub = sub.sample(frac=1, random_state=42)
            sub = sub.iloc[:n_min]
            block_balanced.append(sub)
            logger.info(f"Bloc {block_start} : classe {c} réduite de {len(block[block[label_col]==c])} à {len(sub)}")
        block_balanced = pd.concat(block_balanced)
        block_balanced = block_balanced.sort_index()
        blocks.append(block_balanced)
    if not blocks:
        logger.error("Aucun bloc équilibré généré (vérifier les paramètres ou la distribution des classes)")
        raise ValueError("Aucun bloc équilibré généré")
    balanced_df = pd.concat(blocks).sort_index()
    logger.info(f"DataFrame équilibré : {balanced_df.shape[0]} lignes, répartition : {balanced_df[label_col].value_counts().to_dict()}")
    return balanced_df 