import pandas as pd
import logging

logger = logging.getLogger(__name__)

def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère un signal de trading simple basé sur RSI et SMA (Placeholder).
    """
    df = df.copy()
    if 'rsi' not in df.columns:
        df['rsi'] = 50  # Dummy

    df['signal'] = 0
    df.loc[df['rsi'] < 30, 'signal'] = 1  # Buy
    df.loc[df['rsi'] > 70, 'signal'] = -1 # Sell

    return df
