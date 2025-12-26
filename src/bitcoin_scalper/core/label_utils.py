"""
Label Utility Module for Bitcoin Scalper.

This module provides standardized functions for encoding and decoding directional labels.
The project standardizes on the following mapping for internal model training (CatBoost/XGBoost):

Mapping:
    0 (Neutral)  -> 0
    1 (Buy)      -> 1
   -1 (Sell)     -> 2

This ensures that the '0' class remains 0, which facilitates filtering logic where
0 implies 'No Trade'.

Functions:
    encode_primary(y): Maps business labels {-1, 0, 1} to model labels {0, 1, 2}.
    decode_primary(y): Maps model labels {0, 1, 2} back to business labels {-1, 0, 1}.
"""

import numpy as np
import pandas as pd
from typing import Union

# Constants for the standard mapping
PRIMARY_LABEL_MAPPING = {0: 0, 1: 1, -1: 2}
PRIMARY_LABEL_DECODING = {0: 0, 1: 1, 2: -1}

def encode_primary(y: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Encode business labels {-1, 0, 1} into model-friendly integers {2, 0, 1}.

    Mapping:
        0  -> 0 (Neutral)
        1  -> 1 (Buy)
       -1  -> 2 (Sell)

    Args:
        y: Input labels (Series or array).

    Returns:
        Encoded labels.
    """
    if isinstance(y, pd.Series):
        # Fill NaN with 0 (Neutral) before encoding to be safe
        return y.map(PRIMARY_LABEL_MAPPING).fillna(0).astype(int)

    # Numpy array handling
    # Vectorized lookup
    # Only works if y contains valid keys. Fallback to 0 if unknown?
    # Using np.vectorize with a dict get
    mapper = np.vectorize(lambda x: PRIMARY_LABEL_MAPPING.get(x, 0))
    return mapper(y).astype(int)

def decode_primary(y: Union[pd.Series, np.ndarray, int]) -> Union[pd.Series, np.ndarray, int]:
    """
    Decode model integers {0, 1, 2} back to business labels {0, 1, -1}.

    Mapping:
        0 -> 0 (Neutral)
        1 -> 1 (Buy)
        2 -> -1 (Sell)

    Args:
        y: Encoded labels (Series, array, or scalar).

    Returns:
        Decoded labels.
    """
    if isinstance(y, (int, np.integer)):
        return PRIMARY_LABEL_DECODING.get(int(y), 0)

    if isinstance(y, pd.Series):
        return y.map(PRIMARY_LABEL_DECODING).fillna(0).astype(int)

    # Numpy array handling
    mapper = np.vectorize(lambda x: PRIMARY_LABEL_DECODING.get(x, 0))
    return mapper(y).astype(int)
