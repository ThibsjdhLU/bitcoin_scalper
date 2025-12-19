"""
Primary and meta-labeling functions for supervised learning.

This module provides functions to convert Triple Barrier events into labels
suitable for machine learning models, including:
- Primary labels: Direct classification from barrier touches
- Meta-labels: Secondary model labels for filtering false positives

Meta-labeling is a powerful technique where a primary model generates trading
signals, and a secondary model (meta-model) predicts whether each signal will
be profitable. This allows the system to filter out low-confidence bets.

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 3: Labeling.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def generate_labels_from_barriers(
    events: pd.DataFrame,
    close: pd.Series
) -> pd.Series:
    """
    Convert Triple Barrier events into classification labels.
    
    Takes the output from apply_triple_barrier or get_events and converts it
    into labels suitable for supervised learning:
    - 1: Profit target was hit first (bullish outcome)
    - -1: Stop loss was hit first (bearish outcome)
    - 0: Vertical barrier was hit (neutral outcome, time expired)
    
    Args:
        events: DataFrame from apply_triple_barrier/get_events with 'type' column.
        close: Price series (not used directly, included for API consistency).
    
    Returns:
        Series of labels {-1, 0, 1} aligned with events.index.
        
    Example:
        >>> events = get_events(close, timestamps, pt_sl=0.02, ...)
        >>> labels = generate_labels_from_barriers(events, close)
        >>> print(labels.value_counts())
        
    Notes:
        - This is the most straightforward labeling approach
        - Labels directly reflect which barrier was touched
        - Can be used as-is for ternary classification
        - Consider removing neutral labels (0) for binary classification
    """
    if 'type' not in events.columns:
        raise ValueError("Events DataFrame must contain 'type' column")
    
    labels = events['type'].copy()
    
    # Log distribution
    label_counts = labels.value_counts()
    total = len(labels)
    
    logger.info(
        f"Label distribution: "
        f"Long (+1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total*100:.1f}%), "
        f"Short (-1): {label_counts.get(-1, 0)} ({label_counts.get(-1, 0)/total*100:.1f}%), "
        f"Neutral (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/total*100:.1f}%)"
    )
    
    return labels


def get_labels(
    events: pd.DataFrame,
    close: pd.Series,
    label_type: str = 'fixed',
    return_threshold: Optional[float] = None
) -> pd.Series:
    """
    Generate primary labels from Triple Barrier events.
    
    Provides multiple labeling strategies based on barrier touch results:
    
    1. 'fixed': Simple barrier-based labels (1, -1, 0)
    2. 'sign': Sign of return at barrier touch (1 if return > 0, else -1)
    3. 'threshold': Thresholded return (1 if return > threshold, -1 if < -threshold, else 0)
    4. 'binary': Remove neutral labels, only keep strong signals (1, -1)
    
    Args:
        events: DataFrame from get_events containing barrier information.
        close: Price series with DatetimeIndex.
        label_type: Labeling strategy to use. Options:
                   - 'fixed': Use barrier type directly (default)
                   - 'sign': Use sign of actual return
                   - 'threshold': Threshold-based on return magnitude
                   - 'binary': Fixed labels excluding neutrals
        return_threshold: Threshold for 'threshold' label_type (e.g., 0.005 for 0.5%).
    
    Returns:
        Series of labels with same index as events.
        
    Example:
        >>> # Simple barrier-based labels
        >>> labels = get_labels(events, close, label_type='fixed')
        
        >>> # Binary labels (no neutrals)
        >>> labels = get_labels(events, close, label_type='binary')
        
        >>> # Threshold-based labels (must exceed 0.5% return)
        >>> labels = get_labels(events, close, label_type='threshold', 
        ...                     return_threshold=0.005)
        
    Notes:
        - 'fixed' is simplest and most aligned with barrier definition
        - 'sign' uses actual returns, which may differ from barrier expectations
        - 'threshold' helps filter out marginal wins/losses
        - 'binary' can improve model performance by removing ambiguous examples
    """
    if 'type' not in events.columns or 'return' not in events.columns:
        raise ValueError("Events must contain 'type' and 'return' columns")
    
    if label_type == 'fixed':
        # Use barrier type directly
        labels = events['type'].copy()
        
    elif label_type == 'sign':
        # Use sign of actual return
        labels = np.sign(events['return'])
        labels = labels.replace(0, 0)  # Keep zeros as neutral
        
    elif label_type == 'threshold':
        # Threshold-based labeling
        if return_threshold is None:
            raise ValueError("return_threshold must be specified for 'threshold' label_type")
        
        labels = pd.Series(0, index=events.index)
        labels[events['return'] > return_threshold] = 1
        labels[events['return'] < -return_threshold] = -1
        
    elif label_type == 'binary':
        # Remove neutral labels
        labels = events['type'].copy()
        labels = labels[labels != 0]
        
        logger.info(f"Binary labeling: Kept {len(labels)}/{len(events)} non-neutral events")
        
    else:
        raise ValueError(
            f"Unknown label_type: {label_type}. "
            f"Must be one of: 'fixed', 'sign', 'threshold', 'binary'"
        )
    
    # Log distribution
    label_counts = labels.value_counts()
    total = len(labels)
    
    logger.info(
        f"Labels ({label_type}): "
        f"+1: {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total*100:.1f}%), "
        f"-1: {label_counts.get(-1, 0)} ({label_counts.get(-1, 0)/total*100:.1f}%), "
        f"0: {label_counts.get(0, 0)} ({label_counts.get(0, 0)/total*100:.1f}%)"
    )
    
    return labels


def get_meta_labels(
    events: pd.DataFrame,
    close: pd.Series,
    primary_model_predictions: Optional[pd.Series] = None,
    side_from_predictions: bool = True
) -> pd.Series:
    """
    Generate meta-labels for training a secondary filtering model.
    
    Meta-labeling is used to train a model that filters out false positives from
    a primary model. The meta-model learns to predict: "Given that the primary
    model says to trade, will this trade be profitable?"
    
    The workflow is:
    1. Primary model generates trading signals (1: buy, -1: sell)
    2. Execute trades and observe outcomes using Triple Barrier Method
    3. Create meta-labels: 1 if trade was profitable, 0 if not
    4. Train meta-model to predict profitability given primary signal
    5. In production: Only trade when both primary and meta-models agree
    
    Args:
        events: DataFrame from get_events with barrier information.
        close: Price series with DatetimeIndex.
        primary_model_predictions: Series of primary model predictions {-1, 1}
                                  aligned with events.index. If None, assumes
                                  all events came from positive predictions.
        side_from_predictions: If True, use primary_model_predictions as the
                              trading side for return calculation. If False,
                              assumes all trades are in the direction of the
                              primary signal (long for buy, short for sell).
    
    Returns:
        Series of binary meta-labels:
        - 1: Trade would be profitable (keep the signal)
        - 0: Trade would be unprofitable (filter out the signal)
        
    Example:
        >>> # Train primary model
        >>> primary_model.fit(X_train, y_train)
        >>> primary_pred = primary_model.predict(X_test)
        
        >>> # Generate meta-labels
        >>> events = get_events(close, event_times, pt_sl=0.02, ...)
        >>> meta_labels = get_meta_labels(events, close, primary_pred)
        
        >>> # Train meta-model
        >>> meta_model.fit(X_test, meta_labels)
        
        >>> # In production
        >>> if primary_model.predict(X) == 1 and meta_model.predict(X) == 1:
        ...     execute_trade()
        
    Notes:
        - Meta-labeling significantly improves Sharpe ratio by filtering bad bets
        - The meta-model should use same or additional features as primary model
        - Can use probability estimates instead of hard predictions
        - Particularly effective for reducing drawdowns
    """
    if 'return' not in events.columns:
        raise ValueError("Events must contain 'return' column")
    
    # If no primary predictions provided, assume all events are from model predictions
    if primary_model_predictions is None:
        # Use the side that would make the return positive
        # If return > 0, label as 1 (successful)
        # If return <= 0, label as 0 (unsuccessful)
        meta_labels = (events['return'] > 0).astype(int)
        
    else:
        # Align predictions with events
        predictions = primary_model_predictions.loc[events.index]
        
        if side_from_predictions:
            # Calculate return considering the side predicted
            # If model predicted long (1) and return > 0, or predicted short (-1) and return < 0
            # then label is 1 (successful)
            adjusted_return = events['return'] * predictions
            meta_labels = (adjusted_return > 0).astype(int)
        else:
            # Simply check if return was positive
            meta_labels = (events['return'] > 0).astype(int)
    
    # Log distribution
    success_rate = meta_labels.mean()
    logger.info(
        f"Meta-labels: "
        f"Successful (1): {meta_labels.sum()} ({success_rate*100:.1f}%), "
        f"Unsuccessful (0): {(~meta_labels.astype(bool)).sum()} ({(1-success_rate)*100:.1f}%)"
    )
    
    # Warn if highly imbalanced
    if success_rate < 0.3 or success_rate > 0.7:
        logger.warning(
            f"Meta-labels are imbalanced (success rate: {success_rate:.2%}). "
            f"Consider adjusting barriers or using class weights."
        )
    
    return meta_labels


def get_bins(
    labels: pd.Series,
    num_bins: int = 10
) -> pd.Series:
    """
    Discretize continuous labels into bins for classification.
    
    Useful when working with continuous returns that need to be converted
    into discrete classes for classification models.
    
    Args:
        labels: Series of continuous labels (e.g., returns).
        num_bins: Number of bins to create.
    
    Returns:
        Series of discretized labels (0 to num_bins-1).
        
    Example:
        >>> returns = events['return']
        >>> binned_labels = get_bins(returns, num_bins=5)
        >>> # Now have 5 classes representing return quintiles
        
    Notes:
        - Uses quantile-based binning for balanced classes
        - Alternative to threshold-based binary labeling
        - Can capture more nuanced return patterns
    """
    binned = pd.qcut(labels, q=num_bins, labels=False, duplicates='drop')
    
    logger.info(f"Discretized labels into {binned.nunique()} bins")
    
    return binned


def apply_weighting(
    labels: pd.Series,
    returns: pd.Series,
    weighting_scheme: str = 'return'
) -> pd.Series:
    """
    Apply sample weighting based on label characteristics.
    
    Sample weights can be used during model training to emphasize certain
    examples over others. Common schemes:
    - 'return': Weight by absolute return (larger moves get more weight)
    - 'time': Weight recent examples more (time decay)
    - 'uniform': Equal weights (baseline)
    
    Args:
        labels: Series of labels.
        returns: Series of returns from events.
        weighting_scheme: Weighting scheme to apply.
    
    Returns:
        Series of sample weights aligned with labels.
        
    Example:
        >>> labels = get_labels(events, close)
        >>> weights = apply_weighting(labels, events['return'], 'return')
        >>> model.fit(X, labels, sample_weight=weights)
        
    Notes:
        - Can improve model focus on high-impact examples
        - Helps with imbalanced datasets
        - Consider normalizing weights for better stability
    """
    if weighting_scheme == 'uniform':
        weights = pd.Series(1.0, index=labels.index)
        
    elif weighting_scheme == 'return':
        # Weight by absolute return
        weights = returns.abs()
        # Normalize to sum to number of samples
        weights = weights / weights.sum() * len(weights)
        
    elif weighting_scheme == 'time':
        # Exponential time decay (more recent = higher weight)
        time_idx = np.arange(len(labels))
        weights = np.exp(time_idx / len(time_idx))
        weights = pd.Series(weights, index=labels.index)
        # Normalize
        weights = weights / weights.sum() * len(weights)
        
    else:
        raise ValueError(
            f"Unknown weighting_scheme: {weighting_scheme}. "
            f"Must be one of: 'uniform', 'return', 'time'"
        )
    
    logger.info(
        f"Applied '{weighting_scheme}' weighting: "
        f"mean={weights.mean():.3f}, std={weights.std():.3f}"
    )
    
    return weights
