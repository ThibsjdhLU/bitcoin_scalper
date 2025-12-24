"""
Combinatorial Purged Cross-Validation for Financial Time Series.

This module implements scientifically rigorous cross-validation for financial ML:

1. **Purging**: Removes training samples whose labels overlap temporally with test set
   to prevent look-ahead bias from Triple Barrier labeling.

2. **Embargo**: Adds a buffer period after test sets to eliminate serial correlation
   between training and test data.

3. **Combinatorial**: Generates multiple train-test splits to create a distribution
   of performance metrics rather than point estimates.

These techniques are the "Gold Standard" for preventing overfitting and ensuring
backtest validity in financial machine learning.

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 7: Cross-Validation in Finance.
"""

import numpy as np
import pandas as pd
from typing import Optional, Generator, Tuple, List
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import _BaseKFold
import logging
from itertools import combinations
import warnings

logger = logging.getLogger(__name__)


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold Cross-Validation for time series with overlapping labels.
    
    This class implements K-Fold cross-validation with purging to prevent look-ahead
    bias in financial time series. When labels are computed using methods like the
    Triple Barrier that look forward in time, standard cross-validation can leak
    information from the test set into training.
    
    Purging removes training samples whose label computation period overlaps with
    the test set. Embargo adds an additional buffer after the test set.
    
    This is compatible with scikit-learn's cross-validation API and can be used
    with any sklearn model or pipeline.
    
    Attributes:
        n_splits: Number of folds for cross-validation.
        embargo_pct: Percentage of samples to embargo after each test set (0.0-1.0).
                     Recommended: 0.01 (1%) for high-frequency data.
        
    Example:
        >>> import pandas as pd
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import cross_val_score
        >>> 
        >>> # Assume we have features X, labels y, and event end times t1
        >>> # where t1[i] is when the label for sample i was determined
        >>> t1 = pd.Series(event_end_times, index=X.index)
        >>> 
        >>> # Create purged K-fold
        >>> cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
        >>> 
        >>> # Use with sklearn
        >>> model = RandomForestClassifier()
        >>> scores = cross_val_score(model, X, y, cv=cv.split(X, y, t1=t1))
        >>> 
        >>> print(f"Cross-val scores: {scores}")
        >>> print(f"Mean score: {scores.mean():.3f} (+/- {scores.std():.3f})")
        
    Notes:
        - Requires event end times (t1) to determine purge windows
        - Works with both pandas DataFrames/Series and numpy arrays
        - Maintains temporal ordering of data
        - Reduces training set size but increases validity
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
    ):
        """
        Initialize Purged K-Fold cross-validator.
        
        Args:
            n_splits: Number of folds. Must be at least 2.
            embargo_pct: Fraction of samples to embargo after test set.
                        Range [0.0, 1.0]. Default 0.01 (1%).
        
        Raises:
            ValueError: If n_splits < 2 or embargo_pct not in [0, 1].
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        if not 0 <= embargo_pct <= 1:
            raise ValueError(f"embargo_pct must be in [0, 1], got {embargo_pct}")
        
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        t1: Optional[pd.Series] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test sets.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target variable (optional, for compatibility).
            groups: Group labels (optional, not used).
            t1: Series with index matching X, containing the timestamp when
                each sample's label was determined (e.g., barrier touch time
                from Triple Barrier method). Required for purging.
        
        Yields:
            train_indices: Indices for training set (after purging).
            test_indices: Indices for test set.
            
        Raises:
            ValueError: If t1 is not provided or has mismatched length.
            
        Example:
            >>> for train_idx, test_idx in cv.split(X, y, t1=event_times):
            ...     X_train, X_test = X[train_idx], X[test_idx]
            ...     y_train, y_test = y[train_idx], y[test_idx]
            ...     model.fit(X_train, y_train)
            ...     score = model.score(X_test, y_test)
        """
        if t1 is None:
            warnings.warn(
                "t1 (event end times) not provided. Purging disabled. "
                "This may lead to look-ahead bias.",
                UserWarning
            )
            # Fall back to standard K-fold without purging
            t1 = pd.Series(np.arange(len(X)), index=range(len(X)))
        
        if len(t1) != len(X):
            raise ValueError(
                f"t1 length ({len(t1)}) must match X length ({len(X)})"
            )
        
        # Convert X to get indices
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Ensure t1 is a Series with proper index
        if not isinstance(t1, pd.Series):
            t1 = pd.Series(t1, index=indices)
        elif len(t1.index) != n_samples:
            # Reset index to match sample count
            t1 = pd.Series(t1.values, index=indices)
        
        # Calculate test set size and embargo size
        test_size = n_samples // self.n_splits
        embargo_size = int(test_size * self.embargo_pct)
        
        logger.info(
            f"PurgedKFold: n_samples={n_samples}, n_splits={self.n_splits}, "
            f"test_size={test_size}, embargo_size={embargo_size}"
        )
        
        # Generate splits
        for i in range(self.n_splits):
            # Define test set range
            test_start = i * test_size
            test_end = test_start + test_size if i < self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            
            # Get start time of test set
            test_start_time = t1.iloc[test_start]
            
            # Get end time of test set (for embargo)
            test_end_time = t1.iloc[test_end - 1]
            
            # Initialize train indices (all except test)
            train_indices = np.concatenate([
                indices[:test_start],
                indices[test_end:]
            ])
            
            # Purging: Remove training samples whose t1 >= test_start_time
            # This prevents label leakage from samples whose labeling period
            # overlaps with the test set
            # Only purge samples that come BEFORE the test set (not after)
            train_before_test = train_indices[train_indices < test_start]
            train_after_test = train_indices[train_indices >= test_end]
            
            # For samples before test, check if their t1 overlaps with test period
            if len(train_before_test) > 0:
                train_t1_before = t1.iloc[train_before_test]
                purge_mask_before = train_t1_before < test_start_time
                train_before_test = train_before_test[purge_mask_before]
                
                n_purged = (~purge_mask_before).sum()
                if n_purged > 0:
                    logger.debug(f"Fold {i+1}: Purged {n_purged} samples before test set")
            
            # Combine: samples before test (purged) + samples after test
            train_indices = np.concatenate([train_before_test, train_after_test])
            
            # Embargo: Remove samples immediately after test set
            # This eliminates serial correlation between test and subsequent training
            if embargo_size > 0 and test_end < n_samples:
                embargo_end = min(test_end + embargo_size, n_samples)
                embargo_indices = indices[test_end:embargo_end]
                
                # Remove embargo samples from training
                train_indices = train_indices[~np.isin(train_indices, embargo_indices)]
                
                logger.debug(
                    f"Fold {i+1}: Embargoed {len(embargo_indices)} samples "
                    f"after test set"
                )
            
            logger.info(
                f"Fold {i+1}/{self.n_splits}: train_size={len(train_indices)}, "
                f"test_size={len(test_indices)}"
            )
            
            yield train_indices, test_indices
    
    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """
        Get the number of splitting iterations.
        
        Args:
            X: Not used, present for API compatibility.
            y: Not used, present for API compatibility.
            groups: Not used, present for API compatibility.
        
        Returns:
            Number of splits.
        """
        return self.n_splits


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation for robust performance estimation.
    
    Instead of a single train-test split sequence, this generates multiple
    combinations of training and test paths. This creates a distribution of
    performance metrics rather than point estimates, providing:
    
    1. More robust performance estimates
    2. Confidence intervals for metrics
    3. Better understanding of strategy stability across different market regimes
    
    The combinatorial approach tests the model on diverse historical scenarios
    (e.g., bull markets, bear markets, crises) in various combinations.
    
    Attributes:
        n_splits: Number of groups to split data into.
        n_test_groups: Number of groups to use as test set in each combination.
        embargo_pct: Embargo percentage (same as PurgedKFold).
        
    Example:
        >>> from bitcoin_scalper.validation import CombinatorialPurgedCV
        >>> import numpy as np
        >>> 
        >>> # Create combinatorial CV with 8 paths, test on 2 at a time
        >>> cv = CombinatorialPurgedCV(n_splits=8, n_test_groups=2)
        >>> 
        >>> # Get all train-test combinations
        >>> results = []
        >>> for train_idx, test_idx in cv.split(X, y, t1=event_times):
        ...     model.fit(X[train_idx], y[train_idx])
        ...     score = model.score(X[test_idx], y[test_idx])
        ...     results.append(score)
        >>> 
        >>> # Analyze distribution
        >>> print(f"Mean score: {np.mean(results):.3f}")
        >>> print(f"Std score: {np.std(results):.3f}")
        >>> print(f"95% CI: [{np.percentile(results, 2.5):.3f}, "
        ...       f"{np.percentile(results, 97.5):.3f}]")
        
    Notes:
        - Number of combinations = C(n_splits, n_test_groups)
        - Can generate many splits, so computation may be expensive
        - Provides much better understanding of model robustness
        - Useful for final model validation before deployment
    """
    
    def __init__(
        self,
        n_splits: int = 8,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
    ):
        """
        Initialize Combinatorial Purged Cross-Validator.
        
        Args:
            n_splits: Number of groups to split data into.
            n_test_groups: Number of groups to use as test in each combination.
            embargo_pct: Fraction of samples to embargo after test set.
        
        Raises:
            ValueError: If parameters are invalid.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        if n_test_groups < 1 or n_test_groups >= n_splits:
            raise ValueError(
                f"n_test_groups must be in [1, n_splits), got {n_test_groups}"
            )
        if not 0 <= embargo_pct <= 1:
            raise ValueError(f"embargo_pct must be in [0, 1], got {embargo_pct}")
        
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct
        
        # Calculate number of combinations
        from math import comb
        self.n_combinations = comb(n_splits, n_test_groups)
        
        logger.info(
            f"CombinatorialPurgedCV: {self.n_splits} groups, "
            f"{self.n_test_groups} test groups per combination, "
            f"{self.n_combinations} total combinations"
        )
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        t1: Optional[pd.Series] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate combinatorial purged splits.
        
        Args:
            X: Training data.
            y: Target variable (optional).
            groups: Group labels (optional).
            t1: Event end times for purging.
        
        Yields:
            train_indices: Training set indices (after purging).
            test_indices: Test set indices.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Ensure t1 is properly formatted
        if t1 is None:
            warnings.warn(
                "t1 not provided for CombinatorialPurgedCV. "
                "Purging disabled.",
                UserWarning
            )
            t1 = pd.Series(np.arange(n_samples), index=indices)
        elif not isinstance(t1, pd.Series):
            t1 = pd.Series(t1, index=indices)
        
        # Split data into groups
        group_size = n_samples // self.n_splits
        embargo_size = int(group_size * self.embargo_pct)
        
        # Create all possible combinations of test groups
        test_combinations = list(combinations(range(self.n_splits), self.n_test_groups))
        
        logger.info(
            f"Generating {len(test_combinations)} combinatorial splits "
            f"with group_size={group_size}, embargo_size={embargo_size}"
        )
        
        for comb_idx, test_groups in enumerate(test_combinations):
            # Collect test indices from selected groups
            test_indices = []
            for group_idx in test_groups:
                start = group_idx * group_size
                end = start + group_size if group_idx < self.n_splits - 1 else n_samples
                test_indices.extend(indices[start:end])
            
            test_indices = np.array(test_indices)
            
            # Get test period boundaries for purging
            test_start_time = t1.iloc[test_indices.min()]
            test_end_time = t1.iloc[test_indices.max()]
            
            # All other indices are potential training
            train_indices = np.setdiff1d(indices, test_indices)
            
            # Separate training samples before and after test groups
            test_min = test_indices.min()
            test_max = test_indices.max()
            
            train_before = train_indices[train_indices < test_min]
            train_after = train_indices[train_indices > test_max]
            
            # Purge: Remove training samples before test with t1 >= test_start_time
            if len(train_before) > 0:
                train_t1_before = t1.iloc[train_before]
                purge_mask = train_t1_before < test_start_time
                train_before = train_before[purge_mask]
            
            # Combine purged training samples
            train_indices = np.concatenate([train_before, train_after]) if len(train_after) > 0 else train_before
            
            # Embargo: Remove samples after test groups
            if embargo_size > 0:
                embargo_indices = []
                for group_idx in test_groups:
                    end = (group_idx + 1) * group_size
                    if end < n_samples:
                        embargo_end = min(end + embargo_size, n_samples)
                        embargo_indices.extend(indices[end:embargo_end])
                
                if embargo_indices:
                    train_indices = train_indices[
                        ~np.isin(train_indices, embargo_indices)
                    ]
            
            logger.debug(
                f"Combination {comb_idx + 1}/{len(test_combinations)}: "
                f"test_groups={test_groups}, train_size={len(train_indices)}, "
                f"test_size={len(test_indices)}"
            )
            
            yield train_indices, test_indices
    
    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """
        Get number of combinations.
        
        Returns:
            Number of train-test split combinations.
        """
        return self.n_combinations
