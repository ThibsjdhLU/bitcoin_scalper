"""
Drift Detection for Online Learning and Model Monitoring.

This module implements algorithms to detect concept drift - changes in the statistical
properties of the prediction target over time. In financial markets, this occurs when:
- Market regimes change (bull → bear market)
- New market participants or regulations alter dynamics
- Model assumptions become invalid

Detecting drift allows systems to:
1. Stop trading when the model is no longer valid
2. Trigger automatic retraining on recent data
3. Alert operators to market regime changes

References:
    Bifet, A., & Gavaldà, R. (2007). Learning from Time-Changing Data with Adaptive Windowing.
    Gama, J., et al. (2014). A Survey on Concept Drift Adaptation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from collections import deque
import logging
import warnings

logger = logging.getLogger(__name__)


class ADWINDetector:
    """
    ADWIN (ADaptive WINdowing) drift detector.
    
    ADWIN maintains a sliding window of recent values and automatically shrinks or
    expands the window based on detecting significant changes in the mean. When a
    statistically significant change is detected, it drops older data and signals drift.
    
    This is a lightweight implementation suitable for monitoring prediction errors
    or any time series for distributional changes.
    
    Algorithm:
    1. Maintain sliding window of recent observations
    2. For each new observation, test if dropping old data significantly changes mean
    3. Use Hoeffding bound to test significance with confidence δ
    4. When drift detected, drop old observations and signal alert
    
    Attributes:
        delta: Confidence level for drift detection (smaller = more sensitive).
               Recommended: 0.002 for strict detection, 0.05 for lenient.
        max_window: Maximum window size (for memory constraints).
        drift_detected: Flag indicating if drift was detected in last update.
        n_detections: Counter of total drift detections.
        
    Example:
        >>> from bitcoin_scalper.validation import ADWINDetector
        >>> 
        >>> # Create detector
        >>> detector = ADWINDetector(delta=0.002)
        >>> 
        >>> # Monitor prediction errors online
        >>> for y_true, y_pred in zip(true_values, predictions):
        ...     error = abs(y_true - y_pred)
        ...     drift = detector.update(error)
        ...     
        ...     if drift:
        ...         print("Drift detected! Model may need retraining.")
        ...         # Take action: stop trading, retrain, alert
        
    Notes:
        - Memory efficient: automatically discards old data
        - Online algorithm: processes one observation at a time
        - No assumptions about data distribution
        - Adapts to both gradual and sudden changes
    """
    
    def __init__(
        self,
        delta: float = 0.002,
        max_window: int = 10000,
    ):
        """
        Initialize ADWIN drift detector.
        
        Args:
            delta: Confidence parameter. Smaller values = more sensitive.
                   Range (0, 1). Typical: 0.002 (strict) to 0.05 (lenient).
            max_window: Maximum window size to prevent unbounded memory growth.
                       Typical: 1000-10000 depending on sampling rate.
        
        Raises:
            ValueError: If delta not in (0, 1) or max_window < 2.
        """
        if not 0 < delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if max_window < 2:
            raise ValueError(f"max_window must be >= 2, got {max_window}")
        
        self.delta = delta
        self.max_window = max_window
        
        # Window of recent observations
        self._window = deque(maxlen=max_window)
        
        # Statistics
        self._total = 0.0  # Sum of values in window
        self._variance = 0.0  # Variance estimate
        
        # Detection state
        self.drift_detected = False
        self.n_detections = 0
        self._estimation_mode = True  # In estimation mode initially
        
        logger.info(
            f"ADWINDetector initialized with delta={delta}, "
            f"max_window={max_window}"
        )
    
    def reset(self) -> None:
        """
        Reset the detector to initial state.
        
        Clears the window and all statistics. Use after retraining model
        or when starting monitoring of a new data stream.
        """
        self._window.clear()
        self._total = 0.0
        self._variance = 0.0
        self.drift_detected = False
        self._estimation_mode = True
        logger.info("ADWINDetector reset")
    
    def update(self, value: float) -> bool:
        """
        Update detector with new observation and check for drift.
        
        Args:
            value: New observation (typically prediction error or loss).
        
        Returns:
            True if drift detected, False otherwise.
            
        Example:
            >>> detector = ADWINDetector()
            >>> 
            >>> # Stable period
            >>> for i in range(100):
            ...     error = np.random.normal(0.1, 0.02)
            ...     assert not detector.update(error)
            >>> 
            >>> # Drift: error distribution changes
            >>> for i in range(50):
            ...     error = np.random.normal(0.5, 0.1)  # Much higher error
            ...     if detector.update(error):
            ...         print(f"Drift detected at sample {100 + i}")
            ...         break
        """
        self.drift_detected = False
        
        # Add new value to window
        self._window.append(value)
        self._total += value
        
        # Need at least 2 values to detect drift
        if len(self._window) < 2:
            return False
        
        # Exit estimation mode after sufficient samples
        if self._estimation_mode and len(self._window) >= 30:
            self._estimation_mode = False
            logger.debug("Exiting estimation mode, drift detection active")
        
        # Don't detect drift during estimation
        if self._estimation_mode:
            return False
        
        # Check for drift by comparing window sub-ranges
        drift = self._detect_change()
        
        if drift:
            self.drift_detected = True
            self.n_detections += 1
            logger.warning(
                f"Drift detected! Total detections: {self.n_detections}, "
                f"window_size: {len(self._window)}"
            )
        
        return drift
    
    def _detect_change(self) -> bool:
        """
        Internal method to detect change in window.
        
        Uses sliding window approach with Hoeffding bound to test if
        dropping older observations significantly changes the mean.
        
        Returns:
            True if significant change detected.
        """
        n = len(self._window)
        
        if n < 10:  # Need minimum samples for reliable detection
            return False
        
        # Convert to array for efficient computation
        window_arr = np.array(self._window)
        
        # Try different cut points to find significant change
        # Check cuts from 20% to 80% of window
        min_cut = max(5, int(n * 0.2))
        max_cut = min(n - 5, int(n * 0.8))
        
        for cut in range(min_cut, max_cut, max(1, (max_cut - min_cut) // 10)):
            # Split window at cut point
            w0 = window_arr[:cut]
            w1 = window_arr[cut:]
            
            # Calculate means
            mu0 = w0.mean()
            mu1 = w1.mean()
            
            # Calculate difference
            diff = abs(mu0 - mu1)
            
            # Calculate Hoeffding bound
            n0 = len(w0)
            n1 = len(w1)
            m = 1.0 / n0 + 1.0 / n1
            
            # Estimate variance for bound calculation
            var = window_arr.var()
            if var == 0:
                var = 1e-10  # Prevent division by zero
            
            # Hoeffding bound with variance adjustment
            epsilon = np.sqrt(2 * var * m * np.log(2 / self.delta))
            
            # Detect if difference exceeds bound
            if diff > epsilon:
                # Drift detected - drop older observations
                n_dropped = cut
                for _ in range(n_dropped):
                    dropped = self._window.popleft()
                    self._total -= dropped
                
                logger.debug(
                    f"Drift: mu0={mu0:.4f}, mu1={mu1:.4f}, diff={diff:.4f}, "
                    f"bound={epsilon:.4f}, dropped {n_dropped} old samples"
                )
                
                return True
        
        return False
    
    def get_window_size(self) -> int:
        """
        Get current window size.
        
        Returns:
            Number of observations in window.
        """
        return len(self._window)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get current detector statistics.
        
        Returns:
            Dictionary with:
            - window_size: Current window size
            - mean: Mean of values in window
            - variance: Variance of values in window
            - n_detections: Total drift detections
            
        Example:
            >>> stats = detector.get_statistics()
            >>> print(f"Mean error: {stats['mean']:.4f}")
            >>> print(f"Detections: {stats['n_detections']}")
        """
        if len(self._window) == 0:
            return {
                'window_size': 0,
                'mean': 0.0,
                'variance': 0.0,
                'n_detections': self.n_detections,
            }
        
        window_arr = np.array(self._window)
        return {
            'window_size': len(self._window),
            'mean': window_arr.mean(),
            'variance': window_arr.var(),
            'n_detections': self.n_detections,
        }


class DriftScanner:
    """
    High-level drift monitoring system for trading models.
    
    This class provides a convenient interface for monitoring model performance
    and detecting drift in production. It wraps ADWIN and adds:
    - Multiple monitoring streams (error, returns, volatility)
    - Drift history logging
    - Integration with river library if available
    
    Use this in production to continuously monitor if your model remains valid.
    
    Attributes:
        detector: Underlying ADWIN detector.
        history: List of drift detection events with timestamps.
        
    Example:
        >>> from bitcoin_scalper.validation import DriftScanner
        >>> 
        >>> # Initialize scanner
        >>> scanner = DriftScanner(delta=0.002)
        >>> 
        >>> # In production loop
        >>> for timestamp, y_true, y_pred in trading_stream:
        ...     error = abs(y_true - y_pred)
        ...     
        ...     if scanner.scan(error, timestamp):
        ...         print(f"Drift detected at {timestamp}")
        ...         # Stop trading and retrain
        ...         model = retrain_model()
        ...         scanner.reset()
        
    Notes:
        - Automatically logs drift events with timestamps
        - Can monitor multiple metrics simultaneously
        - Integration with river library for advanced online learning
    """
    
    def __init__(
        self,
        delta: float = 0.002,
        max_window: int = 10000,
        use_river: bool = True,
    ):
        """
        Initialize drift scanner.
        
        Args:
            delta: ADWIN confidence parameter (0 < delta < 1).
            max_window: Maximum window size.
            use_river: Try to use river library if available (more efficient).
        """
        self.delta = delta
        self.max_window = max_window
        
        # Try to use river's ADWIN if available
        self._use_river = False
        if use_river:
            try:
                from river.drift import ADWIN as RiverADWIN
                self.detector = RiverADWIN(delta=delta)
                self._use_river = True
                logger.info("Using river.drift.ADWIN for drift detection")
            except ImportError:
                logger.info("river not available, using built-in ADWIN")
                self.detector = ADWINDetector(delta=delta, max_window=max_window)
        else:
            self.detector = ADWINDetector(delta=delta, max_window=max_window)
        
        # Drift history
        self.history: List[Dict[str, Any]] = []
    
    def scan(
        self,
        value: float,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> bool:
        """
        Scan new observation for drift.
        
        Args:
            value: New observation (typically prediction error).
            timestamp: Timestamp of observation (for logging).
        
        Returns:
            True if drift detected.
        """
        if self._use_river:
            # River ADWIN has different API
            self.detector.update(value)
            drift = self.detector.drift_detected
        else:
            drift = self.detector.update(value)
        
        if drift:
            event = {
                'timestamp': timestamp if timestamp else pd.Timestamp.now(),
                'value': value,
            }
            self.history.append(event)
            logger.warning(f"Drift event recorded: {event}")
        
        return drift
    
    def reset(self) -> None:
        """
        Reset the drift detector.
        
        Call this after retraining the model or when starting fresh monitoring.
        """
        if self._use_river:
            # Recreate river ADWIN (no reset method)
            from river.drift import ADWIN as RiverADWIN
            self.detector = RiverADWIN(delta=self.delta)
        else:
            self.detector.reset()
        
        logger.info("DriftScanner reset")
    
    def get_drift_history(self) -> pd.DataFrame:
        """
        Get history of drift detections as DataFrame.
        
        Returns:
            DataFrame with columns: timestamp, value.
            Empty DataFrame if no drifts detected.
            
        Example:
            >>> history = scanner.get_drift_history()
            >>> print(f"Detected {len(history)} drift events")
            >>> if len(history) > 0:
            ...     print(history[['timestamp', 'value']])
        """
        if not self.history:
            return pd.DataFrame(columns=['timestamp', 'value'])
        
        return pd.DataFrame(self.history)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current detector statistics.
        
        Returns:
            Dictionary with detector statistics and drift count.
        """
        if self._use_river:
            return {
                'n_detections': len(self.history),
                'using_river': True,
            }
        else:
            stats = self.detector.get_statistics()
            stats['using_river'] = False
            return stats
