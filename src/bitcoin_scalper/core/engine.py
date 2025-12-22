"""
Trading Engine - The Brain Stem of the Trading Bot.

This is the central orchestrator that brings together all components:
- Data ingestion and feature engineering (Section 1)
- Model inference (ML or RL) (Sections 3 & 4)
- Risk management and position sizing (Section 6)
- Drift detection and monitoring (Section 5)
- Order execution

The engine operates in a hot path loop:
1. Receive market data tick
2. Update features
3. Check for concept drift
4. Get trading signal from model
5. Apply risk management
6. Execute orders

The engine is designed to be robust - errors in one tick don't crash the system.
All operations are logged for audit trail and debugging.

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Chan, E. (2017). Machine Trading.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
from enum import Enum
import logging

from bitcoin_scalper.core.logger import TradingLogger
from bitcoin_scalper.core.data_cleaner import DataCleaner
from bitcoin_scalper.core.feature_engineering import FeatureEngineering
from bitcoin_scalper.core.risk_management import RiskManager
from bitcoin_scalper.risk.sizing import KellySizer, TargetVolatilitySizer


class TradingMode(Enum):
    """Trading mode enumeration."""
    ML = "ml"  # Machine Learning (XGBoost, CatBoost)
    RL = "rl"  # Reinforcement Learning (PPO, DQN)


class MarketRegime(Enum):
    """Market regime enumeration for RL agent selection."""
    TRENDING = "trending"  # Bull/bear trends - use PPO
    RANGING = "ranging"    # Choppy/sideways - use DQN
    UNKNOWN = "unknown"    # Cannot determine - use default


class TradingEngine:
    """
    Main trading engine orchestrating all components.
    
    This is the "brain stem" that coordinates:
    - Data flow (ingestion -> features -> predictions)
    - Model inference (ML or RL based on mode)
    - Risk management (position sizing, limits)
    - Drift detection (concept drift monitoring)
    - Order execution
    
    The engine is designed to be:
    - Robust: Errors don't crash the system
    - Observable: Comprehensive logging for debugging
    - Flexible: Supports ML and RL modes
    - Production-ready: Optimized hot path
    
    Attributes:
        mode: Trading mode (ML or RL)
        logger: Structured logger instance
        data_cleaner: Data cleaning component
        feature_eng: Feature engineering component
        risk_manager: Risk management component
        ml_model: Machine learning model (if ML mode)
        rl_agent: RL agent (if RL mode)
        drift_detector: Concept drift detector
    """
    
    def __init__(
        self,
        connector,  # Generic connector (MT5RestClient, BinanceConnector, PaperMT5Client, etc.)
        mode: TradingMode = TradingMode.ML,
        symbol: str = "BTCUSD",
        timeframe: str = "M1",
        log_dir: Optional[Path] = None,
        risk_params: Optional[Dict[str, Any]] = None,
        position_sizer: str = "kelly",  # "kelly" or "target_vol"
        drift_detection: bool = True,
        safe_mode_on_drift: bool = True,
    ):
        """
        Initialize the trading engine.
        
        Args:
            connector: Exchange/broker connector for market data and execution
                      (e.g., MT5RestClient, BinanceConnector, PaperMT5Client)
            mode: Trading mode (ML or RL)
            symbol: Trading symbol
            timeframe: Data timeframe
            log_dir: Directory for logs
            risk_params: Risk management parameters
            position_sizer: Position sizing method
            drift_detection: Enable drift detection
            safe_mode_on_drift: Enter safe mode when drift detected
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.mode = mode
        self.connector = connector  # Generic connector
        self.drift_detection_enabled = drift_detection
        self.safe_mode_on_drift = safe_mode_on_drift
        self.in_safe_mode = False
        
        # Initialize logger
        self.logger = TradingLogger(log_dir=log_dir)
        self.logger.info(f"Initializing TradingEngine in {mode.value.upper()} mode")
        
        # Initialize components
        self._init_data_components()
        self._init_model_components()
        self._init_risk_components(risk_params or {}, position_sizer)
        self._init_drift_detection()
        
        # State tracking
        self.tick_count = 0
        self.last_signal = None
        self.last_prediction_time = None
        self.market_regime = MarketRegime.UNKNOWN
        
        self.logger.info("TradingEngine initialized successfully")
    
    def _init_data_components(self):
        """Initialize data cleaning and feature engineering."""
        self.logger.info("Initializing data components")
        self.data_cleaner = DataCleaner()
        self.feature_eng = FeatureEngineering()
        self.logger.info("Data components initialized")
    
    def _init_model_components(self):
        """Initialize ML or RL model."""
        self.logger.info(f"Initializing model components for {self.mode.value} mode")
        
        self.ml_model = None
        self.ml_pipeline = None
        self.features_list = None
        self.rl_agent = None
        self.rl_env = None
        
        if self.mode == TradingMode.ML:
            # ML models will be loaded separately via load_ml_model()
            self.logger.info("ML mode: model will be loaded via load_ml_model()")
        
        elif self.mode == TradingMode.RL:
            # RL agents will be loaded separately via load_rl_agent()
            self.logger.info("RL mode: agent will be loaded via load_rl_agent()")
        
        self.logger.info("Model components initialized")
    
    def _init_risk_components(self, risk_params: Dict[str, Any], position_sizer: str):
        """Initialize risk management and position sizing."""
        self.logger.info("Initializing risk components")
        
        # Risk manager
        self.risk_manager = RiskManager(
            client=self.connector,
            max_drawdown=risk_params.get('max_drawdown', 0.05),
            max_daily_loss=risk_params.get('max_daily_loss', 0.05),
            risk_per_trade=risk_params.get('risk_per_trade', 0.01),
            max_position_size=risk_params.get('max_position_size', 1.0),
        )
        
        # Position sizer
        if position_sizer == "kelly":
            self.position_sizer = KellySizer(
                kelly_fraction=risk_params.get('kelly_fraction', 0.25)
            )
            self.logger.info("Using Kelly position sizing")
        elif position_sizer == "target_vol":
            self.position_sizer = TargetVolatilitySizer(
                target_vol=risk_params.get('target_volatility', 0.15)
            )
            self.logger.info("Using target volatility position sizing")
        else:
            raise ValueError(f"Unknown position sizer: {position_sizer}")
        
        self.logger.info("Risk components initialized")
    
    def _init_drift_detection(self):
        """Initialize concept drift detection."""
        if not self.drift_detection_enabled:
            self.drift_detector = None
            self.logger.info("Drift detection disabled")
            return
        
        self.logger.info("Initializing drift detection")
        
        # Initialize drift detector using DriftScanner from validation module
        try:
            from bitcoin_scalper.validation.drift import DriftScanner
            
            # DriftScanner will automatically try to use river.drift.ADWIN if available,
            # or fall back to built-in ADWINDetector implementation
            self.drift_detector = DriftScanner(
                delta=0.002,  # Confidence level for drift detection
                max_window=10000,
                use_river=True,  # Try to use river if available
            )
            
            self.logger.info("Drift detection initialized with DriftScanner")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize drift detector: {e}")
            self.drift_detector = None
            self.logger.warning("Drift detection disabled due to initialization error")
        
        # Track errors for drift detection
        self.prediction_errors = []
        self.max_error_window = 100
    
    def load_ml_model(
        self,
        model_path: str,
        features_list: Optional[List[str]] = None
    ):
        """
        Load a trained ML model.
        
        Args:
            model_path: Path to model file
            features_list: List of feature names (if None, will try to load from file)
        """
        if self.mode != TradingMode.ML:
            raise ValueError("Cannot load ML model in RL mode")
        
        self.logger.info(f"Loading ML model from {model_path}")
        
        try:
            import joblib
            from bitcoin_scalper.core.export import load_objects
            
            # Try to load using the export module
            try:
                objects = load_objects(model_path)
                self.ml_pipeline = objects.get('pipeline')
                self.ml_model = objects.get('model')
                self.features_list = None  # load_objects doesn't return features_list
                self.logger.info("ML model loaded successfully via load_objects")
            except Exception as e:
                # Fallback to direct load
                self.logger.warning(f"load_objects failed: {e}, trying direct load")
                
                # Load CatBoost model properly
                try:
                    from catboost import CatBoostClassifier
                    # Load model as class method
                    self.ml_model = CatBoostClassifier().load_model(f"{model_path}_model.cbm")
                except ImportError:
                    self.logger.warning("CatBoost not installed, trying joblib for XGBoost or other models")
                    # Try joblib as fallback (for XGBoost or other models)
                    try:
                        self.ml_model = joblib.load(f"{model_path}_model.pkl")
                    except Exception as e3:
                        self.logger.error(f"Failed to load model with joblib: {e3}")
                        raise
                except Exception as e2:
                    self.logger.warning(f"CatBoost load failed: {e2}, trying joblib")
                    # Try joblib as last resort (for XGBoost or other models)
                    self.ml_model = joblib.load(f"{model_path}_model.pkl")
                
                if features_list:
                    self.features_list = features_list
                else:
                    # Try to load features list
                    try:
                        self.features_list = joblib.load(f"{model_path}_features.pkl")
                    except:
                        self.logger.warning("Could not load features list")
                        self.features_list = None
                
                self.logger.info("ML model loaded successfully via direct load")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load ML model: {e}")
            return False
    
    def load_rl_agent(
        self,
        agent_path: str,
        agent_type: str = "ppo"  # "ppo" or "dqn"
    ):
        """
        Load a trained RL agent.
        
        Args:
            agent_path: Path to agent checkpoint
            agent_type: Type of agent ("ppo" or "dqn")
        """
        if self.mode != TradingMode.RL:
            raise ValueError("Cannot load RL agent in ML mode")
        
        self.logger.info(f"Loading RL agent ({agent_type}) from {agent_path}")
        
        try:
            from stable_baselines3 import PPO, DQN
            
            if agent_type.lower() == "ppo":
                self.rl_agent = PPO.load(agent_path)
                self.logger.info("PPO agent loaded successfully")
            elif agent_type.lower() == "dqn":
                self.rl_agent = DQN.load(agent_path)
                self.logger.info("DQN agent loaded successfully")
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load RL agent: {e}")
            return False
    
    def process_tick(
        self,
        market_data: Union[Dict[str, Any], pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Process a market data tick - THE HOT PATH.
        
        This is the critical method called for each market update.
        It orchestrates the entire trading pipeline:
        1. Clean and validate data
        2. Compute features
        3. Check for drift
        4. Get prediction/signal
        5. Apply risk management
        6. Generate order instructions
        
        The method is wrapped in try/except to ensure one bad tick
        doesn't crash the entire system.
        
        Args:
            market_data: Raw market data (dict or DataFrame)
        
        Returns:
            Dict with:
            - signal: Trading signal ('buy', 'sell', 'hold', None)
            - confidence: Model confidence (if available)
            - volume: Recommended position size
            - reason: Why this decision was made
            - drift_detected: Whether drift was detected
            - error: Error message (if any)
        """
        self.tick_count += 1
        start_time = time.time()
        
        result = {
            'signal': None,
            'confidence': None,
            'volume': 0.0,
            'reason': '',
            'drift_detected': False,
            'error': None,
            'tick_number': self.tick_count,
        }
        
        try:
            # Step 1: Clean and validate data
            if isinstance(market_data, dict):
                cleaned_data = self.data_cleaner.clean_ohlcv([market_data])
                if not cleaned_data:
                    result['error'] = 'Data cleaning failed'
                    result['reason'] = 'Invalid market data'
                    return result
                df = pd.DataFrame(cleaned_data)
            elif isinstance(market_data, list):
                # Handle list of dictionaries (from paper trading or other sources)
                df = pd.DataFrame(market_data)
            else:
                df = market_data.copy()
            
            # --- COMPATIBILITY PATCH: Rename standard columns to Legacy MT5 format ---
            # This ensures the features match the trained model's expectations (e.g., <TICKVOL>)
            if 'close' in df.columns:
                rename_map = {
                    'open': '<OPEN>',
                    'high': '<HIGH>',
                    'low': '<LOW>',
                    'close': '<CLOSE>',
                    'tick_volume': '<TICKVOL>',
                    'volume': '<TICKVOL>',  # Handle both naming conventions
                    'real_volume': '<VOL>'
                }
                df = df.rename(columns=rename_map)
            # -----------------------------------------------------------------------
            
            # Step 2: Compute features
            try:
                df = self.feature_eng.add_indicators(df)
                df = self.feature_eng.add_features(df)
            except Exception as e:
                self.logger.error(f"Feature engineering failed: {e}")
                result['error'] = f'Feature engineering error: {e}'
                result['reason'] = 'Cannot compute features'
                return result
            
            # --- PREFIXING LOGIC: Add timeframe prefix to raw columns ---
            # The model was trained on features like "1min_<CLOSE>", "1min_<TICKVOL>", etc.
            # We need to create these prefixed columns from the existing raw columns.
            try:
                # Determine the prefix based on timeframe
                prefix = self._get_timeframe_prefix(self.timeframe)
                
                # Raw tags that may need prefixing
                raw_tags = ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>', '<VOL>']
                
                # Create prefixed columns if the raw column exists
                for tag in raw_tags:
                    if tag in df.columns:
                        prefixed_col = prefix + tag
                        df[prefixed_col] = df[tag]
                
            except Exception as e:
                self.logger.warning(f"Prefixing logic failed: {e}")
                # Don't fail the entire tick processing for this
            # ----------------------------------------------------------------
            
            # Step 3: Check for drift (if enabled and not in safe mode)
            if self.drift_detection_enabled and not self.in_safe_mode:
                drift_detected = self._check_drift(df)
                result['drift_detected'] = drift_detected
                
                if drift_detected:
                    self.logger.log_drift(
                        drift_detected=True,
                        action_taken='safe_mode' if self.safe_mode_on_drift else 'continue'
                    )
                    
                    if self.safe_mode_on_drift:
                        self.in_safe_mode = True
                        result['signal'] = 'hold'
                        result['reason'] = 'Drift detected - entering safe mode'
                        return result
            
            # Step 4: Get signal from model
            signal, confidence = self._get_signal(df)
            result['signal'] = signal
            result['confidence'] = confidence
            
            if signal is None or signal == 'hold':
                result['reason'] = 'No trading opportunity'
                return result
            
            # Step 5: Apply risk management
            risk_check = self.risk_manager.can_open_position(self.symbol, 0.01)
            
            if not risk_check['allowed']:
                result['signal'] = 'hold'
                result['reason'] = f"Risk check failed: {risk_check['reason']}"
                self.logger.warning(
                    f"Order blocked by risk manager: {risk_check['reason']}"
                )
                return result
            
            # Step 6: Calculate position size
            try:
                # Get price from either renamed or original column
                price_col = '<CLOSE>' if '<CLOSE>' in df.columns else 'close'
                price = df[price_col].iloc[-1]
                volume = self._calculate_position_size(
                    signal=signal,
                    price=price,
                    confidence=confidence,
                    df=df
                )
                result['volume'] = volume
                result['reason'] = f"Signal: {signal}, Confidence: {confidence:.2f}"
            except Exception as e:
                self.logger.error(f"Position sizing failed: {e}")
                result['error'] = f'Position sizing error: {e}'
                result['signal'] = 'hold'
                result['reason'] = 'Cannot calculate position size'
                return result
            
            # Log the signal
            self.logger.log_signal(
                symbol=self.symbol,
                signal=signal,
                confidence=confidence,
                features={
                    'price': float(df['<CLOSE>' if '<CLOSE>' in df.columns else 'close'].iloc[-1]),
                    'volume_suggested': volume,
                }
            )
            
            return result
            
        except Exception as e:
            # Catch-all for unexpected errors
            self.logger.error(
                f"Unexpected error in process_tick: {e}",
                tick_number=self.tick_count
            )
            result['error'] = f'Unexpected error: {e}'
            result['reason'] = 'System error'
            return result
        
        finally:
            # Log performance metric
            elapsed = (time.time() - start_time) * 1000  # ms
            self.logger.log_metric(
                'tick_processing_time',
                elapsed,
                unit='ms',
                tick_number=self.tick_count
            )
    
    def _get_timeframe_prefix(self, timeframe: str) -> str:
        """
        Map timeframe string to prefix format used in feature columns.
        
        Args:
            timeframe: Timeframe string (e.g., "M1", "1m", "M5", "5m")
        
        Returns:
            Prefix string (e.g., "1min_", "5min_")
        """
        # Normalize timeframe to lowercase for consistent handling
        tf_lower = timeframe.lower()
        
        # Map various timeframe formats to prefix
        timeframe_map = {
            'm1': '1min_',
            '1m': '1min_',
            '1min': '1min_',
            'm5': '5min_',
            '5m': '5min_',
            '5min': '5min_',
            'm15': '15min_',
            '15m': '15min_',
            '15min': '15min_',
            'm30': '30min_',
            '30m': '30min_',
            '30min': '30min_',
            'h1': '1h_',
            '1h': '1h_',
            'h4': '4h_',
            '4h': '4h_',
            'd1': '1d_',
            '1d': '1d_',
        }
        
        return timeframe_map.get(tf_lower, '1min_')  # Default to 1min_ if unknown
    
    def _get_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
        """
        Get trading signal from model (ML or RL).
        
        Args:
            df: DataFrame with features
        
        Returns:
            Tuple of (signal, confidence)
            signal: 'buy', 'sell', 'hold', or None
            confidence: Model confidence (0-1) or None
        """
        try:
            if self.mode == TradingMode.ML:
                return self._get_ml_signal(df)
            elif self.mode == TradingMode.RL:
                return self._get_rl_signal(df)
            else:
                self.logger.error(f"Unknown trading mode: {self.mode}")
                return None, None
                
        except Exception as e:
            self.logger.error(f"Error getting signal: {e}")
            return None, None
    
    def _get_ml_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
        """Get signal from ML model."""
        if self.ml_model is None:
            self.logger.warning("ML model not loaded")
            
            # TASK 3: Temporary "coin flip" logic for debugging paper trading
            # Generate random signal with 50% probability
            import random
            if random.random() < 0.50:  # 50% chance
                signal = random.choice(['buy', 'sell'])
                confidence = random.uniform(0.6, 0.8)  # Simulated confidence
                self.logger.info(f"[DEBUG] Random Coin Flip Signal: {signal} (confidence: {confidence:.2f})")
                return signal, confidence
            
            return None, None
        
        try:
            # Prepare features
            if self.features_list:
                missing_cols = [col for col in self.features_list if col not in df.columns]
                if missing_cols:
                    self.logger.warning(f"Missing features: {missing_cols}")
                    # Use available features
                    available_features = [col for col in self.features_list if col in df.columns]
                    X = df[available_features].tail(1)
                else:
                    X = df[self.features_list].tail(1)
            else:
                # Use all numeric columns
                X = df.select_dtypes(include=[np.number]).tail(1)
            
            # Get prediction
            if self.ml_pipeline:
                pred = self.ml_pipeline.predict(X)[0]
            else:
                pred = self.ml_model.predict(X)[0]
            
            # Get confidence if available
            confidence = None
            try:
                if hasattr(self.ml_model, 'predict_proba'):
                    proba = self.ml_model.predict_proba(X)[0]
                    confidence = float(np.max(proba))
                elif hasattr(self.ml_pipeline, 'predict_proba'):
                    proba = self.ml_pipeline.predict_proba(X)[0]
                    confidence = float(np.max(proba))
            except:
                pass
            
            # Map prediction to signal
            if pred == 1:
                signal = 'buy'
            elif pred == -1:
                signal = 'sell'
            else:
                signal = 'hold'
            
            return signal, confidence
            
        except Exception as e:
            self.logger.error(f"ML prediction failed: {e}")
            return None, None
    
    def _get_rl_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
        """Get signal from RL agent."""
        if self.rl_agent is None:
            self.logger.warning("RL agent not loaded")
            
            # TASK 3: Temporary "coin flip" logic for debugging paper trading
            # Generate random signal with 50% probability
            import random
            if random.random() < 0.50:  # 50% chance
                signal = random.choice(['buy', 'sell'])
                confidence = random.uniform(0.6, 0.8)  # Simulated confidence
                self.logger.info(f"[DEBUG] Random Coin Flip Signal: {signal} (confidence: {confidence:.2f})")
                return signal, confidence
            
            return None, None
        
        try:
            # Prepare observation (last N rows as state)
            # This depends on how the RL environment was set up
            # For now, use a simple approach
            obs = df.select_dtypes(include=[np.number]).tail(30).values.flatten()
            
            # Pad or truncate to expected size (depends on agent training)
            # This is a placeholder - actual implementation depends on env setup
            expected_size = self.rl_agent.observation_space.shape[0] if hasattr(self.rl_agent, 'observation_space') else len(obs)
            if len(obs) < expected_size:
                obs = np.pad(obs, (0, expected_size - len(obs)))
            elif len(obs) > expected_size:
                obs = obs[:expected_size]
            
            # Get action
            action, _states = self.rl_agent.predict(obs, deterministic=True)
            
            # Map action to signal
            # Assuming: 0=hold, 1=buy, 2=sell
            if action == 1:
                signal = 'buy'
            elif action == 2:
                signal = 'sell'
            else:
                signal = 'hold'
            
            # RL agents don't typically output confidence
            # Could use value function as proxy
            confidence = None
            
            return signal, confidence
            
        except Exception as e:
            self.logger.error(f"RL prediction failed: {e}")
            return None, None
    
    def _check_drift(self, df: pd.DataFrame) -> bool:
        """
        Check for concept drift using DriftScanner.
        
        This method monitors model prediction errors (or price volatility if no model)
        to detect distributional changes in the data. When drift is detected, the
        system can enter safe mode to avoid trading on an invalid model.
        
        Args:
            df: Current market data with features
        
        Returns:
            True if drift detected, False otherwise
        """
        if self.drift_detector is None:
            return False
        
        try:
            # Calculate a drift metric to monitor
            # Option 1: If we have recent predictions, use prediction error
            # Option 2: Use price volatility as a proxy for regime change
            
            # For now, use price volatility as the drift metric
            # In a more advanced setup, you'd track actual prediction errors
            price_col = '<CLOSE>' if '<CLOSE>' in df.columns else 'close'
            if price_col in df.columns and len(df) >= 20:
                # Calculate recent volatility
                returns = df[price_col].pct_change().dropna()
                if len(returns) >= 2:
                    recent_volatility = float(returns.tail(20).std())
                    
                    # Feed to drift detector
                    drift_detected = self.drift_detector.scan(
                        value=recent_volatility,
                        timestamp=pd.Timestamp.now()
                    )
                    
                    return drift_detected
            
            return False
            
        except Exception as e:
            self.logger.error(f"Drift detection failed: {e}")
            return False
    
    def _calculate_position_size(
        self,
        signal: str,
        price: float,
        confidence: Optional[float],
        df: pd.DataFrame
    ) -> float:
        """
        Calculate position size using configured position sizer.
        
        Args:
            signal: Trading signal
            price: Current price
            confidence: Model confidence
            df: Market data for volatility calculation
        
        Returns:
            Position size (volume in lots)
        """
        try:
            # Get account info
            account = self.connector._request("GET", "/account")
            capital = account.get('balance', 10000.0)
            
            # Calculate size based on position sizer type
            if isinstance(self.position_sizer, KellySizer):
                # Kelly sizing needs win probability and payoff ratio
                # Use confidence as proxy for win probability
                win_prob = confidence if confidence else 0.55
                
                # Estimate payoff ratio from recent trades or use default
                payoff_ratio = 1.5  # Default 1.5:1 reward/risk
                
                size = self.position_sizer.calculate_size(
                    capital=capital,
                    price=price,
                    win_prob=win_prob,
                    payoff_ratio=payoff_ratio
                )
            
            elif isinstance(self.position_sizer, TargetVolatilitySizer):
                # Target vol sizing needs asset volatility
                if 'atr' in df.columns and len(df) > 1:
                    volatility = float(df['atr'].iloc[-1]) / price
                else:
                    # Estimate from recent price changes
                    price_col = '<CLOSE>' if '<CLOSE>' in df.columns else 'close'
                    returns = df[price_col].pct_change().dropna()
                    volatility = float(returns.std()) if len(returns) > 0 else 0.02
                
                size = self.position_sizer.calculate_size(
                    capital=capital,
                    price=price,
                    asset_volatility=volatility
                )
            
            else:
                # Fallback to fixed percentage
                size = capital * 0.01 / price
            
            # Apply risk manager limits
            max_size = self.risk_manager.max_position_size
            size = min(size, max_size)
            
            # Ensure minimum lot size
            min_size = 0.01
            size = max(size, min_size)
            
            return round(size, 2)
            
        except Exception as e:
            self.logger.error(f"Position sizing failed: {e}")
            return 0.01  # Return minimum size as fallback
    
    def execute_order(
        self,
        signal: str,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a trading order.
        
        Args:
            signal: 'buy' or 'sell'
            volume: Position size
            sl: Stop loss price (optional)
            tp: Take profit price (optional)
        
        Returns:
            Dict with execution result
        """
        if signal not in ['buy', 'sell']:
            return {'success': False, 'error': 'Invalid signal'}
        
        try:
            # Execute order via connector
            result = self.connector.send_order(
                symbol=self.symbol,
                action=signal,
                volume=volume,
                sl=sl,
                tp=tp
            )
            
            # Log the trade
            self.logger.log_trade(
                symbol=self.symbol,
                side=signal,
                volume=volume,
                price=None,  # Market order
                sl=sl,
                tp=tp,
                reason=f"Engine execution tick #{self.tick_count}"
            )
            
            return {'success': True, 'result': result}
            
        except Exception as e:
            self.logger.error(
                f"Order execution failed: {e}",
                symbol=self.symbol,
                signal=signal,
                volume=volume
            )
            return {'success': False, 'error': str(e)}
    
    def reset_safe_mode(self):
        """Exit safe mode (after model retraining or manual intervention)."""
        self.in_safe_mode = False
        self.logger.info("Safe mode reset")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current engine status.
        
        Returns:
            Dict with status information
        """
        return {
            'mode': self.mode.value,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'tick_count': self.tick_count,
            'in_safe_mode': self.in_safe_mode,
            'last_signal': self.last_signal,
            'drift_detection_enabled': self.drift_detection_enabled,
            'model_loaded': self.ml_model is not None or self.rl_agent is not None,
        }
