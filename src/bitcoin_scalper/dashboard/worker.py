"""
Trading Worker Thread. 

Runs the TradingEngine in a separate thread to prevent UI freezing. 
Emits signals for real-time data updates to the main UI thread.
"""

import time
import traceback
from typing import Optional, Dict, Any, List
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
import pandas as pd

from bitcoin_scalper.core. engine import TradingEngine, TradingMode
from bitcoin_scalper.core.config import TradingConfig


class TradingWorker(QThread):
    """
    Worker thread for running the trading engine.
    
    Signals:
        log_message: (str) - Log message to display
        price_update: (float, float, float, float, float, int) - OHLCV data
        signal_generated: (str, float) - Trading signal and confidence
        trade_executed: (str, float, float) - Trade type, price, volume
        metric_update: (str, Any) - Metric name and value
        error_occurred: (str) - Error message
        status_changed:  (str) - Status message
    """
    
    # Signals for UI updates
    log_message = pyqtSignal(str)
    price_update = pyqtSignal(float, float, float, float, float, int)  # OHLCV + timestamp
    signal_generated = pyqtSignal(str, float)  # signal, confidence
    trade_executed = pyqtSignal(str, float, float)  # side, price, volume
    metric_update = pyqtSignal(str, object)  # metric_name, value
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    
    def __init__(self, config: TradingConfig, model_path: Optional[str] = None):
        """
        Initialize the trading worker.
        
        Args:
            config: Trading configuration
            model_path: Path to trained model file
        """
        super().__init__()
        
        self.config = config
        self. model_path = model_path
        self.engine: Optional[TradingEngine] = None
        self._running = False
        self._paused = False
        
        # Trading state (initial balance will be set from config in _initialize_engine)
        self.balance = 0.0
        self.initial_balance = 0.0
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        
    def run(self):
        """Main worker thread loop."""
        try:
            self._running = True
            self.status_changed. emit("Initializing...")
            
            # Initialize the trading engine
            self._initialize_engine()
            
            if self.engine is None:
                self.error_occurred.emit("Failed to initialize trading engine")
                return
            
            self.status_changed.emit("Running")
            self.log_message.emit("✓ Trading engine started successfully")
            
            # Main trading loop with M1 timing (matching engine_main.py)
            tick_interval = 60  # Process ticks every 60 seconds for M1 timeframe
            last_tick_time = 0
            tick_count = 0
            
            while self._running:
                if self._paused:
                    time.sleep(0.1)
                    continue
                
                try:
                    current_time = time.time()
                    
                    # Only process if enough time has passed (matching engine_main.py logic)
                    if current_time - last_tick_time < tick_interval: 
                        time.sleep(1)  # Check every second but only process on interval
                        continue
                    
                    last_tick_time = current_time
                    
                    # Get market data from connector
                    market_data = self._fetch_market_data()
                    
                    if market_data: 
                        # Process tick through engine
                        result = self. engine. process_tick(market_data)
                        
                        # Emit updates
                        self._process_tick_result(result, market_data)
                        
                        tick_count += 1
                        
                        # Update metrics periodically
                        if tick_count % 10 == 0:
                            self._update_metrics()
                    
                except Exception as e: 
                    self.error_occurred.emit(f"Tick processing error: {str(e)}")
                    self.log_message.emit(f"❌ Error: {str(e)}")
                    time.sleep(5)  # Wait before retrying
            
            self.status_changed. emit("Stopped")
            self.log_message.emit("Trading engine stopped")
            
        except Exception as e:
            self. error_occurred.emit(f"Worker thread crashed: {str(e)}")
            self.log_message.emit(f"❌ CRITICAL ERROR: {str(e)}")
            self.log_message.emit(traceback.format_exc())
    
    def _initialize_engine(self):
        """Initialize the trading engine with configuration."""
        
        try:
            initial_balance = getattr(self. config, 'paper_initial_balance', 15000.0)
            # Utiliser Binance pour des données RÉELLES
            from bitcoin_scalper.connectors. binance_connector import BinanceConnector
            
            connector = BinanceConnector(
                api_key=self.config.binance_api_key,
                api_secret=self. config.binance_api_secret,
                testnet=self. config.binance_testnet
            )
            
            # Store initial balance for tracking
            self.initial_balance = initial_balance
            self.balance = initial_balance
            
            self.log_message.emit(f"Creating trading engine (mode: {self.config.mode})")
            
            # Prepare risk parameters dict (matching engine_main.py line 278-285)
            risk_params = {
                'max_drawdown':  self.config.max_drawdown,
                'max_daily_loss': self.config.max_daily_loss,
                'risk_per_trade': self.config.risk_per_trade,
                'max_position_size': self.config.max_position_size,
                'kelly_fraction': self.config.kelly_fraction,
                'target_volatility': self.config.target_volatility,
            }
            
            # Initialize engine (matching engine_main.py line 272-290)
            self.engine = TradingEngine(
                connector=connector,
                mode=TradingMode.ML if self.config.mode == 'ml' else TradingMode. RL,
                symbol=self.config.symbol,
                timeframe=self.config. timeframe,
                log_dir=Path("logs"),
                risk_params=risk_params,
                position_sizer=self.config.position_sizer,
                drift_detection=self.config.drift_enabled,
                safe_mode_on_drift=self.config.safe_mode_on_drift,
                meta_threshold=self.config.meta_threshold,
            )
            
            # Load model if specified
            if self. model_path and self.config.mode == 'ml': 
                self.log_message.emit(f"Loading ML model from {self.model_path}")
                success = self.engine.load_ml_model(
                    model_path=self.model_path,
                    meta_threshold=self.config.meta_threshold
                )
                if success:
                    self.log_message.emit("✓ Model loaded successfully")
                else: 
                    self.log_message. emit("⚠️  Model loading failed")
            
            self.log_message.emit("✓ Engine initialized")
            
        except Exception as e:
            self.error_occurred.emit(f"Engine initialization failed: {str(e)}")
            self.log_message.emit(f"❌ Failed to initialize engine: {str(e)}")
            self.engine = None
    
    def _fetch_market_data(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch market data from connector.
        
        Returns:
            List of OHLCV dictionaries or None if fetch fails
        """
        try: 
            if self.engine and self.engine. connector:
                # Get sufficient data for indicators (matching engine_main.py line 330)
                data = self.engine.connector.get_ohlcv(
                    symbol=self.config.symbol,
                    timeframe=self.config.timeframe,
                    limit=5000  # Match engine_main.py paper mode for proper indicator calculation
                )
                return data
        except Exception as e:
            self.log_message.emit(f"⚠️  Data fetch error: {str(e)}")
        
        return None
    
    def _process_tick_result(self, result: Dict[str, Any], market_data: List[Dict[str, Any]]):
        """
        Process tick result and emit appropriate signals.
        
        Args:
            result: Result from engine. process_tick()
            market_data: Raw market data
        """
        try: 
            # Extract latest candle data
            if market_data and len(market_data) > 0:
                latest = market_data[-1]
                self.price_update.emit(
                    float(latest.get('open', 0)),
                    float(latest.get('high', 0)),
                    float(latest.get('low', 0)),
                    float(latest.get('close', 0)),
                    float(latest.get('volume', 0)),
                    int(latest.get('timestamp', time.time()))
                )
            
            # Emit signal if generated
            signal = result.get('signal')
            confidence = result.get('confidence')
            
            if signal and confidence is not None:
                self.signal_generated.emit(signal, confidence)
                
                # Log the signal
                conf_str = f"{confidence:.2%}" if confidence else "N/A"
                self.log_message.emit(
                    f"📊 Signal: {signal. upper()} (confidence: {conf_str})"
                )
            
            # Check if trade was executed
            if signal in ['buy', 'sell'] and result.get('volume', 0) > 0:
                price = latest.get('close', 0) if market_data else 0
                volume = result.get('volume', 0)
                
                self.trade_executed.emit(signal, price, volume)
                self.trade_count += 1
                
                # Update P&L (simplified)
                self.log_message.emit(
                    f"✅ Trade executed: {signal. upper()} "
                    f"{volume:.2f} @ ${price:.2f}"
                )
            
            # Log reason if trade was rejected
            if result.get('reason'):
                self.log_message. emit(f"ℹ️  {result['reason']}")
            
            # Log errors
            if result.get('error'):
                self.log_message.emit(f"❌ {result['error']}")
        
        except Exception as e: 
            self.log_message.emit(f"⚠️  Result processing error: {str(e)}")
    
    def _update_metrics(self):
        """Update and emit trading metrics."""
        try:
            # Get balance from connector
            if self.engine and self.engine. connector:
                account = self.engine.connector._request("GET", "/account")
                self.balance = account.get('balance', self.balance)
            
            # Calculate P&L
            pnl = self.balance - self.initial_balance
            pnl_pct = (pnl / self.initial_balance) * 100
            
            # Calculate winrate
            total_trades = self.wins + self.losses
            winrate = (self.wins / total_trades * 100) if total_trades > 0 else 0
            
            # Emit metrics
            self.metric_update. emit('balance', self.balance)
            self.metric_update.emit('pnl', pnl)
            self.metric_update. emit('pnl_pct', pnl_pct)
            self.metric_update.emit('trades', self.trade_count)
            self.metric_update.emit('winrate', winrate)
            
        except Exception as e:
            self.log_message.emit(f"⚠️  Metrics update error: {str(e)}")
    
    @pyqtSlot()
    def stop(self):
        """Stop the worker thread."""
        self._running = False
        self.log_message.emit("Stopping trading engine...")
    
    @pyqtSlot()
    def pause(self):
        """Pause trading."""
        self._paused = True
        self.status_changed.emit("Paused")
        self.log_message.emit("Trading paused")
    
    @pyqtSlot()
    def resume(self):
        """Resume trading."""
        self._paused = False
        self.status_changed.emit("Running")
        self.log_message.emit("Trading resumed")
    
    @pyqtSlot(float)
    def update_meta_threshold(self, threshold: float):
        """
        Update the meta-labeling threshold.
        
        Args:
            threshold: New threshold value (0.0 to 1.0)
        """
        if self.engine and hasattr(self.engine, 'meta_threshold'):
            old_threshold = self.engine.meta_threshold
            self.engine. meta_threshold = threshold
            
            # Also update in the model if it's a MetaModel
            if self.engine.ml_model and hasattr(self.engine.ml_model, 'meta_threshold'):
                self.engine.ml_model.meta_threshold = threshold
            
            self.log_message.emit(
                f"⚙️  Meta threshold updated:  {old_threshold:. 2f} → {threshold:.2f}"
            )
