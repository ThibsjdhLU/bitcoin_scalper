"""
Structured Logging System for Trading Bot.

This module provides comprehensive logging for:
- Trade execution logs (what was traded, when, why)
- Error logs (what went wrong, stack traces)
- Performance metrics (latency, throughput, PnL)
- Decision audit trail (for debugging "Why did the bot do that?")

Features:
- JSON structured logging for machine parsing
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Separate log files for trades, errors, and metrics
- Real-time log rotation to prevent disk space issues
- Thread-safe logging for concurrent operations

References:
    Python Logging Best Practices
    Structured Logging: https://www.structlog.org/
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler
import threading

# Thread-local storage for request context
_context = threading.local()


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs logs as JSON for easy parsing and analysis.
    Includes timestamp, level, message, and custom fields.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields from record
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Add context if available
        if hasattr(_context, 'fields'):
            log_data.update(_context.fields)
        
        return json.dumps(log_data)


class TradingLogger:
    """
    Main logger for the trading bot.
    
    Provides specialized logging methods for different types of events:
    - Trade logs: Execution details
    - Decision logs: Why signals were generated
    - Error logs: Exceptions and failures
    - Metric logs: Performance measurements
    
    All logs are structured (JSON) for easy parsing and analysis.
    """
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB per file
        backup_count: int = 5,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        """
        Initialize the trading logger.
        
        Args:
            log_dir: Directory for log files. If None, uses ./logs
            max_bytes: Maximum size per log file before rotation
            backup_count: Number of backup files to keep
            console_level: Minimum level for console output
            file_level: Minimum level for file output
        """
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create separate loggers for different purposes
        self.main_logger = self._setup_logger(
            "trading_engine",
            self.log_dir / "engine.log",
            max_bytes,
            backup_count,
            console_level,
            file_level
        )
        
        self.trade_logger = self._setup_logger(
            "trades",
            self.log_dir / "trades.log",
            max_bytes,
            backup_count,
            logging.INFO,  # Always log trades
            logging.INFO
        )
        
        self.error_logger = self._setup_logger(
            "errors",
            self.log_dir / "errors.log",
            max_bytes,
            backup_count,
            logging.ERROR,
            logging.ERROR
        )
        
        self.metrics_logger = self._setup_logger(
            "metrics",
            self.log_dir / "metrics.log",
            max_bytes,
            backup_count,
            logging.WARNING,  # Don't spam console with metrics
            logging.INFO
        )
    
    def _setup_logger(
        self,
        name: str,
        log_file: Path,
        max_bytes: int,
        backup_count: int,
        console_level: int,
        file_level: int
    ) -> logging.Logger:
        """Setup a logger with file and console handlers."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # Capture everything, handlers will filter
        logger.propagate = False  # Don't propagate to root logger
        
        # Remove existing handlers (for reinit)
        logger.handlers.clear()
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        # Use simple format for console readability
        console_formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s][%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_trade(
        self,
        symbol: str,
        side: str,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        reason: str = "",
        **kwargs
    ):
        """
        Log a trade execution.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            volume: Trade volume
            price: Execution price (None if market order)
            sl: Stop loss level
            tp: Take profit level
            reason: Why this trade was made
            **kwargs: Additional fields
        """
        trade_data = {
            'symbol': symbol,
            'side': side,
            'volume': volume,
            'price': price,
            'stop_loss': sl,
            'take_profit': tp,
            'reason': reason,
            **kwargs
        }
        
        # Create log record with extra fields
        record = self.trade_logger.makeRecord(
            self.trade_logger.name,
            logging.INFO,
            "(trade)",
            0,
            f"TRADE: {side.upper()} {volume} {symbol} @ {price}",
            (),
            None
        )
        record.extra_fields = trade_data
        self.trade_logger.handle(record)
    
    def log_signal(
        self,
        symbol: str,
        signal: str,
        confidence: Optional[float] = None,
        features: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Log a trading signal generation.
        
        Args:
            symbol: Trading symbol
            signal: Signal type ('buy', 'sell', 'hold')
            confidence: Model confidence (0-1)
            features: Key features that influenced the decision
            **kwargs: Additional fields
        """
        signal_data = {
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'features': features or {},
            **kwargs
        }
        
        record = self.main_logger.makeRecord(
            self.main_logger.name,
            logging.INFO,
            "(signal)",
            0,
            f"SIGNAL: {signal.upper()} for {symbol} (confidence: {confidence})",
            (),
            None
        )
        record.extra_fields = signal_data
        self.main_logger.handle(record)
    
    def log_error(
        self,
        error: Exception,
        context: str = "",
        **kwargs
    ):
        """
        Log an error with context.
        
        Args:
            error: The exception
            context: Where the error occurred
            **kwargs: Additional fields
        """
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            **kwargs
        }
        
        record = self.error_logger.makeRecord(
            self.error_logger.name,
            logging.ERROR,
            "(error)",
            0,
            f"ERROR in {context}: {error}",
            (),
            (type(error), error, error.__traceback__)
        )
        record.extra_fields = error_data
        self.error_logger.handle(record)
    
    def log_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "",
        **kwargs
    ):
        """
        Log a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            **kwargs: Additional fields
        """
        metric_data = {
            'metric': metric_name,
            'value': value,
            'unit': unit,
            **kwargs
        }
        
        record = self.metrics_logger.makeRecord(
            self.metrics_logger.name,
            logging.INFO,
            "(metric)",
            0,
            f"METRIC: {metric_name} = {value} {unit}",
            (),
            None
        )
        record.extra_fields = metric_data
        self.metrics_logger.handle(record)
    
    def log_drift(
        self,
        drift_detected: bool,
        drift_score: Optional[float] = None,
        action_taken: str = "",
        **kwargs
    ):
        """
        Log concept drift detection.
        
        Args:
            drift_detected: Whether drift was detected
            drift_score: Drift detection score
            action_taken: What action was taken (retrain, safe mode, etc.)
            **kwargs: Additional fields
        """
        drift_data = {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'action_taken': action_taken,
            **kwargs
        }
        
        level = logging.WARNING if drift_detected else logging.INFO
        record = self.main_logger.makeRecord(
            self.main_logger.name,
            level,
            "(drift)",
            0,
            f"DRIFT: {'DETECTED' if drift_detected else 'OK'} - {action_taken}",
            (),
            None
        )
        record.extra_fields = drift_data
        self.main_logger.handle(record)
    
    def info(self, message: str, **kwargs):
        """Log an info message."""
        if kwargs:
            record = self.main_logger.makeRecord(
                self.main_logger.name, logging.INFO, "(info)", 0, message, (), None
            )
            record.extra_fields = kwargs
            self.main_logger.handle(record)
        else:
            self.main_logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        if kwargs:
            record = self.main_logger.makeRecord(
                self.main_logger.name, logging.WARNING, "(warning)", 0, message, (), None
            )
            record.extra_fields = kwargs
            self.main_logger.handle(record)
        else:
            self.main_logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log an error message."""
        if kwargs:
            record = self.error_logger.makeRecord(
                self.error_logger.name, logging.ERROR, "(error)", 0, message, (), None
            )
            record.extra_fields = kwargs
            self.error_logger.handle(record)
        else:
            self.error_logger.error(message)
    
    def debug(self, message: str, **kwargs):
        """Log a debug message."""
        if kwargs:
            record = self.main_logger.makeRecord(
                self.main_logger.name, logging.DEBUG, "(debug)", 0, message, (), None
            )
            record.extra_fields = kwargs
            self.main_logger.handle(record)
        else:
            self.main_logger.debug(message)


def get_logger(
    log_dir: Optional[Path] = None,
    console_level: int = logging.INFO
) -> TradingLogger:
    """
    Get or create the trading logger instance.
    
    Args:
        log_dir: Directory for log files
        console_level: Console logging level
    
    Returns:
        TradingLogger instance
    """
    return TradingLogger(log_dir=log_dir, console_level=console_level)
