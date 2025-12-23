#!/usr/bin/env python3
"""
Main Entry Point for the Trading Engine.

This script provides command-line interface for running the trading bot in different modes:
- live: Real trading with actual orders
- paper: Paper trading (simulation without real orders)
- backtest: Historical backtesting

Usage:
    python engine_main.py --mode live --config config/engine_config.yaml
    python engine_main.py --mode paper --config config/engine_config.yaml
    python engine_main.py --mode backtest --config config/engine_config.yaml --start 2024-01-01 --end 2024-12-31

The engine orchestrates all components:
- Data ingestion and feature engineering
- Model inference (ML or RL)
- Risk management and position sizing
- Drift detection and monitoring
- Order execution

All operations are logged for audit trail and debugging.
"""

import argparse
import sys
import time
import signal
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bitcoin_scalper.core.engine import TradingEngine, TradingMode
from bitcoin_scalper.core.config import TradingConfig
from bitcoin_scalper.core.logger import TradingLogger
from bitcoin_scalper.connectors.mt5_rest_client import MT5RestClient
from bitcoin_scalper.connectors.binance_connector import BinanceConnector
from bitcoin_scalper.connectors.paper import PaperMT5Client
from bitcoin_scalper.core.data_cleaner import DataCleaner
from bitcoin_scalper.core.backtesting import Backtester

# Global flag for graceful shutdown
shutdown_flag = False


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_flag
    print("\n[INFO] Shutdown signal received. Stopping engine...")
    shutdown_flag = True


def run_live_mode(config: TradingConfig, logger: TradingLogger):
    """
    Run the trading engine in live mode.
    
    This mode connects to the broker/exchange and executes real trades.
    """
    logger.info("Starting engine in LIVE mode")
    
    # Initialize connector based on exchange configuration
    if config.exchange == "binance":
        logger.info("Initializing Binance connector")
        connector = BinanceConnector(
            api_key=config.binance_api_key,
            api_secret=config.binance_api_secret,
            testnet=config.binance_testnet
        )
    elif config.exchange == "mt5":
        logger.info("Initializing MT5 connector")
        connector = MT5RestClient(
            base_url=config.mt5_rest_url,
            api_key=config.mt5_api_key
        )
    else:
        logger.error(f"Unsupported exchange: {config.exchange}")
        raise ValueError(f"Unsupported exchange: {config.exchange}")
    
    # Determine trading mode
    mode = TradingMode.ML if config.mode.lower() == "ml" else TradingMode.RL
    
    # Initialize trading engine
    engine = TradingEngine(
        connector=connector,
        mode=mode,
        symbol=config.symbol,
        timeframe=config.timeframe,
        log_dir=Path(config.log_dir) if config.log_dir else None,
        risk_params={
            'max_drawdown': config.max_drawdown,
            'max_daily_loss': config.max_daily_loss,
            'risk_per_trade': config.risk_per_trade,
            'max_position_size': config.max_position_size,
            'kelly_fraction': config.kelly_fraction,
            'target_volatility': config.target_volatility,
        },
        position_sizer=config.position_sizer,
        drift_detection=config.drift_enabled,
        safe_mode_on_drift=config.safe_mode_on_drift,
    )
    
    # Load model
    if config.model_path:
        if mode == TradingMode.ML:
            success = engine.load_ml_model(config.model_path)
            if not success:
                logger.error("Failed to load ML model. Exiting.")
                return
        else:
            success = engine.load_rl_agent(config.model_path, config.model_type)
            if not success:
                logger.error("Failed to load RL agent. Exiting.")
                return
    else:
        logger.warning("No model path specified. Using default strategy.")
    
    logger.info("Engine initialized successfully. Starting main loop...")
    
    # Main trading loop
    tick_interval = 60  # Check market every 60 seconds (for M1)
    last_tick_time = 0
    
    while not shutdown_flag:
        try:
            current_time = time.time()
            
            # Only process if enough time has passed
            if current_time - last_tick_time < tick_interval:
                time.sleep(1)
                continue
            
            last_tick_time = current_time
            
            # Get latest market data
            try:
                # Fetch OHLCV data from connector
                if config.exchange == "binance":
                    # Binance returns DataFrame directly
                    df = connector.fetch_ohlcv(
                        config.symbol,
                        timeframe=config.timeframe,
                        limit=1500  # Get last 1500 candles for indicators
                    )
                    
                    if df.empty or len(df) < 30:
                        logger.warning("Insufficient market data")
                        continue
                    
                    # Convert DataFrame to list of dicts for compatibility
                    ohlcv = df.reset_index().to_dict('records')
                else:
                    # MT5 returns list of dicts
                    ohlcv = connector.get_ohlcv(
                        config.symbol,
                        timeframe=config.timeframe,
                        limit=1500  # Get last 1500 candles for indicators
                    )
                    
                    if not ohlcv or len(ohlcv) < 30:
                        logger.warning("Insufficient market data")
                        continue
                
            except Exception as e:
                logger.error(f"Failed to fetch market data: {e}")
                continue
            
            # Process the tick
            result = engine.process_tick(ohlcv)
            
            # Log the result
            if result.get('error'):
                logger.error(
                    f"Tick processing error: {result['error']}",
                    tick_number=result['tick_number']
                )
                continue
            
            signal = result.get('signal')
            volume = result.get('volume', 0)
            
            if signal and signal != 'hold' and volume > 0:
                logger.info(
                    f"Signal: {signal}, Volume: {volume}, Reason: {result.get('reason')}",
                    confidence=result.get('confidence'),
                    drift_detected=result.get('drift_detected')
                )
                
                # Calculate SL/TP if enabled
                sl, tp = None, None
                if config.use_sl_tp:
                    df = pd.DataFrame(ohlcv)
                    close_price = df['close'].iloc[-1]
                    
                    # Try ATR-based SL/TP
                    if 'atr' in df.columns:
                        atr = df['atr'].iloc[-1]
                        if signal == 'buy':
                            sl = close_price - config.sl_atr_mult * atr
                            tp = close_price + config.tp_atr_mult * atr
                        else:  # sell
                            sl = close_price + config.sl_atr_mult * atr
                            tp = close_price - config.tp_atr_mult * atr
                    else:
                        # Fallback to percentage-based
                        if signal == 'buy':
                            sl = close_price * (1 - config.default_sl_pct)
                            tp = close_price * (1 + config.default_tp_pct)
                        else:  # sell
                            sl = close_price * (1 + config.default_sl_pct)
                            tp = close_price * (1 - config.default_tp_pct)
                
                # Execute order
                execution_result = engine.execute_order(
                    signal=signal,
                    volume=volume,
                    sl=sl,
                    tp=tp
                )
                
                if execution_result.get('success'):
                    logger.info(f"Order executed successfully: {signal} {volume}")
                else:
                    logger.error(f"Order execution failed: {execution_result.get('error')}")
            
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            time.sleep(5)  # Wait before retrying
    
    logger.info("Engine stopped")


def run_paper_mode(config: TradingConfig, logger: TradingLogger):
    """
    Run the trading engine in paper trading mode.
    
    This mode simulates trades without executing real orders.
    Uses PaperMT5Client to track simulated positions and P&L.
    """
    logger.info("="*60)
    logger.info("Starting engine in PAPER mode")
    logger.info("SIMULATION ONLY - NO REAL TRADES WILL BE EXECUTED")
    logger.info("="*60)
    
    # Initialize paper trading client
    paper_client = PaperMT5Client(
        initial_balance=config.paper_initial_balance if hasattr(config, 'paper_initial_balance') else 15000.0,
        enable_slippage=config.paper_simulate_slippage if hasattr(config, 'paper_simulate_slippage') else False,
    )
    
    # Set initial price for symbol
    # In production, this would be fetched from real market data
    initial_price = 50000.0  # Default BTC price
    paper_client.set_price(config.symbol, initial_price)
    
    logger.info(f"Paper account initialized with ${paper_client.balance:.2f}")
    
    # Determine trading mode
    mode = TradingMode.ML if config.mode.lower() == "ml" else TradingMode.RL
    
    # Initialize trading engine with paper client
    engine = TradingEngine(
        connector=paper_client,
        mode=mode,
        symbol=config.symbol,
        timeframe=config.timeframe,
        log_dir=Path(config.log_dir) if config.log_dir else None,
        risk_params={
            'max_drawdown': config.max_drawdown,
            'max_daily_loss': config.max_daily_loss,
            'risk_per_trade': config.risk_per_trade,
            'max_position_size': config.max_position_size,
            'kelly_fraction': config.kelly_fraction,
            'target_volatility': config.target_volatility,
        },
        position_sizer=config.position_sizer,
        drift_detection=config.drift_enabled,
        safe_mode_on_drift=config.safe_mode_on_drift,
    )
    
    # Load model
    if config.model_path:
        if mode == TradingMode.ML:
            success = engine.load_ml_model(config.model_path)
            if not success:
                logger.warning("Failed to load ML model. Running without model.")
        else:
            success = engine.load_rl_agent(config.model_path, config.model_type)
            if not success:
                logger.warning("Failed to load RL agent. Running without model.")
    else:
        logger.warning("No model path specified. Engine will not generate signals.")
    
    logger.info("Paper trading engine initialized successfully. Starting main loop...")
    
    # Main trading loop
    tick_interval = 60  # Check market every 60 seconds (for M1)
    last_tick_time = 0
    
    while not shutdown_flag:
        try:
            current_time = time.time()
            
            # Only process if enough time has passed
            if current_time - last_tick_time < tick_interval:
                time.sleep(1)
                continue
            
            last_tick_time = current_time
            
            # Get simulated market data from paper client
            try:
                ohlcv = paper_client.get_ohlcv(
                    config.symbol,
                    timeframe=config.timeframe,
                    limit=5000  # Get last 1500 candles for indicators
                )
                
                if not ohlcv or len(ohlcv) < 30:
                    logger.warning("Insufficient market data")
                    continue
                
            except Exception as e:
                logger.error(f"Failed to generate market data: {e}")
                continue
            
            # Process the tick
            result = engine.process_tick(ohlcv)
            
            # Log the result
            if result.get('error'):
                logger.error(
                    f"Tick processing error: {result['error']}",
                    tick_number=result['tick_number']
                )
                continue
            
            signal = result.get('signal')
            volume = result.get('volume', 0)
            
            if signal and signal != 'hold' and volume > 0:
                # Get current price
                df = pd.DataFrame(ohlcv)
                close_price = df['close'].iloc[-1]
                
                logger.info("="*60)
                logger.info(f"[PAPER] Signal: {signal.upper()}, Volume: {volume}")
                logger.info(f"[PAPER] Price: ${close_price:.2f}")
                logger.info(f"[PAPER] Reason: {result.get('reason')}")
                logger.info(f"[PAPER] Confidence: {result.get('confidence')}")
                logger.info(f"[PAPER] Drift Detected: {result.get('drift_detected')}")
                logger.info("="*60)
                
                # Calculate SL/TP if enabled
                sl, tp = None, None
                if config.use_sl_tp:
                    # Try ATR-based SL/TP
                    if 'atr' in df.columns:
                        atr = df['atr'].iloc[-1]
                        if signal == 'buy':
                            sl = close_price - config.sl_atr_mult * atr
                            tp = close_price + config.tp_atr_mult * atr
                        else:  # sell
                            sl = close_price + config.sl_atr_mult * atr
                            tp = close_price - config.tp_atr_mult * atr
                    else:
                        # Fallback to percentage-based
                        if signal == 'buy':
                            sl = close_price * (1 - config.default_sl_pct)
                            tp = close_price * (1 + config.default_tp_pct)
                        else:  # sell
                            sl = close_price * (1 + config.default_sl_pct)
                            tp = close_price * (1 - config.default_tp_pct)
                
                # Execute paper order
                execution_result = engine.execute_order(
                    signal=signal,
                    volume=volume,
                    sl=sl,
                    tp=tp
                )
                
                if execution_result.get('success'):
                    logger.info(
                        f"[PAPER] ✓ Order executed: {signal.upper()} {volume} {config.symbol} @ ${close_price:.2f}"
                    )
                    
                    # Show account status
                    account = paper_client._request("GET", "/account")
                    positions = paper_client.get_positions()
                    logger.info(
                        f"[PAPER] Account - Balance: ${account['balance']:.2f}, "
                        f"Equity: ${account['equity']:.2f}, "
                        f"Open Positions: {len(positions)}"
                    )
                else:
                    logger.error(f"[PAPER] Order execution failed: {execution_result.get('error')}")
            else:
                # Log periodic status even without signals
                if engine.tick_count % 10 == 0:  # Every 10 ticks
                    account = paper_client._request("GET", "/account")
                    positions = paper_client.get_positions()
                    logger.info(
                        f"[PAPER] Status - Tick: {engine.tick_count}, "
                        f"Balance: ${account['balance']:.2f}, "
                        f"Equity: ${account['equity']:.2f}, "
                        f"Positions: {len(positions)}"
                    )
            
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            time.sleep(5)  # Wait before retrying
    
    # Final summary
    logger.info("="*60)
    logger.info("Paper Trading Session Summary")
    logger.info("="*60)
    account = paper_client._request("GET", "/account")
    positions = paper_client.get_positions()
    
    logger.info(f"Final Balance: ${account['balance']:.2f}")
    logger.info(f"Final Equity: ${account['equity']:.2f}")
    logger.info(f"P&L: ${account['balance'] - paper_client.initial_balance:.2f}")
    logger.info(f"Open Positions: {len(positions)}")
    logger.info(f"Total Ticks Processed: {engine.tick_count}")
    logger.info(f"Total Orders: {len(paper_client.order_history)}")
    logger.info("="*60)
    logger.info("Paper trading engine stopped")


def run_backtest_mode(
    config: TradingConfig,
    logger: TradingLogger,
    start_date: str,
    end_date: str
):
    """
    Run historical backtesting.
    
    Args:
        config: Trading configuration
        logger: Logger instance
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
    """
    logger.info(f"Starting BACKTEST mode: {start_date} to {end_date}")
    
    # Load historical data
    # This is a placeholder - actual implementation would load from database or files
    logger.error("Backtest mode not yet implemented")
    logger.info("To implement: Load historical data, run engine in simulation, report KPIs")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Trading Engine - Bitcoin Scalper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run live trading
  python engine_main.py --mode live --config config/engine_config.yaml
  
  # Run paper trading
  python engine_main.py --mode paper --config config/engine_config.yaml
  
  # Run backtest
  python engine_main.py --mode backtest --config config/engine_config.yaml --start 2024-01-01 --end 2024-12-31
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['live', 'paper', 'backtest'],
        help='Trading mode: live (real), paper (simulation), or backtest (historical)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Start date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='End date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            config = TradingConfig.from_yaml(str(config_path))
        elif config_path.suffix == '.json':
            config = TradingConfig.from_json(str(config_path))
        else:
            print(f"[ERROR] Unsupported config format: {config_path.suffix}")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        sys.exit(1)
    
    # Override log level if verbose
    if args.verbose:
        config.log_level = 'DEBUG'
    
    # Initialize logger
    logger = TradingLogger(
        log_dir=Path(config.log_dir) if config.log_dir else None,
        console_level=getattr(logging, config.log_level)
    )
    
    logger.info("="*60)
    logger.info("Trading Engine Starting")
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Symbol: {config.symbol}")
    logger.info(f"Timeframe: {config.timeframe}")
    logger.info(f"Model Type: {config.model_type}")
    logger.info("="*60)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run appropriate mode
    try:
        if args.mode == 'live':
            run_live_mode(config, logger)
        elif args.mode == 'paper':
            run_paper_mode(config, logger)
        elif args.mode == 'backtest':
            if not args.start or not args.end:
                print("[ERROR] Backtest mode requires --start and --end dates")
                sys.exit(1)
            run_backtest_mode(config, logger, args.start, args.end)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        logger.info("="*60)
        logger.info("Trading Engine Stopped")
        logger.info("="*60)


if __name__ == "__main__":
    main()
