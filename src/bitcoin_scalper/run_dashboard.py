#!/usr/bin/env python3
"""
Bitcoin Scalper Trading Dashboard Launcher. 

A professional PyQt6-based GUI for monitoring and controlling
the trading bot with real-time visualization. 

Usage:
    python src/bitcoin_scalper/run_dashboard. py [--config CONFIG_PATH] [--model MODEL_PATH] [--demo]
"""

import sys
import argparse
import subprocess
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bitcoin_scalper.core. config import TradingConfig
from bitcoin_scalper.dashboard. main_window import MainWindow


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bitcoin Scalper Trading Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Launch with default config
    python src/bitcoin_scalper/run_dashboard. py
    
    # Launch in demo mode (paper trading with engine)
    python src/bitcoin_scalper/run_dashboard.py --demo
    
    # Launch with custom config and model
    python src/bitcoin_scalper/run_dashboard.py \\
        --config config/engine_config.yaml \\
        --model models/meta_model_production.pkl
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/engine_config.yaml',
        help='Path to engine configuration YAML file (default: config/engine_config. yaml)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='/bitcoin_scalper/models/meta_model_production.pkl',
        help='Path to trained model file (default: None - will use demo mode)'
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run in demo mode with paper trading engine'
    )
    
    return parser.parse_args()


def launch_engine(config_path:  str, demo: bool = False):
    """
    Launch the trading engine as a subprocess.
    
    Args:
        config_path: Path to the engine configuration file
        demo: If True, launch in paper trading mode
        
    Returns:
        subprocess.Popen: The engine process
    """
    engine_script = Path(__file__).parent / "engine_main.py"
    
    if demo:
        # Launch engine in paper trading mode
        cmd = [
            sys.executable,
            str(engine_script),
            "--mode", "paper",
            "--config", config_path
        ]
        print(f"🚀 Launching engine in PAPER mode: {' '.join(cmd)}")
    else:
        # Launch engine in live mode (if user wants real trading)
        cmd = [
            sys.executable,
            str(engine_script),
            "--mode", "live",
            "--config", config_path
        ]
        print(f"🚀 Launching engine in LIVE mode: {' '.join(cmd)}")
    
    # Start the engine process
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess. PIPE,
            stderr=subprocess. PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        print(f"✓ Engine process started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Failed to start engine: {e}")
        return None


def main():
    """Main entry point."""
    print("="*70)
    print("Bitcoin Scalper Trading Dashboard")
    print("="*70)
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path. exists():
        print(f"✓ Loading config from: {config_path}")
        config = TradingConfig.from_yaml(str(config_path))
    else:
        print(f"⚠️  Config file not found: {config_path}")
        print("Using default configuration")
        config = TradingConfig()
    
    # Check model path
    model_path = args.model
    if model_path:
        model_path = Path(model_path)
        if model_path.exists():
            print(f"✓ Model file found: {model_path}")
            model_path = str(model_path)
        else:
            print(f"⚠️  Model file not found:  {model_path}")
            print("Continuing without model (demo mode)")
            model_path = None
    else:
        print("ℹ️  No model specified - running in demo mode")
    
    # Display configuration
    print("\nConfiguration:")
    print(f"  Symbol: {config.symbol}")
    print(f"  Timeframe: {config.timeframe}")
    print(f"  Mode: {config. mode. upper()}")
    print(f"  Meta Threshold: {config.meta_threshold:. 2f}")
    
    # Launch engine if demo mode is enabled
    engine_process = None
    if args.demo:
        print("\n" + "="*70)
        print("DEMO MODE:  Launching Paper Trading Engine")
        print("="*70)
        engine_process = launch_engine(str(config_path), demo=True)
        
        if not engine_process:
            print("❌ Failed to start engine.  Dashboard will run without engine.")
    
    print("\n" + "="*70)
    print("Launching dashboard...")
    print("="*70 + "\n")
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Bitcoin Scalper")
    app.setOrganizationName("Trading Systems")
    
    # Create and show main window
    window = MainWindow(config, model_path)
    window.show()
    
    # Log startup message
    window.log_console.append_log("="*60)
    window.log_console.append_log("Bitcoin Scalper Trading Dashboard")
    window.log_console.append_log("="*60)
    window.log_console.append_log(f"Configuration loaded: {config.symbol} {config.timeframe}")
    window.log_console.append_log(f"Meta threshold: {config.meta_threshold:.2f}")
    
    if args.demo and engine_process:
        window.log_console.append_log("="*60)
        window.log_console.append_log("🎯 DEMO MODE ACTIVE")
        window.log_console. append_log(f"Engine running in PAPER trading mode (PID: {engine_process.pid})")
        window.log_console.append_log("All trades are simulated - no real money at risk")
        window.log_console.append_log("="*60)
    elif model_path: 
        window.log_console. append_log(f"Model:  {Path(model_path).name}")
    else:
        window.log_console.append_log("Running in DEMO mode (no model loaded)")
    
    window.log_console.append_log("Dashboard ready - monitoring engine...")
    window.log_console.append_log("="*60)
    
    # Run application
    exit_code = app.exec()
    
    # Cleanup:  terminate engine process if it was started
    if engine_process: 
        print("\n" + "="*70)
        print("Shutting down engine...")
        print("="*70)
        engine_process.terminate()
        try:
            engine_process. wait(timeout=5)
            print("✓ Engine stopped gracefully")
        except subprocess.TimeoutExpired:
            print("⚠️  Engine did not stop gracefully, forcing...")
            engine_process.kill()
            engine_process.wait()
            print("✓ Engine terminated")
    
    sys.exit(exit_code)


if __name__ == "__main__": 
    try:
        main()
    except KeyboardInterrupt: 
        print("\n\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback. print_exc()
        sys.exit(1)
