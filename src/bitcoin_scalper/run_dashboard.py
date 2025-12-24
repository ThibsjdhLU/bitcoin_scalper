#!/usr/bin/env python3
"""
Bitcoin Scalper Trading Dashboard Launcher.

A professional PyQt6-based GUI for monitoring and controlling
the trading bot with real-time visualization.

Usage:
    python src/bitcoin_scalper/run_dashboard.py [--config CONFIG_PATH] [--model MODEL_PATH]
"""

import sys
import argparse
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bitcoin_scalper.core.config import TradingConfig
from bitcoin_scalper.dashboard.main_window import MainWindow


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bitcoin Scalper Trading Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Launch with default config
    python src/bitcoin_scalper/run_dashboard.py
    
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
        help='Path to engine configuration YAML file (default: config/engine_config.yaml)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to trained model file (default: None - will use demo mode)'
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run in demo mode with simulated data'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    print("="*70)
    print("Bitcoin Scalper Trading Dashboard")
    print("="*70)
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
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
            print(f"⚠️  Model file not found: {model_path}")
            print("Continuing without model (demo mode)")
            model_path = None
    else:
        print("ℹ️  No model specified - running in demo mode")
    
    # Display configuration
    print("\nConfiguration:")
    print(f"  Symbol: {config.symbol}")
    print(f"  Timeframe: {config.timeframe}")
    print(f"  Mode: {config.mode.upper()}")
    print(f"  Meta Threshold: {config.meta_threshold:.2f}")
    
    print("\n" + "="*70)
    print("Launching dashboard...")
    print("="*70 + "\n")
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Bitcoin Scalper")
    app.setOrganizationName("Trading Systems")
    app.setAttribute(Qt.ApplicationAttribute. AA_EnableHighDpiScaling)
    # Enable high DPI support for macOS
    #app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    
    # Create and show main window
    window = MainWindow(config, model_path)
    window.show()
    
    # Log startup message
    window.log_console.append_log("="*60)
    window.log_console.append_log("Bitcoin Scalper Trading Dashboard")
    window.log_console.append_log("="*60)
    window.log_console.append_log(f"Configuration loaded: {config.symbol} {config.timeframe}")
    window.log_console.append_log(f"Meta threshold: {config.meta_threshold:.2f}")
    if model_path:
        window.log_console.append_log(f"Model: {Path(model_path).name}")
    else:
        window.log_console.append_log("Running in DEMO mode (no model loaded)")
    window.log_console.append_log("Click 'Start' to begin trading")
    window.log_console.append_log("="*60)
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
