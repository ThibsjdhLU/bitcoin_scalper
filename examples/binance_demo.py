#!/usr/bin/env python3
"""
Demonstration script for Binance connector integration.

This script shows how to:
1. Initialize the Binance connector
2. Fetch market data
3. Verify data format
4. Use with the trading engine

Usage:
    # Set environment variables first
    export BINANCE_API_KEY="your_key"
    export BINANCE_API_SECRET="your_secret"
    export BINANCE_TESTNET="true"
    
    # Run the demo
    python examples/binance_demo.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
from bitcoin_scalper.connectors.binance_connector import BinanceConnector
from bitcoin_scalper.core.engine import TradingEngine, TradingMode


def demo_connector_basic():
    """Demonstrate basic Binance connector usage."""
    print("=" * 60)
    print("DEMO 1: Basic Binance Connector Usage")
    print("=" * 60)
    
    # Initialize connector (using testnet by default)
    api_key = os.getenv("BINANCE_API_KEY", "test_key")
    api_secret = os.getenv("BINANCE_API_SECRET", "test_secret")
    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    
    print(f"\nInitializing Binance connector (testnet={testnet})...")
    
    try:
        connector = BinanceConnector(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )
        print("✓ Connector initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize connector: {e}")
        print("\nNote: This is expected if you haven't set API credentials.")
        print("Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")
        return
    
    # Fetch OHLCV data
    print("\nFetching OHLCV data for BTC/USDT...")
    try:
        df = connector.fetch_ohlcv("BTC/USDT", timeframe="1m", limit=10)
        print(f"✓ Fetched {len(df)} candles")
        
        # Display data format
        print("\nData Format:")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Index: {df.index.name} (type: {type(df.index).__name__})")
        print(f"  Shape: {df.shape}")
        
        print("\nSample data:")
        print(df.head())
        
        print("\nData types:")
        print(df.dtypes)
        
    except Exception as e:
        print(f"✗ Failed to fetch data: {e}")
        return
    
    # Get balance
    print("\nFetching USDT balance...")
    try:
        balance = connector.get_balance("USDT")
        print(f"✓ Free USDT balance: {balance}")
    except Exception as e:
        print(f"✗ Failed to fetch balance: {e}")
    
    print("\n" + "=" * 60)


def demo_engine_integration():
    """Demonstrate using Binance connector with trading engine."""
    print("=" * 60)
    print("DEMO 2: Trading Engine Integration")
    print("=" * 60)
    
    # Create sample data in Binance format
    print("\nCreating sample Binance-format data...")
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    df = pd.DataFrame({
        'open': [50000.0 + i * 10 for i in range(100)],
        'high': [50100.0 + i * 10 for i in range(100)],
        'low': [49900.0 + i * 10 for i in range(100)],
        'close': [50050.0 + i * 10 for i in range(100)],
        'volume': [100.0 + i for i in range(100)],
    }, index=dates)
    df.index.name = 'date'
    
    print(f"✓ Created sample data with {len(df)} candles")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Index: {df.index.name}")
    
    # Create mock connector for demo
    from unittest.mock import Mock
    mock_connector = Mock()
    mock_connector._request = Mock(return_value={'balance': 10000.0, 'equity': 10000.0})
    
    # Initialize engine
    print("\nInitializing trading engine...")
    try:
        engine = TradingEngine(
            connector=mock_connector,
            mode=TradingMode.ML,
            symbol="BTC/USDT",
            timeframe="1m",
            log_dir=None,
            drift_detection=False,
        )
        print("✓ Engine initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize engine: {e}")
        return
    
    # Process tick
    print("\nProcessing market data tick...")
    try:
        result = engine.process_tick(df)
        print(f"✓ Tick processed successfully")
        print(f"  Signal: {result.get('signal')}")
        print(f"  Confidence: {result.get('confidence')}")
        print(f"  Reason: {result.get('reason')}")
        
        if result.get('error'):
            print(f"  Error: {result.get('error')}")
        
    except Exception as e:
        print(f"✗ Failed to process tick: {e}")
        return
    
    print("\n✅ Engine successfully processes Binance data format!")
    print("=" * 60)


def demo_column_compatibility():
    """Demonstrate column name compatibility."""
    print("=" * 60)
    print("DEMO 3: Column Name Compatibility")
    print("=" * 60)
    
    print("\nBinance format uses lowercase column names:")
    print("  ['date', 'open', 'high', 'low', 'close', 'volume']")
    
    print("\nFeature engineering defaults to lowercase:")
    print("  add_indicators(df, price_col='close', high_col='high', ...)")
    
    print("\nThis means:")
    print("  ✓ No column renaming needed")
    print("  ✓ Direct compatibility with ML pipeline")
    print("  ✓ Consistent data format across the system")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=50, freq='1min')
    df = pd.DataFrame({
        'open': [50000.0 + i * 10 for i in range(50)],
        'high': [50100.0 + i * 10 for i in range(50)],
        'low': [49900.0 + i * 10 for i in range(50)],
        'close': [50050.0 + i * 10 for i in range(50)],
        'volume': [100.0 + i for i in range(50)],
    }, index=dates)
    df.index.name = 'date'
    
    print("\nApplying feature engineering...")
    from bitcoin_scalper.core.feature_engineering import FeatureEngineering
    
    fe = FeatureEngineering()
    df_features = fe.add_indicators(df, price_col='close', high_col='high', 
                                     low_col='low', volume_col='volume')
    
    print(f"✓ Features added successfully")
    print(f"  Original columns: {list(df.columns)}")
    print(f"  Total columns after features: {len(df_features.columns)}")
    print(f"  Sample features: {list(df_features.columns[:10])}")
    
    print("\n" + "=" * 60)


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("BINANCE CONNECTOR DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Run demos
    demo_connector_basic()
    print()
    demo_engine_integration()
    print()
    demo_column_compatibility()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("✅ Binance connector successfully created")
    print("✅ Engine refactored to be exchange-agnostic")
    print("✅ Data format standardized with lowercase columns")
    print("✅ Feature engineering compatible out of the box")
    print()
    print("Next steps:")
    print("1. Set your Binance API credentials in environment variables")
    print("2. Update config/engine_config.yaml with 'exchange: binance'")
    print("3. Run: python src/bitcoin_scalper/engine_main.py --mode paper --config config/engine_config.yaml")
    print()


if __name__ == "__main__":
    main()
