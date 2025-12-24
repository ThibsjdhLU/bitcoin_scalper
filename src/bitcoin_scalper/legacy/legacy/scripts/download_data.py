#!/usr/bin/env python3
"""
Historical Data Download Script for Binance.

This script downloads historical OHLCV data from Binance using the public API
(no authentication required) and saves it to CSV for training purposes.

Features:
- Fetches data for any symbol and timeframe
- Automatic pagination for large date ranges
- Saves to standardized CSV format
- Displays download statistics

Usage:
    # Download 1 year of BTC/USDT 1-minute data (default)
    python src/bitcoin_scalper/scripts/download_data.py
    
    # Download 30 days of ETH/USDT 5-minute data
    python src/bitcoin_scalper/scripts/download_data.py --symbol ETH/USDT --timeframe 5m --days 30
    
    # Download specific date range
    python src/bitcoin_scalper/scripts/download_data.py --start-date 2024-01-01 --end-date 2024-12-31
    
    # Download to custom location
    python src/bitcoin_scalper/scripts/download_data.py --output custom_data.csv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bitcoin_scalper.connectors.binance_public import BinancePublicClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download historical market data from Binance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 1 year of BTC/USDT 1-minute data
  python %(prog)s
  
  # Download 30 days of ETH/USDT 5-minute data
  python %(prog)s --symbol ETH/USDT --timeframe 5m --days 30
  
  # Download specific date range
  python %(prog)s --start-date 2024-01-01 --end-date 2024-12-31
  
  # Download to custom location
  python %(prog)s --output data/custom/my_data.csv
        """
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Trading pair symbol (default: BTC/USDT)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1m',
        help='Candle timeframe: 1m, 5m, 15m, 1h, 4h, 1d, etc. (default: 1m)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days to fetch (default: 365)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: data/raw/BINANCE_{symbol}_{timeframe}.csv)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def format_symbol_for_filename(symbol: str) -> str:
    """
    Format symbol for use in filename.
    
    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        
    Returns:
        Formatted symbol (e.g., "BTCUSDT")
    """
    return symbol.replace('/', '')


def get_output_path(symbol: str, timeframe: str, output: str = None) -> Path:
    """
    Get output file path.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        output: Custom output path (optional)
        
    Returns:
        Path object for output file
    """
    if output:
        return Path(output)
    
    # Default path: data/raw/BINANCE_{symbol}_{timeframe}.csv
    repo_root = Path(__file__).parent.parent.parent.parent
    data_dir = repo_root / 'data' / 'raw'
    
    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    symbol_str = format_symbol_for_filename(symbol)
    filename = f"BINANCE_{symbol_str}_{timeframe}.csv"
    
    return data_dir / filename


def print_stats(df, symbol: str, timeframe: str, output_path: Path):
    """
    Print download statistics.
    
    Args:
        df: Downloaded DataFrame
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        output_path: Path where data was saved
    """
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Symbol:       {symbol}")
    print(f"Timeframe:    {timeframe}")
    print(f"Start Date:   {df.index[0]}")
    print(f"End Date:     {df.index[-1]}")
    print(f"Total Rows:   {len(df):,}")
    print(f"Saved to:     {output_path}")
    print(f"File Size:    {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 60)
    
    # Show sample data
    print("\nSample Data (first 5 rows):")
    print(df.head())
    print("\nSample Data (last 5 rows):")
    print(df.tail())
    print()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print header
    print("\n" + "=" * 60)
    print("BINANCE HISTORICAL DATA DOWNLOADER")
    print("=" * 60)
    print(f"Symbol:       {args.symbol}")
    print(f"Timeframe:    {args.timeframe}")
    
    # Determine date range
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
        print(f"Date Range:   {start_date} to {end_date}")
    else:
        end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Days:         {args.days} (from {start_date} to {end_date})")
    
    output_path = get_output_path(args.symbol, args.timeframe, args.output)
    print(f"Output:       {output_path}")
    print("=" * 60)
    print()
    
    try:
        # Initialize client
        logger.info("Initializing Binance public API client...")
        client = BinancePublicClient()
        
        # Fetch data
        logger.info("Fetching historical data...")
        df = client.fetch_history(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            logger.error("No data fetched! Check symbol and date range.")
            sys.exit(1)
        
        # Save to CSV
        logger.info(f"Saving to {output_path}...")
        df.to_csv(output_path)
        
        # Print statistics
        print_stats(df, args.symbol, args.timeframe, output_path)
        
        logger.info("✅ Download completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
