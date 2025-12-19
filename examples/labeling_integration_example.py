"""
Integration Example: Using the Triple Barrier Method for Labeling

This example demonstrates how to use the new labeling module to:
1. Estimate dynamic volatility
2. Apply the Triple Barrier Method
3. Generate labels for supervised learning
4. Create meta-labels for filtering

This integrates with the existing data pipeline from Section 1.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import labeling functions
from src.bitcoin_scalper.labeling import (
    estimate_ewma_volatility,
    get_events,
    get_labels,
    get_meta_labels
)


def example_basic_triple_barrier():
    """
    Basic example: Apply Triple Barrier Method to price data.
    """
    print("=" * 80)
    print("Example 1: Basic Triple Barrier Method")
    print("=" * 80)
    
    # Create sample price data (simulated minute-level Bitcoin data)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    np.random.seed(42)
    
    # Simulate price with some volatility and drift
    returns = np.random.randn(1000) * 0.001 + 0.0001  # Small upward drift
    prices = pd.Series(50000 * np.exp(np.cumsum(returns)), index=dates)
    
    print(f"\nPrice data: {len(prices)} observations")
    print(f"Start price: ${prices.iloc[0]:.2f}")
    print(f"End price: ${prices.iloc[-1]:.2f}")
    print(f"Return: {(prices.iloc[-1]/prices.iloc[0] - 1)*100:.2f}%")
    
    # Step 1: Estimate dynamic volatility
    print("\n" + "-" * 80)
    print("Step 1: Estimate Volatility (EWMA)")
    print("-" * 80)
    
    volatility = estimate_ewma_volatility(prices, span=100)
    print(f"Volatility mean: {volatility.mean():.6f}")
    print(f"Volatility std: {volatility.std():.6f}")
    print(f"Current volatility: {volatility.iloc[-1]:.6f}")
    
    # Step 2: Define event timestamps (e.g., potential trading signals)
    print("\n" + "-" * 80)
    print("Step 2: Define Event Timestamps")
    print("-" * 80)
    
    # Select events every 50 bars
    event_indices = range(100, 900, 50)
    event_times = pd.DatetimeIndex([dates[i] for i in event_indices])
    print(f"Number of events: {len(event_times)}")
    
    # Step 3: Apply Triple Barrier Method
    print("\n" + "-" * 80)
    print("Step 3: Apply Triple Barrier Method")
    print("-" * 80)
    
    # Use 2-sigma barriers based on volatility
    pt_sl = 2.0 * volatility.loc[event_times]
    
    events = get_events(
        close=prices,
        timestamps=event_times,
        pt_sl=pt_sl,
        max_holding_period=pd.Timedelta('15min')  # 15-minute time limit
    )
    
    print(f"\nEvents processed: {len(events)}")
    print(f"\nBarrier touch distribution:")
    type_counts = events['type'].value_counts()
    print(f"  Profit targets hit: {type_counts.get(1, 0)} ({type_counts.get(1, 0)/len(events)*100:.1f}%)")
    print(f"  Stop losses hit: {type_counts.get(-1, 0)} ({type_counts.get(-1, 0)/len(events)*100:.1f}%)")
    print(f"  Time limits hit: {type_counts.get(0, 0)} ({type_counts.get(0, 0)/len(events)*100:.1f}%)")
    
    print(f"\nReturn statistics:")
    print(f"  Mean return: {events['return'].mean()*100:.3f}%")
    print(f"  Median return: {events['return'].median()*100:.3f}%")
    print(f"  Std return: {events['return'].std()*100:.3f}%")
    
    # Step 4: Generate labels
    print("\n" + "-" * 80)
    print("Step 4: Generate Labels")
    print("-" * 80)
    
    labels = get_labels(events, prices, label_type='fixed')
    print(f"\nLabel distribution:")
    label_counts = labels.value_counts()
    print(f"  Long (1): {label_counts.get(1, 0)}")
    print(f"  Short (-1): {label_counts.get(-1, 0)}")
    print(f"  Neutral (0): {label_counts.get(0, 0)}")
    
    return prices, events, labels


def example_meta_labeling():
    """
    Advanced example: Meta-labeling for filtering false positives.
    """
    print("\n\n" + "=" * 80)
    print("Example 2: Meta-Labeling")
    print("=" * 80)
    
    # Create sample price data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    np.random.seed(43)
    returns = np.random.randn(1000) * 0.001
    prices = pd.Series(50000 * np.exp(np.cumsum(returns)), index=dates)
    
    # Step 1: Simulate primary model predictions
    print("\nStep 1: Simulate Primary Model Predictions")
    print("-" * 80)
    
    event_indices = range(100, 900, 50)
    event_times = pd.DatetimeIndex([dates[i] for i in event_indices])
    
    # Simulate a primary model that predicts bullish (1) or bearish (-1)
    np.random.seed(44)
    primary_predictions = pd.Series(
        np.random.choice([1, -1], size=len(event_times), p=[0.6, 0.4]),
        index=event_times
    )
    
    print(f"Primary model predictions: {len(primary_predictions)}")
    print(f"  Bullish signals: {(primary_predictions == 1).sum()}")
    print(f"  Bearish signals: {(primary_predictions == -1).sum()}")
    
    # Step 2: Apply Triple Barrier with side from predictions
    print("\nStep 2: Apply Triple Barrier Method")
    print("-" * 80)
    
    volatility = estimate_ewma_volatility(prices, span=100)
    pt_sl = 2.0 * volatility.loc[event_times]
    
    events = get_events(
        close=prices,
        timestamps=event_times,
        pt_sl=pt_sl,
        max_holding_period=pd.Timedelta('15min'),
        side=primary_predictions  # Use predictions as trade direction
    )
    
    # Step 3: Generate meta-labels
    print("\nStep 3: Generate Meta-Labels")
    print("-" * 80)
    
    meta_labels = get_meta_labels(
        events,
        prices,
        primary_model_predictions=primary_predictions,
        side_from_predictions=True
    )
    
    print(f"\nMeta-label distribution:")
    print(f"  Successful trades (1): {(meta_labels == 1).sum()} ({(meta_labels == 1).mean()*100:.1f}%)")
    print(f"  Failed trades (0): {(meta_labels == 0).sum()} ({(meta_labels == 0).mean()*100:.1f}%)")
    
    print("\nMeta-labeling interpretation:")
    print("  - Meta-label = 1: Primary model signal would be profitable → TAKE THE TRADE")
    print("  - Meta-label = 0: Primary model signal would be unprofitable → FILTER OUT")
    
    return events, meta_labels


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print(" LABELING MODULE - INTEGRATION EXAMPLES")
    print("=" * 80)
    print("\nDemonstrating Section 2: LABELS & TARGETS Implementation")
    print("\n" + "=" * 80 + "\n")
    
    # Run examples
    prices, events, labels = example_basic_triple_barrier()
    events_meta, meta_labels = example_meta_labeling()
    
    print("\n\n" + "=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    print("\nThe labeling module provides:")
    print("  1. Dynamic volatility estimation (EWMA)")
    print("  2. Triple Barrier Method for realistic labeling")
    print("  3. Primary labels for classification")
    print("  4. Meta-labels for filtering false positives")
    print("\nKey advantages over simple return-based labeling:")
    print("  ✓ Incorporates risk management (TP/SL)")
    print("  ✓ Adapts to changing volatility")
    print("  ✓ Realistic holding periods (time limit)")
    print("  ✓ Meta-labeling improves Sharpe ratio")
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
