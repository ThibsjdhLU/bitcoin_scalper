"""
Example Integration: Validation & Risk Management Modules

This example demonstrates how to use the new validation and risk management
modules together with the existing ML/RL infrastructure.

It shows:
1. Using Purged Cross-Validation for model evaluation
2. Implementing position sizing with Kelly Criterion
3. Running a realistic backtest with all components
4. Monitoring for drift in production
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import validation and risk modules
from bitcoin_scalper.validation import PurgedKFold, CombinatorialPurgedCV, DriftScanner, Backtester
from bitcoin_scalper.risk import KellySizer, TargetVolatilitySizer

# Simulate some market data
def generate_sample_data(n_samples=1000):
    """Generate sample price and feature data."""
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='5min')
    
    # Random walk price
    returns = np.random.randn(n_samples) * 0.002
    prices = pd.Series(50000 * (1 + returns).cumprod(), index=dates)
    
    # Generate features (technical indicators)
    features = pd.DataFrame({
        'returns': prices.pct_change(),
        'volatility': prices.pct_change().rolling(20).std(),
        'rsi': np.random.uniform(30, 70, n_samples),
        'volume': np.random.uniform(100, 1000, n_samples),
    }, index=dates)
    
    # Generate labels (1=buy, -1=sell, 0=hold)
    labels = pd.Series(np.random.choice([-1, 0, 1], n_samples, p=[0.3, 0.4, 0.3]), index=dates)
    
    # Generate t1 (label end times) - simulate Triple Barrier
    t1 = pd.Series(dates + timedelta(minutes=15), index=range(n_samples))
    
    return prices, features.fillna(0), labels, t1


def example_1_purged_cross_validation():
    """
    Example 1: Using Purged K-Fold for model validation.
    
    Demonstrates how to use PurgedKFold with scikit-learn models
    to prevent look-ahead bias in time series validation.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Purged Cross-Validation")
    print("="*60)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    # Generate data
    prices, features, labels, t1 = generate_sample_data(n_samples=500)
    
    X = features.values
    y = labels.values
    
    # Create purged K-fold cross-validator
    cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
    
    # Train model with purged CV
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv.split(X, y, t1=t1), scoring='accuracy')
    
    print(f"\nCross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print("\nNote: Purging removes training samples with look-ahead bias")


def example_2_position_sizing():
    """
    Example 2: Position sizing with Kelly Criterion.
    
    Shows how to calculate optimal position sizes based on
    model confidence and expected returns.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Kelly Criterion Position Sizing")
    print("="*60)
    
    # Initialize Half-Kelly sizer (conservative)
    sizer = KellySizer(kelly_fraction=0.5, max_leverage=1.0)
    
    # Trading scenario
    capital = 10000  # $10k portfolio
    btc_price = 50000  # BTC at $50k
    
    # Model predictions
    win_probability = 0.65  # 65% win rate
    payoff_ratio = 2.0      # 2:1 reward/risk
    
    # Calculate position size
    position_size = sizer.calculate_size(
        capital=capital,
        price=btc_price,
        win_prob=win_probability,
        payoff_ratio=payoff_ratio
    )
    
    position_value = position_size * btc_price
    position_pct = (position_value / capital) * 100
    
    print(f"\nPortfolio: ${capital:,.0f}")
    print(f"BTC Price: ${btc_price:,.0f}")
    print(f"Model Edge: {win_probability:.1%} win rate, {payoff_ratio:.1f}:1 payoff")
    print(f"\nRecommended Position:")
    print(f"  Size: {position_size:.6f} BTC")
    print(f"  Value: ${position_value:,.2f}")
    print(f"  Allocation: {position_pct:.1f}% of capital")
    
    # Target volatility sizing
    print("\n" + "-"*60)
    print("Alternative: Target Volatility Sizing")
    print("-"*60)
    
    vol_sizer = TargetVolatilitySizer(target_volatility=0.40, max_leverage=1.0)
    
    asset_volatility = 0.80  # BTC at 80% annual vol
    
    vol_position = vol_sizer.calculate_size(
        capital=capital,
        price=btc_price,
        asset_volatility=asset_volatility
    )
    
    vol_value = vol_position * btc_price
    vol_pct = (vol_value / capital) * 100
    
    print(f"\nTarget Portfolio Vol: 40%")
    print(f"BTC Volatility: {asset_volatility:.0%}")
    print(f"\nRecommended Position:")
    print(f"  Size: {vol_position:.6f} BTC")
    print(f"  Value: ${vol_value:,.2f}")
    print(f"  Allocation: {vol_pct:.1f}% of capital")


def example_3_backtesting():
    """
    Example 3: Event-driven backtesting with realistic costs.
    
    Demonstrates full backtest with:
    - ML signals
    - Position sizing
    - Slippage and commissions
    - Performance metrics
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Realistic Backtesting")
    print("="*60)
    
    # Generate market data
    prices, features, labels, t1 = generate_sample_data(n_samples=200)
    
    # Use labels as signals (in practice, these would be from your ML model)
    signals = labels
    
    # Initialize backtester with Kelly sizer
    kelly_sizer = KellySizer(kelly_fraction=0.25)  # Conservative quarter-Kelly
    
    backtester = Backtester(
        initial_capital=10000,
        commission_pct=0.001,   # 0.1% commission
        slippage_pct=0.0005,    # 0.05% slippage
        position_sizer=kelly_sizer
    )
    
    # Run backtest
    print("\nRunning backtest...")
    results = backtester.run(
        prices=prices,
        signals=signals,
        signal_params={'win_prob': 0.55, 'payoff_ratio': 1.5}
    )
    
    # Display results
    print(results.summary())


def example_4_drift_monitoring():
    """
    Example 4: Online drift detection for production monitoring.
    
    Shows how to monitor model performance in real-time and
    detect when the model becomes invalid.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Drift Detection")
    print("="*60)
    
    # Initialize drift scanner
    scanner = DriftScanner(delta=0.002, use_river=False)
    
    print("\nSimulating online trading with drift detection...")
    print("Phase 1: Stable model performance")
    
    # Phase 1: Stable performance
    for i in range(50):
        error = np.random.normal(0.05, 0.01)  # Low, stable errors
        timestamp = pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i)
        
        if scanner.scan(error, timestamp):
            print(f"  ⚠️ Drift detected at {timestamp}")
    
    print(f"  Model stable, no drift detected")
    
    # Phase 2: Regime change (drift)
    print("\nPhase 2: Market regime change")
    
    for i in range(50, 100):
        error = np.random.normal(0.30, 0.05)  # Much higher errors
        timestamp = pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i)
        
        if scanner.scan(error, timestamp):
            print(f"  ⚠️ DRIFT DETECTED at {timestamp}")
            print("  Action: Stop trading and retrain model")
            break
    
    # Show drift history
    history = scanner.get_drift_history()
    if len(history) > 0:
        print(f"\nTotal drift events: {len(history)}")
        print("Latest events:")
        print(history.tail())
    
    stats = scanner.get_statistics()
    print(f"\nScanner statistics: {stats}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("VALIDATION & RISK MANAGEMENT MODULE EXAMPLES")
    print("="*70)
    
    # Run examples
    example_1_purged_cross_validation()
    example_2_position_sizing()
    example_3_backtesting()
    example_4_drift_monitoring()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
