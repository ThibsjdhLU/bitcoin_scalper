"""
Risk management module for position sizing and capital allocation.

This module provides mathematical methods for determining optimal position sizes:
- Kelly Criterion (Fractional) for risk-adjusted sizing
- Target Volatility Sizing for consistent portfolio volatility

These methods answer "How much to buy?" based on model confidence and market conditions.

References:
    Kelly, J. L. (1956). A New Interpretation of Information Rate.
    López de Prado, M. (2018). Advances in Financial Machine Learning.
"""

from .sizing import KellySizer, TargetVolatilitySizer, PositionSizer

__all__ = [
    'KellySizer',
    'TargetVolatilitySizer',
    'PositionSizer',
]
