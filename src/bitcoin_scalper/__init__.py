"""
Bitcoin Scalper - Bot de trading algorithmique pour Bitcoin
"""

__version__ = "0.1.0"

from .core.data_fetcher import DataFetcher
from .core.order_executor import OrderExecutor
from .core.risk_management import RiskManager
from .strategies.strategy import BaseStrategy
from .strategies.regime_switcher import RegimeSwitcher
from .strategies.fractal_analysis import FractalAnalysis
from .strategies.ensemble import EnsembleStrategy
from .analysis.market_generator import MarketGenerator

__all__ = [
    'DataFetcher',
    'OrderExecutor',
    'RiskManager',
    'BaseStrategy',
    'RegimeSwitcher',
    'FractalAnalysis',
    'EnsembleStrategy',
    'MarketGenerator'
]
