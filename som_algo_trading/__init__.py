"""
SOM Algo Trading
================
A Self-Organizing Map (SOM) based algorithmic trading pipeline that clusters
historical market regimes and generates T+1 forecasts based on structural
similarity to those historical patterns.

Modules
-------
- config      : Hyperparameter dataclass and defaults
- data         : Data engineering pipeline (Phase 1)
- som          : SOM core — training and clustering (Phase 2)
- inference    : Live inference and risk management (Phase 3)
- forecaster   : Statistical forecasting engine (Phase 4)
- pipeline     : End-to-end orchestrator tying all phases together
- utils        : Shared utilities (logging, metrics, plotting)

Quick Start
-----------
>>> from som_algo_trading import SOMTradingPipeline, TradingConfig
>>> config = TradingConfig(window_size=30, grid_rows=10, grid_cols=10)
>>> pipeline = SOMTradingPipeline(config)
>>> pipeline.fit(prices_array)
>>> signal = pipeline.predict(live_prices_array)
>>> print(signal)
"""

from .config import TradingConfig
from .pipeline import SOMTradingPipeline
from .data import DataEngineer
from .som import SOMCore
from .inference import LiveInference
from .forecaster import ForecastingEngine
from . import utils

__version__ = "1.0.0"
__author__ = "SOM Algo Trading"
__all__ = [
    "TradingConfig",
    "SOMTradingPipeline",
    "DataEngineer",
    "SOMCore",
    "LiveInference",
    "ForecastingEngine",
    "utils",
]
