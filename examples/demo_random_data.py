"""
examples/demo_random_data.py
-----------------------------
End-to-end demonstration of the SOM Algo Trading pipeline using
synthetically generated price data.  No real market data required.

Run from the project root:
    python examples/demo_random_data.py

What this script does
---------------------
1.  Generates a multi-regime synthetic price series (bull / bear / sideways).
2.  Auto-selects a SOM grid size using the built-in heuristic.
3.  Trains the full pipeline on 80 % of the data.
4.  Runs a walk-forward backtest on the held-out 20 %.
5.  Prints a detailed backtest summary.
6.  Simulates a "live" prediction on the most recent prices.
7.  Saves and reloads the pipeline to demonstrate persistence.
8.  (Optional) Plots the U-matrix and equity curve if matplotlib is available.
"""

import sys
import os

# Allow running the script without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import logging

from som_algo_trading import SOMTradingPipeline, TradingConfig
from som_algo_trading.utils import (
    setup_logging,
    generate_regime_prices,
    backtest_summary,
    plot_u_matrix,
    plot_cluster_signals,
    plot_equity_curve,
)

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  1. Generate synthetic data
# ─────────────────────────────────────────────
N_DAYS = 1500
logger.info("Generating %d days of synthetic multi-regime prices ...", N_DAYS)
prices = generate_regime_prices(n_days=N_DAYS, random_state=42)
logger.info("Price range: %.2f – %.2f", prices.min(), prices.max())

# ─────────────────────────────────────────────
#  2. Auto-select SOM grid
# ─────────────────────────────────────────────
WINDOW_SIZE = 30
TRAIN_RATIO = 0.80

# Estimate number of training windows
n_windows_approx = int((N_DAYS - WINDOW_SIZE - 1) * TRAIN_RATIO)
rows, cols = TradingConfig.recommend_grid(n_windows_approx)
logger.info(
    "Recommended SOM grid: %dx%d (%d nodes) for ~%d windows",
    rows, cols, rows * cols, n_windows_approx,
)

# ─────────────────────────────────────────────
#  3. Configure and train
# ─────────────────────────────────────────────
config = TradingConfig(
    window_size=WINDOW_SIZE,
    grid_rows=rows,
    grid_cols=cols,
    sigma=3.0,
    learning_rate=0.5,
    num_iterations=5_000,          # increase for production use
    anomaly_threshold=None,        # auto-calculate from 95th-pct of training QEs
    anomaly_percentile=95.0,
    min_cluster_samples=20,        # lowered for this demo dataset size
    win_rate_threshold=0.58,
    expected_return_threshold=0.0005,
    train_ratio=TRAIN_RATIO,
    random_state=42,
)
print("\n", config, "\n")

pipeline = SOMTradingPipeline(config)
pipeline.fit(prices)

# ─────────────────────────────────────────────
#  4. Walk-forward backtest
# ─────────────────────────────────────────────
logger.info("Running backtest on held-out test set ...")
bt = pipeline.backtest()
print(backtest_summary(bt))

# ─────────────────────────────────────────────
#  5. Live prediction
# ─────────────────────────────────────────────
# Simulate "today's" close: use the last window_size + 5 prices
live_window = prices[-(WINDOW_SIZE + 5):]
result = pipeline.predict(live_window)

print("\n=== LIVE SIGNAL ===")
print(result.summary())

# ─────────────────────────────────────────────
#  6. Save and reload
# ─────────────────────────────────────────────
SAVE_DIR = "/tmp/som_trading_demo"
pipeline.save(SAVE_DIR)
logger.info("Pipeline saved to %s", SAVE_DIR)

loaded = SOMTradingPipeline.load(SAVE_DIR)
result2 = loaded.predict(live_window)
logger.info("Reloaded pipeline signal: %s (should match above)", result2.signal)
assert result2.signal == result.signal, "Mismatch after save/load!"
print("\nSave/load round-trip: OK ✓")

# ─────────────────────────────────────────────
#  7. Optional visualisations
# ─────────────────────────────────────────────
try:
    import matplotlib  # noqa: F401
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

if HAS_MPL:
    print("\nGenerating visualisations ...")
    plot_u_matrix(pipeline.som_core, title="SOM U-Matrix — Synthetic Regime Data")
    plot_cluster_signals(
        pipeline.forecaster,
        grid_rows=config.grid_rows,
        grid_cols=config.grid_cols,
        title="Cluster Signal Map",
    )
    plot_equity_curve(bt, title="Strategy vs Buy-and-Hold (Test Set)")
else:
    print("\nInstall matplotlib for visualisations:  pip install matplotlib")

print("\nDemo complete.")
