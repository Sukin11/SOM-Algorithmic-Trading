# SOM Algo Trading

A production-grade, modular **Self-Organizing Map (SOM)** algorithmic trading pipeline that discovers historical market regimes and generates deterministic **BUY / SELL / HOLD** signals for the next trading day (T+1).

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Using Your Own Data](#using-your-own-data)
6. [Configuration Reference](#configuration-reference)
7. [Pipeline Phases](#pipeline-phases)
8. [Backtest & Evaluation](#backtest--evaluation)
9. [Save & Load a Trained Pipeline](#save--load-a-trained-pipeline)
10. [Visualisations](#visualisations)
11. [Running the Tests](#running-the-tests)
12. [Extending the Pipeline](#extending-the-pipeline)
13. [Disclaimer](#disclaimer)

---

## How It Works

```
Raw Prices
    │
    ▼ Phase 1 — Data Engineering
    │   • Daily % returns
    │   • Sliding windows (length = WINDOW_SIZE)
    │   • Z-score standardisation (scaler fitted on train only)
    │
    ▼ Phase 2 — SOM Training & Clustering
    │   • PCA-initialised MiniSom grid
    │   • Each training window → Best Matching Unit (BMU)
    │   • cluster_map: {(row, col): [window indices]}
    │
    ▼ Phase 3 — Live Inference & Risk Gate
    │   • Scale today's window with the saved scaler
    │   • Find its BMU
    │   • Compute Quantization Error (QE)
    │   • HALT if QE > anomaly_threshold (unseen regime)
    │
    ▼ Phase 4 — Statistical Forecasting
        • Look up historical forward returns for the matched cluster
        • Compute win rate & expected return
        • Emit BUY / SELL / HOLD
```

---

## Project Structure

```
som_algo_trading/
│
├── som_algo_trading/          # Main package
│   ├── __init__.py            # Public API surface
│   ├── config.py              # TradingConfig dataclass (all hyperparameters)
│   ├── data.py                # Phase 1 — DataEngineer
│   ├── som.py                 # Phase 2 — SOMCore
│   ├── inference.py           # Phase 3 — LiveInference + InferenceResult
│   ├── forecaster.py          # Phase 4 — ForecastingEngine + ForecastResult
│   ├── pipeline.py            # End-to-end SOMTradingPipeline orchestrator
│   └── utils.py               # Logging, synthetic data, metrics, plots
│
├── tests/
│   ├── test_data.py           # DataEngineer unit tests
│   ├── test_som.py            # SOMCore unit tests
│   ├── test_forecaster.py     # ForecastingEngine unit tests
│   └── test_pipeline.py       # Full pipeline integration tests
│
├── examples/
│   └── demo_random_data.py    # Self-contained demo (no real data needed)
│
├── setup.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1 — Clone the repository

```bash
git clone https://github.com/your-username/som-algo-trading.git
cd som-algo-trading
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical arrays |
| `scikit-learn` | `StandardScaler`, PCA |
| `minisom` | Self-Organizing Map implementation |
| `matplotlib` *(optional)* | U-matrix, equity curve plots |

### 4 — Install the package in editable mode

```bash
pip install -e .
```

### 5 — Verify the installation

```bash
python -c "from som_algo_trading import SOMTradingPipeline; print('OK')"
```

---

## Quick Start

Run the self-contained demo (uses synthetic data — no real data required):

```bash
python examples/demo_random_data.py
```

Expected output (truncated):

```
==================================================
         BACKTEST SUMMARY
==================================================
  Total test windows   : 234
  BUY  signals         : 47
  SELL signals         : 31
  HOLD signals         : 156
  HALTED (anomaly)     : 0
--------------------------------------------------
  Directional accuracy : 61.54%
  BUY  win rate        : 63.83%
  SELL win rate        : 58.06%
--------------------------------------------------
  Sharpe ratio         : 0.87
  Max drawdown         : -8.23%
  Cumulative return    : +5.41%
==================================================

=== LIVE SIGNAL ===
Signal          : HOLD
BMU             : (4, 3)
Cluster matches : 18
Reason          : Cluster (4, 3): WR=55.56%, E[r]=+0.0003% — insufficient directional conviction ...
```

---

## Using Your Own Data

### Step 1 — Prepare your price series

Your data must be a **1-D NumPy array of daily closing prices in chronological order**:

```python
import numpy as np

# Option A: from a CSV
import pandas as pd
df = pd.read_csv("prices.csv", parse_dates=["Date"], index_col="Date")
prices = df["Close"].dropna().to_numpy()

# Option B: from a dict / list
prices = np.array([100.0, 101.2, 99.8, 102.5, ...])
```

> **Requirements:**
> - All values must be **positive** (> 0).
> - No NaN or Inf values.
> - Minimum length: `window_size + 2` (but 500+ days recommended for meaningful clusters).

---

### Step 2 — Configure hyperparameters

```python
from som_algo_trading import TradingConfig

# Auto-recommend a grid size based on your data size
rows, cols = TradingConfig.recommend_grid(n_windows=len(prices) - 30 - 1)

config = TradingConfig(
    # Feature engineering
    window_size=30,             # days per feature window

    # SOM architecture
    grid_rows=rows,
    grid_cols=cols,
    sigma=3.0,                  # initial neighbourhood radius
    learning_rate=0.5,
    num_iterations=10_000,      # increase for larger datasets

    # Risk management
    anomaly_threshold=None,     # None = auto (95th pct of training QEs)
    anomaly_percentile=95.0,

    # Signal generation
    min_cluster_samples=30,     # minimum matches to generate a signal
    win_rate_threshold=0.60,    # 60% historical win rate required
    expected_return_threshold=0.001,   # 0.1% minimum expected return

    # Pipeline
    train_ratio=0.80,           # 80% train, 20% held-out test
    random_state=42,
)
```

---

### Step 3 — Train the pipeline

```python
from som_algo_trading import SOMTradingPipeline

pipeline = SOMTradingPipeline(config)
pipeline.fit(prices)
```

Training prints progress logs:

```
2025-01-15 10:00:01 | INFO     | ... | Phase 1 — Data Engineering
2025-01-15 10:00:01 | INFO     | ... | DataEngineer fitted | prices=1500  windows=1469  train=1175  test=294
2025-01-15 10:00:01 | INFO     | ... | Phase 2 — SOM Training
2025-01-15 10:00:01 | INFO     | ... | Initialising SOM weights via PCA (10x10, sigma=3.00, lr=0.500) ...
2025-01-15 10:00:04 | INFO     | ... | Training SOM for 10000 iterations ...
2025-01-15 10:00:18 | INFO     | ... | SOM training complete.
2025-01-15 10:00:18 | INFO     | ... | Cluster map built | 1175 windows → 87 / 100 nodes used
2025-01-15 10:00:18 | INFO     | ... | Anomaly threshold set to 0.8341 (95th percentile of training QEs).
```

---

### Step 4 — Get a live trading signal

Pass the **most recent closing prices** (at least `window_size + 1` days):

```python
# Simulate "today's" data: last 35 prices from your series
live_prices = prices[-(config.window_size + 5):]

result = pipeline.predict(live_prices)
print(result.summary())
```

Output:

```
Signal          : BUY
BMU             : (3, 7)
Cluster matches : 42
Expected return : +0.1823%
Win rate        : 64.29%
Reason          : Cluster (3, 7): WR=64.29% >= 60% and E[r]=+0.1823% >= 0.100% (42 matches).
```

Access fields programmatically:

```python
result.signal           # "BUY", "SELL", or "HOLD"
result.bmu              # (row, col) e.g. (3, 7)
result.n_matches        # 42
result.expected_return  # 0.001823
result.win_rate         # 0.6429
result.reason           # full explanation string
result.matched_returns  # np.ndarray of historical forward returns
```

---

### Step 5 — Evaluate on the held-out test set

```python
from som_algo_trading.utils import backtest_summary

bt = pipeline.backtest()
print(backtest_summary(bt))
```

---

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | int | 30 | Days per sliding-window feature vector |
| `grid_rows` | int | 10 | SOM grid rows |
| `grid_cols` | int | 10 | SOM grid columns |
| `sigma` | float | 3.0 | Initial neighbourhood radius |
| `learning_rate` | float | 0.5 | Initial SOM learning rate |
| `num_iterations` | int | 10 000 | SOM training epochs |
| `anomaly_threshold` | float \| None | None | Max QE before halting (auto if None) |
| `anomaly_percentile` | float | 95.0 | Percentile of training QEs used for auto threshold |
| `min_cluster_samples` | int | 30 | Min cluster size for a non-HOLD signal |
| `win_rate_threshold` | float | 0.60 | Min P(profit) for BUY/SELL signal |
| `expected_return_threshold` | float | 0.001 | Min \|E[r]\| for BUY/SELL signal |
| `train_ratio` | float | 0.80 | Train/test split fraction |
| `random_state` | int \| None | 42 | Reproducibility seed |

### Grid-size heuristic

```python
# For N sliding windows in your training data:
total_nodes ≈ 5 × √N

# Example: 1 000 training windows → ~158 nodes → 13×13 grid
rows, cols = TradingConfig.recommend_grid(n_windows=1000)
# → (13, 13)
```

---

## Pipeline Phases

You can use each phase independently if needed:

```python
from som_algo_trading import DataEngineer, SOMCore, LiveInference, ForecastingEngine

# Phase 1
de = DataEngineer(window_size=30)
X_train, y_train, X_test, y_test = de.fit_transform(prices)

# Phase 2
core = SOMCore(grid_rows=10, grid_cols=10, window_size=30, num_iterations=5000)
core.fit(X_train)
core.build_cluster_map(X_train)

# Phase 3
threshold = LiveInference.set_threshold_from_training(de, core, percentile=95)
li = LiveInference(de, core, anomaly_threshold=threshold)
inference_result = li.run(live_prices)

if inference_result.is_ok:
    # Phase 4
    fe = ForecastingEngine(core.cluster_map, y_train,
                           min_cluster_samples=30,
                           win_rate_threshold=0.60,
                           expected_return_threshold=0.001)
    forecast = fe.predict(inference_result.bmu)
    print(forecast.summary())
```

---

## Backtest & Evaluation

`pipeline.backtest()` runs a walk-forward evaluation on the **held-out test windows**:

```python
bt = pipeline.backtest()

# Raw data for custom analysis
bt["signals"]           # List[ForecastResult]
bt["actual_returns"]    # np.ndarray — actual T+1 returns for each test window
bt["accuracy"]          # Overall directional accuracy (BUY+SELL only)
bt["buy_win_rate"]      # Fraction of BUY signals where actual return > 0
bt["sell_win_rate"]     # Fraction of SELL signals where actual return < 0
bt["n_buy"]             # Count of BUY signals
bt["n_sell"]            # Count of SELL signals
bt["n_hold"]            # Count of HOLD signals (includes insufficient-data HOLDs)
bt["n_halted"]          # Count of anomaly-gate halts
```

**Computing additional metrics yourself:**

```python
from som_algo_trading.utils import sharpe_ratio, max_drawdown, cumulative_returns
from som_algo_trading.forecaster import Signal
import numpy as np

signals = bt["signals"]
actual  = bt["actual_returns"]

buy_mask  = np.array([s.signal == Signal.BUY  for s in signals])
sell_mask = np.array([s.signal == Signal.SELL for s in signals])

strategy_returns = np.zeros(len(signals))
strategy_returns[buy_mask]  =  actual[buy_mask]
strategy_returns[sell_mask] = -actual[sell_mask]

print("Sharpe :", sharpe_ratio(strategy_returns[strategy_returns != 0]))
print("Max DD  :", max_drawdown(cumulative_returns(strategy_returns)))
```

---

## Save & Load a Trained Pipeline

```python
# Save after fitting
pipeline.save("models/my_pipeline")

# Files written:
#   models/my_pipeline/config.pkl
#   models/my_pipeline/data_engineer_scaler.pkl
#   models/my_pipeline/som_core.pkl
#   models/my_pipeline/y_train.npy

# Load in a new session
from som_algo_trading import SOMTradingPipeline
pipeline = SOMTradingPipeline.load("models/my_pipeline")
result = pipeline.predict(live_prices)
```

---

## Visualisations

Requires `matplotlib`:

```bash
pip install matplotlib
```

```python
from som_algo_trading.utils import plot_u_matrix, plot_cluster_signals, plot_equity_curve

# 1. U-Matrix — lighter = similar neurons, darker = boundaries between regimes
plot_u_matrix(pipeline.som_core)

# 2. Signal map — green=BUY, red=SELL, grey=HOLD nodes on the grid
plot_cluster_signals(pipeline.forecaster,
                     grid_rows=config.grid_rows,
                     grid_cols=config.grid_cols)

# 3. Equity curve vs buy-and-hold
bt = pipeline.backtest()
plot_equity_curve(bt)
```

---

## Running the Tests

```bash
# Install dev dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=som_algo_trading --cov-report=term-missing
```

Expected output:

```
tests/test_data.py::TestDataEngineer::test_returns_length       PASSED
tests/test_data.py::TestDataEngineer::test_returns_finite        PASSED
tests/test_data.py::TestDataEngineer::test_fit_transform_shapes  PASSED
...
tests/test_som.py::TestSOMCore::test_trained_flag                PASSED
...
tests/test_pipeline.py::TestSOMTradingPipeline::test_is_fitted   PASSED
...
================================ 28 passed in 45.12s ================================
```

---

## Extending the Pipeline

### Use a different SOM library

Replace `SOMCore` with your own class that exposes:
- `fit(X)` → `self`
- `build_cluster_map(X)` → `dict`
- `get_bmu(vec)` → `(row, col)`
- `quantization_error(vec)` → `float`
- `training_qes` → `np.ndarray`

Then pass it to `LiveInference` and `ForecastingEngine` directly.

### Add position sizing

Use `ForecastResult.expected_return` and `ForecastResult.win_rate` to size positions:

```python
result = pipeline.predict(live_prices)
if result.signal == "BUY":
    kelly_fraction = (result.win_rate - (1 - result.win_rate)) / 1.0
    position_size = min(kelly_fraction, 0.20)   # cap at 20 %
```

### Tune hyperparameters

Loop over a grid and call `pipeline.backtest()` to compare:

```python
for ws in [20, 30, 50]:
    for wrt in [0.55, 0.60, 0.65]:
        cfg = TradingConfig(window_size=ws, win_rate_threshold=wrt, ...)
        pipe = SOMTradingPipeline(cfg).fit(prices)
        bt = pipe.backtest()
        print(ws, wrt, bt["accuracy"])
```

---

## Disclaimer

This software is provided for **educational and research purposes only**.
It does not constitute financial advice. Past performance of any backtested
strategy is not indicative of future results. Always consult a qualified
financial professional before trading real capital.
