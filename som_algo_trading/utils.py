"""
utils.py  —  Shared Utilities
------------------------------
Logging setup, performance metrics, and optional matplotlib visualisations.
All plotting functions gracefully skip when matplotlib is unavailable.
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------ #
#  Logging                                                             #
# ------------------------------------------------------------------ #

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Configure root logger with a clean formatter.

    Parameters
    ----------
    level : int
        Logging level, e.g. ``logging.DEBUG``, ``logging.INFO``.
    log_file : str or None
        If provided, also write logs to this file path.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


# ------------------------------------------------------------------ #
#  Price / return generators (for testing)                            #
# ------------------------------------------------------------------ #

def generate_synthetic_prices(
    n_days: int = 1000,
    start_price: float = 100.0,
    annual_drift: float = 0.08,
    annual_volatility: float = 0.20,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """
    Generate a Geometric Brownian Motion price series.

    Parameters
    ----------
    n_days : int
        Total number of daily price observations.
    start_price : float
        Initial price.
    annual_drift : float
        Expected annual return (mu).
    annual_volatility : float
        Annual standard deviation of returns (sigma).
    random_state : int or None

    Returns
    -------
    np.ndarray, shape (n_days,)
    """
    rng = np.random.default_rng(random_state)
    dt = 1 / 252
    daily_drift = (annual_drift - 0.5 * annual_volatility ** 2) * dt
    daily_vol = annual_volatility * np.sqrt(dt)

    log_returns = rng.normal(daily_drift, daily_vol, size=n_days - 1)
    prices = np.empty(n_days)
    prices[0] = start_price
    prices[1:] = start_price * np.exp(np.cumsum(log_returns))
    return prices


def generate_regime_prices(
    n_days: int = 1500,
    start_price: float = 100.0,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """
    Generate a multi-regime price series (bull / bear / sideways) for
    richer SOM testing.

    Parameters
    ----------
    n_days : int
    start_price : float
    random_state : int or None

    Returns
    -------
    np.ndarray, shape (n_days,)
    """
    rng = np.random.default_rng(random_state)
    regimes = [
        {"drift": 0.15,  "vol": 0.15, "days": n_days // 3},   # bull
        {"drift": -0.20, "vol": 0.30, "days": n_days // 3},   # bear
        {"drift": 0.02,  "vol": 0.10, "days": n_days - 2 * (n_days // 3)},  # sideways
    ]
    dt = 1 / 252
    all_returns: list = []
    for r in regimes:
        d = (r["drift"] - 0.5 * r["vol"] ** 2) * dt
        v = r["vol"] * np.sqrt(dt)
        all_returns.extend(rng.normal(d, v, size=r["days"]).tolist())

    prices = np.empty(n_days)
    prices[0] = start_price
    prices[1:] = start_price * np.exp(np.cumsum(all_returns[: n_days - 1]))
    return prices


# ------------------------------------------------------------------ #
#  Performance metrics                                                 #
# ------------------------------------------------------------------ #

def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Annualised Sharpe ratio (risk-free rate = 0).

    Parameters
    ----------
    returns : np.ndarray
        Daily strategy returns.
    periods_per_year : int

    Returns
    -------
    float
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0 or np.std(returns) == 0:
        return float("nan")
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum peak-to-trough drawdown of an equity curve.

    Parameters
    ----------
    equity_curve : np.ndarray
        Cumulative portfolio value (not returns).

    Returns
    -------
    float
        Drawdown expressed as a negative fraction, e.g. -0.25 for -25 %.
    """
    equity_curve = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return float(drawdown.min())


def cumulative_returns(returns: np.ndarray) -> np.ndarray:
    """
    Convert a series of daily returns to a cumulative equity curve
    starting at 1.0.

    Parameters
    ----------
    returns : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return np.cumprod(1 + np.asarray(returns, dtype=float))


def backtest_summary(backtest_result: dict) -> str:
    """
    Format the output of :meth:`SOMTradingPipeline.backtest` as a
    human-readable string.

    Parameters
    ----------
    backtest_result : dict
        As returned by ``pipeline.backtest()``.

    Returns
    -------
    str
    """
    bt = backtest_result
    signals = bt["signals"]
    actual  = bt["actual_returns"]

    from .forecaster import Signal

    buy_mask  = np.array([s.signal == Signal.BUY  for s in signals])
    sell_mask = np.array([s.signal == Signal.SELL for s in signals])

    # Compute strategy returns: BUY → actual; SELL → -actual; else 0
    strategy_returns = np.zeros(len(signals))
    strategy_returns[buy_mask]  =  actual[buy_mask]
    strategy_returns[sell_mask] = -actual[sell_mask]

    sr = sharpe_ratio(strategy_returns[strategy_returns != 0])
    cum = cumulative_returns(strategy_returns)
    mdd = max_drawdown(cum)

    lines = [
        "=" * 50,
        "         BACKTEST SUMMARY",
        "=" * 50,
        f"  Total test windows   : {len(signals)}",
        f"  BUY  signals         : {bt['n_buy']}",
        f"  SELL signals         : {bt['n_sell']}",
        f"  HOLD signals         : {bt['n_hold']}",
        f"  HALTED (anomaly)     : {bt['n_halted']}",
        "-" * 50,
        f"  Directional accuracy : {bt['accuracy']:.2%}" if not np.isnan(bt['accuracy']) else "  Directional accuracy : N/A",
        f"  BUY  win rate        : {bt['buy_win_rate']:.2%}"  if not np.isnan(bt['buy_win_rate'])  else "  BUY  win rate        : N/A",
        f"  SELL win rate        : {bt['sell_win_rate']:.2%}" if not np.isnan(bt['sell_win_rate']) else "  SELL win rate        : N/A",
        "-" * 50,
        f"  Sharpe ratio         : {sr:.2f}" if not np.isnan(sr) else "  Sharpe ratio         : N/A",
        f"  Max drawdown         : {mdd:.2%}",
        f"  Cumulative return    : {(cum[-1] - 1):.2%}",
        "=" * 50,
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Visualisation (optional — requires matplotlib)                     #
# ------------------------------------------------------------------ #

def plot_u_matrix(
    som_core,
    title: str = "SOM U-Matrix",
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plot the Unified Distance Matrix of the trained SOM.

    Requires matplotlib.  If unavailable, prints a warning and returns.

    Parameters
    ----------
    som_core : SOMCore
    title : str
    figsize : tuple
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping U-matrix plot.")
        return

    u_mat = som_core.u_matrix()
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(u_mat, cmap="bone_r", interpolation="nearest")
    plt.colorbar(img, ax=ax, label="Mean distance to neighbours")
    ax.set_title(title)
    ax.set_xlabel("SOM column")
    ax.set_ylabel("SOM row")
    plt.tight_layout()
    plt.show()


def plot_cluster_signals(
    forecaster,
    grid_rows: int,
    grid_cols: int,
    title: str = "Cluster Signal Map",
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Draw a colour-coded grid showing the signal (BUY/SELL/HOLD) per node.

    Requires matplotlib.  If unavailable, prints a warning and returns.

    Parameters
    ----------
    forecaster : ForecastingEngine
    grid_rows : int
    grid_cols : int
    title : str
    figsize : tuple
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed — skipping cluster signal plot.")
        return

    colour_map = {"BUY": "green", "SELL": "red", "HOLD": "grey"}
    grid = np.full((grid_rows, grid_cols), "HOLD", dtype=object)

    stats = forecaster.cluster_statistics()
    for (r, c), s in stats.items():
        grid[r, c] = s["signal"]

    fig, ax = plt.subplots(figsize=figsize)
    for r in range(grid_rows):
        for c in range(grid_cols):
            colour = colour_map.get(grid[r, c], "grey")
            rect = plt.Rectangle([c, grid_rows - r - 1], 1, 1, color=colour, alpha=0.7)
            ax.add_patch(rect)
            n = stats.get((r, c), {}).get("n", 0)
            ax.text(c + 0.5, grid_rows - r - 0.5, str(n),
                    ha="center", va="center", fontsize=7, color="white")

    ax.set_xlim(0, grid_cols)
    ax.set_ylim(0, grid_rows)
    ax.set_xticks(range(grid_cols))
    ax.set_yticks(range(grid_rows))
    ax.set_title(title)

    patches = [mpatches.Patch(color=v, label=k) for k, v in colour_map.items()]
    ax.legend(handles=patches, loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_equity_curve(
    backtest_result: dict,
    title: str = "Strategy Equity Curve",
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """
    Plot the cumulative equity curve from a backtest result dict.

    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping equity curve plot.")
        return

    from .forecaster import Signal

    signals = backtest_result["signals"]
    actual  = backtest_result["actual_returns"]

    buy_mask  = np.array([s.signal == Signal.BUY  for s in signals])
    sell_mask = np.array([s.signal == Signal.SELL for s in signals])

    strat = np.zeros(len(signals))
    strat[buy_mask]  =  actual[buy_mask]
    strat[sell_mask] = -actual[sell_mask]

    eq  = cumulative_returns(strat)
    bnh = cumulative_returns(actual)          # buy-and-hold benchmark

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(eq,  label="SOM Strategy", linewidth=1.5)
    ax.plot(bnh, label="Buy & Hold",   linewidth=1.5, linestyle="--", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Test window index")
    ax.set_ylabel("Cumulative return (1 = start)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
