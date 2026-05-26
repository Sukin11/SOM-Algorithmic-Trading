"""
config.py
---------
Central hyperparameter configuration for the SOM trading pipeline.
All tunable knobs live here — change them to backtest or optimize.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class TradingConfig:
    """
    Hyperparameter container for the SOM-based trading pipeline.

    All four phases read from this single object, making it the sole
    source of truth for every experiment.

    Parameters
    ----------
    window_size : int
        Number of days in each sliding-window feature vector.
        Typical range: 20–60.
    grid_rows : int
        Number of rows in the SOM output grid.
    grid_cols : int
        Number of columns in the SOM output grid.
        Heuristic: total_nodes ≈ 5 * sqrt(N) where N = number of windows.
    sigma : float
        Initial neighbourhood radius for the SOM (Gaussian kernel).
        Typical start: 3.0 – 5.0.
    learning_rate : float
        Initial learning rate for weight updates.
        Typical start: 0.5.
    num_iterations : int
        Total SOM training epochs.
        Typical range: 10 000 – 50 000.
    anomaly_threshold : float or None
        Maximum allowed Quantization Error (QE) for live data.
        If None, it is auto-calculated as the 95th-percentile of
        training QEs after fitting.
    min_cluster_samples : int
        Minimum historical matches a cluster must hold before a
        forecast is generated.  Below this → HOLD.
    win_rate_threshold : float
        Minimum win-rate (P(return > 0)) to trigger BUY/SELL signal.
        BUY requires P(up) >= threshold; SELL requires P(down) >= threshold.
    expected_return_threshold : float
        Minimum absolute expected return to confirm a directional signal.
    train_ratio : float
        Fraction of data used for training (rest is held-out test set).
    random_state : int or None
        Seed for reproducibility.

    Examples
    --------
    >>> cfg = TradingConfig(window_size=30, grid_rows=10, grid_cols=10)
    >>> cfg.total_nodes
    100

    Use the helper to get the SOM grid size recommended by the heuristic:
    >>> TradingConfig.recommend_grid(n_windows=500)
    (8, 8)
    """

    # --- Feature engineering ---
    window_size: int = 30

    # --- SOM architecture ---
    grid_rows: int = 10
    grid_cols: int = 10
    sigma: float = 3.0
    learning_rate: float = 0.5
    num_iterations: int = 10_000

    # --- Risk management ---
    anomaly_threshold: Optional[float] = None   # auto-set after fit if None
    anomaly_percentile: float = 95.0            # percentile used for auto threshold

    # --- Signal generation ---
    min_cluster_samples: int = 30
    win_rate_threshold: float = 0.60
    expected_return_threshold: float = 0.001

    # --- Pipeline behaviour ---
    train_ratio: float = 0.80
    random_state: Optional[int] = 42

    # ------------------------------------------------------------------ #
    #  Derived helpers                                                     #
    # ------------------------------------------------------------------ #

    @property
    def total_nodes(self) -> int:
        """Total number of SOM neurons."""
        return self.grid_rows * self.grid_cols

    @staticmethod
    def recommend_grid(n_windows: int) -> tuple:
        """
        Return (rows, cols) satisfying the heuristic
        total_nodes ≈ 5 * sqrt(N).

        Parameters
        ----------
        n_windows : int
            Number of sliding windows in the training dataset.

        Returns
        -------
        tuple of (int, int)
            Recommended (grid_rows, grid_cols).
        """
        target = max(4, int(5 * math.sqrt(n_windows)))
        side = max(2, round(math.sqrt(target)))
        return (side, side)

    def validate(self):
        """
        Raise ValueError for obviously wrong hyperparameter combinations.
        Called automatically by the pipeline before training.
        """
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2.")
        if self.grid_rows < 2 or self.grid_cols < 2:
            raise ValueError("Grid dimensions must be >= 2.")
        if not (0 < self.learning_rate <= 1):
            raise ValueError("learning_rate must be in (0, 1].")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0.")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be >= 1.")
        if not (0 < self.win_rate_threshold < 1):
            raise ValueError("win_rate_threshold must be in (0, 1).")
        if self.expected_return_threshold < 0:
            raise ValueError("expected_return_threshold must be >= 0.")
        if self.min_cluster_samples < 1:
            raise ValueError("min_cluster_samples must be >= 1.")
        if not (0 < self.train_ratio < 1):
            raise ValueError("train_ratio must be in (0, 1).")

    def __repr__(self) -> str:
        lines = [
            "TradingConfig(",
            f"  window_size            = {self.window_size}",
            f"  grid               = {self.grid_rows}x{self.grid_cols} "
            f"({self.total_nodes} nodes)",
            f"  sigma                  = {self.sigma}",
            f"  learning_rate          = {self.learning_rate}",
            f"  num_iterations         = {self.num_iterations:,}",
            f"  anomaly_threshold      = {self.anomaly_threshold}",
            f"  anomaly_percentile     = {self.anomaly_percentile}",
            f"  min_cluster_samples    = {self.min_cluster_samples}",
            f"  win_rate_threshold     = {self.win_rate_threshold}",
            f"  expected_return_thresh = {self.expected_return_threshold}",
            f"  train_ratio            = {self.train_ratio}",
            f"  random_state           = {self.random_state}",
            ")",
        ]
        return "\n".join(lines)
