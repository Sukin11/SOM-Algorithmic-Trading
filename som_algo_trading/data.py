"""
data.py  —  Phase 1: Data Engineering Pipeline
-----------------------------------------------
Responsibilities
    1. Convert raw prices → daily percentage returns.
    2. Build overlapping sliding-window feature matrix.
    3. Align each window with its T+1 forward return (target).
    4. Z-score standardise using a scaler fitted on training data only.
    5. Persist the scaler so Phase 3 (live inference) uses identical scaling.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataEngineer:
    """
    Phase-1 data engineering for the SOM trading pipeline.

    The scaler is *fitted on training windows only* and must be reused
    (never re-fitted) when processing live data in Phase 3.

    Parameters
    ----------
    window_size : int
        Number of days in each feature window.

    Attributes
    ----------
    scaler : sklearn.preprocessing.StandardScaler
        Fitted after :meth:`fit_transform` is called.
    is_fitted : bool
        True once :meth:`fit_transform` has been called successfully.

    Examples
    --------
    >>> de = DataEngineer(window_size=30)
    >>> X_train, y_train, X_test, y_test = de.fit_transform(prices, train_ratio=0.8)
    >>> live_vec = de.transform_live(live_prices[-31:])
    """

    def __init__(self, window_size: int = 30) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2.")
        self.window_size = window_size
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fit_transform(
        self,
        prices: np.ndarray,
        train_ratio: float = 0.80,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full Phase-1 pipeline: prices → scaled windows + forward returns.

        Parameters
        ----------
        prices : np.ndarray, shape (T,)
            Raw daily closing prices in chronological order.
        train_ratio : float
            Fraction of windows reserved for training; the rest become
            the held-out test set.

        Returns
        -------
        X_train : np.ndarray, shape (n_train, window_size)
            Standardised training feature windows.
        y_train : np.ndarray, shape (n_train,)
            T+1 forward returns aligned with X_train.
        X_test : np.ndarray, shape (n_test, window_size)
            Standardised test feature windows (scaler NOT re-fitted).
        y_test : np.ndarray, shape (n_test,)
            T+1 forward returns aligned with X_test.
        """
        prices = self._validate_prices(prices)

        returns = self._compute_returns(prices)
        X_raw, y = self._build_windows(returns)

        split = int(len(X_raw) * train_ratio)
        if split < 10:
            raise ValueError(
                f"Training set has only {split} windows — need at least 10. "
                "Either reduce window_size or supply more data."
            )

        X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
        y_train, y_test = y[:split], y[split:]

        # Fit scaler on training data ONLY
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)

        self.is_fitted = True

        logger.info(
            "DataEngineer fitted | prices=%d  windows=%d  "
            "train=%d  test=%d",
            len(prices), len(X_raw), len(X_train), len(X_test),
        )
        return X_train, y_train, X_test, y_test

    def transform_live(self, live_prices: np.ndarray) -> np.ndarray:
        """
        Scale the most-recent window using the *training* scaler.

        Parameters
        ----------
        live_prices : np.ndarray
            At least ``window_size + 1`` recent closing prices so that
            returns can be computed and one full window extracted.

        Returns
        -------
        np.ndarray, shape (1, window_size)
            Single scaled feature vector ready for SOM inference.
        """
        self._check_fitted()
        live_prices = self._validate_prices(live_prices, min_len=self.window_size + 1)

        returns = self._compute_returns(live_prices)
        # Take the last window_size returns
        live_window = returns[-self.window_size:].reshape(1, -1)
        scaled = self.scaler.transform(live_window)

        logger.debug("Live window scaled — shape %s", scaled.shape)
        return scaled

    def get_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Utility: compute percentage returns from a price series.

        Parameters
        ----------
        prices : np.ndarray, shape (T,)

        Returns
        -------
        np.ndarray, shape (T-1,)
        """
        prices = self._validate_prices(prices)
        return self._compute_returns(prices)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save_scaler(self, path: str | Path) -> None:
        """Pickle the fitted scaler to *path*."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.scaler, fh)
        logger.info("Scaler saved → %s", path)

    def load_scaler(self, path: str | Path) -> None:
        """Restore a previously saved scaler from *path*."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")
        with open(path, "rb") as fh:
            self.scaler = pickle.load(fh)
        self.is_fitted = True
        logger.info("Scaler loaded ← %s", path)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_returns(prices: np.ndarray) -> np.ndarray:
        """r_t = (P_t - P_{t-1}) / P_{t-1}"""
        return np.diff(prices) / prices[:-1]

    def _build_windows(
        self, returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slide a window of length ``window_size`` across *returns*.

        Each window X_i maps to forward return y_i = returns[i + window_size],
        which is the return on the day *after* the window ends.

        Window indices that cannot have a forward return (the last window)
        are excluded, ensuring X and y are perfectly aligned.
        """
        ws = self.window_size
        L = len(returns)
        # We need at least one return after the window → stop at L - ws - 1 + 1
        n_windows = L - ws - 1  # last valid start index is L - ws - 2
        if n_windows <= 0:
            raise ValueError(
                f"Not enough data to form even one window. "
                f"Need > {ws + 1} returns, got {L}."
            )

        X = np.array([returns[i : i + ws] for i in range(n_windows)])
        y = np.array([returns[i + ws] for i in range(n_windows)])
        return X, y

    @staticmethod
    def _validate_prices(
        prices: np.ndarray, min_len: int = 3
    ) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        if prices.ndim != 1:
            raise ValueError("prices must be a 1-D array.")
        if len(prices) < min_len:
            raise ValueError(
                f"prices must have at least {min_len} elements, got {len(prices)}."
            )
        if np.any(prices <= 0):
            raise ValueError("All prices must be positive (> 0).")
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            raise ValueError("prices contain NaN or Inf values.")
        return prices

    def _check_fitted(self):
        if not self.is_fitted or self.scaler is None:
            raise RuntimeError(
                "DataEngineer is not fitted. Call fit_transform() first "
                "or load a scaler with load_scaler()."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"DataEngineer(window_size={self.window_size}, status={status})"
