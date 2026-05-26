"""
pipeline.py  —  End-to-End Orchestrator
-----------------------------------------
SOMTradingPipeline ties all four phases together behind a single
sklearn-style API (fit / predict / backtest).

Usage
-----
>>> from som_algo_trading import SOMTradingPipeline, TradingConfig
>>> cfg = TradingConfig(window_size=30, grid_rows=10, grid_cols=10)
>>> pipeline = SOMTradingPipeline(cfg)
>>> pipeline.fit(prices)
>>> result = pipeline.predict(recent_prices)
>>> print(result.summary())
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, List

import numpy as np

from .config import TradingConfig
from .data import DataEngineer
from .forecaster import ForecastingEngine, ForecastResult
from .inference import InferenceResult, LiveInference
from .som import SOMCore

logger = logging.getLogger(__name__)


class SOMTradingPipeline:
    """
    End-to-end SOM-based algorithmic trading pipeline.

    Parameters
    ----------
    config : TradingConfig
        All hyperparameters in one place.

    Attributes
    ----------
    data_engineer : DataEngineer
    som_core : SOMCore
    live_inference : LiveInference
    forecaster : ForecastingEngine
    is_fitted : bool
    X_test : np.ndarray or None
    y_test : np.ndarray or None

    Examples
    --------
    Fit on historical prices:
    >>> pipeline = SOMTradingPipeline(TradingConfig())
    >>> pipeline.fit(prices)

    Predict on live data (last window_size+1 prices):
    >>> result = pipeline.predict(live_prices)
    >>> print(result.summary())

    Run a walk-forward backtest on held-out test data:
    >>> bt = pipeline.backtest()
    >>> print(bt)
    """

    def __init__(self, config: Optional[TradingConfig] = None) -> None:
        self.config = config or TradingConfig()
        self.config.validate()

        self.data_engineer: Optional[DataEngineer] = None
        self.som_core: Optional[SOMCore] = None
        self.live_inference: Optional[LiveInference] = None
        self.forecaster: Optional[ForecastingEngine] = None

        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------ #
    #  Fit                                                                 #
    # ------------------------------------------------------------------ #

    def fit(self, prices: np.ndarray) -> "SOMTradingPipeline":
        """
        Run the full training pipeline (Phases 1 → 2) on *prices*.

        Parameters
        ----------
        prices : np.ndarray, shape (T,)
            Chronological daily closing prices.

        Returns
        -------
        self
        """
        cfg = self.config
        logger.info("=== SOMTradingPipeline.fit() START ===")

        # ---- Phase 1: Data engineering -------------------------------- #
        logger.info("Phase 1 — Data Engineering")
        self.data_engineer = DataEngineer(window_size=cfg.window_size)
        X_train, y_train, X_test, y_test = self.data_engineer.fit_transform(
            prices, train_ratio=cfg.train_ratio
        )
        self.X_test = X_test
        self.y_test = y_test

        # ---- Phase 2: SOM training ------------------------------------ #
        logger.info("Phase 2 — SOM Training")
        self.som_core = SOMCore(
            grid_rows=cfg.grid_rows,
            grid_cols=cfg.grid_cols,
            window_size=cfg.window_size,
            sigma=cfg.sigma,
            learning_rate=cfg.learning_rate,
            num_iterations=cfg.num_iterations,
            random_state=cfg.random_state,
        )
        self.som_core.fit(X_train)
        self.som_core.build_cluster_map(X_train)

        # ---- Anomaly threshold ---------------------------------------- #
        if cfg.anomaly_threshold is None:
            threshold = LiveInference.set_threshold_from_training(
                self.data_engineer,
                self.som_core,
                percentile=cfg.anomaly_percentile,
            )
        else:
            threshold = cfg.anomaly_threshold
            logger.info("Using user-supplied anomaly threshold: %.4f", threshold)

        # ---- Phase 3: Live inference module --------------------------- #
        self.live_inference = LiveInference(
            data_engineer=self.data_engineer,
            som_core=self.som_core,
            anomaly_threshold=threshold,
        )

        # ---- Phase 4: Forecasting engine ------------------------------ #
        self.forecaster = ForecastingEngine(
            cluster_map=self.som_core.cluster_map,
            forward_returns=y_train,
            min_cluster_samples=cfg.min_cluster_samples,
            win_rate_threshold=cfg.win_rate_threshold,
            expected_return_threshold=cfg.expected_return_threshold,
        )

        self.is_fitted = True
        logger.info("=== SOMTradingPipeline.fit() COMPLETE ===")
        return self

    # ------------------------------------------------------------------ #
    #  Predict                                                             #
    # ------------------------------------------------------------------ #

    def predict(self, live_prices: np.ndarray) -> ForecastResult:
        """
        Generate a trading signal for the current market.

        Runs Phase 3 (risk gate) → Phase 4 (forecast).
        If the risk gate is triggered, returns a HOLD with the halt message.

        Parameters
        ----------
        live_prices : np.ndarray
            Recent closing prices.  Must contain at least
            ``window_size + 1`` values.

        Returns
        -------
        ForecastResult
        """
        self._check_fitted()

        # Phase 3
        inference_result: InferenceResult = self.live_inference.run(live_prices)

        if not inference_result.is_ok:
            # Return a HOLD forecast with halt reason
            from .forecaster import Signal
            return ForecastResult(
                signal=Signal.HOLD,
                bmu=(-1, -1),
                n_matches=0,
                expected_return=None,
                win_rate=None,
                reason=inference_result.message,
            )

        # Phase 4
        return self.forecaster.predict(inference_result.bmu)

    # ------------------------------------------------------------------ #
    #  Backtest                                                            #
    # ------------------------------------------------------------------ #

    def backtest(self) -> dict:
        """
        Walk-forward backtest on the held-out test windows.

        Each test window is treated as a "live" vector, passed through
        Phase 3 (risk gate) and Phase 4 (forecast).  The actual T+1
        return (``y_test``) is used to evaluate signal accuracy.

        Returns
        -------
        dict with keys:
            - ``signals``        : list of ForecastResult
            - ``actual_returns`` : np.ndarray of y_test values
            - ``accuracy``       : fraction of correct directional calls
            - ``buy_win_rate``   : win rate on BUY signals
            - ``sell_win_rate``  : win rate on SELL signals
            - ``n_buy``          : number of BUY signals
            - ``n_sell``         : number of SELL signals
            - ``n_hold``         : number of HOLD signals
            - ``n_halted``       : number of risk-gate halts
        """
        self._check_fitted()

        if self.X_test is None or len(self.X_test) == 0:
            raise RuntimeError("No test data available. Call fit() first.")

        signals: List[ForecastResult] = []
        n_halted = 0

        from .forecaster import Signal

        for i, vec in enumerate(self.X_test):
            # Simulate live inference using the pre-scaled test vector
            bmu = self.som_core.get_bmu(vec)
            qe = self.som_core.quantization_error(vec)

            if qe > self.live_inference.anomaly_threshold:
                n_halted += 1
                result = ForecastResult(
                    signal=Signal.HOLD,
                    bmu=(-1, -1),
                    n_matches=0,
                    expected_return=None,
                    win_rate=None,
                    reason="HALTED: High QE on test window.",
                )
            else:
                result = self.forecaster.predict(bmu)

            signals.append(result)

        actual = self.y_test
        buy_mask  = np.array([s.signal == Signal.BUY  for s in signals])
        sell_mask = np.array([s.signal == Signal.SELL for s in signals])
        hold_mask = np.array([s.signal == Signal.HOLD for s in signals])

        n_buy  = int(buy_mask.sum())
        n_sell = int(sell_mask.sum())
        n_hold = int(hold_mask.sum())

        # Directional accuracy: BUY correct if actual > 0; SELL if actual < 0
        correct = 0
        total_directional = n_buy + n_sell
        if n_buy > 0:
            correct += int((actual[buy_mask] > 0).sum())
        if n_sell > 0:
            correct += int((actual[sell_mask] < 0).sum())

        accuracy = correct / total_directional if total_directional > 0 else float("nan")

        buy_win_rate = (
            float((actual[buy_mask] > 0).mean()) if n_buy > 0 else float("nan")
        )
        sell_win_rate = (
            float((actual[sell_mask] < 0).mean()) if n_sell > 0 else float("nan")
        )

        result_dict = {
            "signals": signals,
            "actual_returns": actual,
            "accuracy": accuracy,
            "buy_win_rate": buy_win_rate,
            "sell_win_rate": sell_win_rate,
            "n_buy": n_buy,
            "n_sell": n_sell,
            "n_hold": n_hold,
            "n_halted": n_halted,
        }

        logger.info(
            "Backtest | BUY=%d WR=%.2f | SELL=%d WR=%.2f | HOLD=%d | "
            "HALTED=%d | Directional Acc=%.2f",
            n_buy, buy_win_rate if n_buy else 0,
            n_sell, sell_win_rate if n_sell else 0,
            n_hold, n_halted,
            accuracy if not np.isnan(accuracy) else 0,
        )
        return result_dict

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, directory: str | Path) -> None:
        """
        Persist the fitted pipeline to *directory*.

        Saves:
            - ``config.pkl``
            - ``data_engineer_scaler.pkl``
            - ``som_core.pkl``

        Parameters
        ----------
        directory : str or Path
        """
        self._check_fitted()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        with open(d / "config.pkl", "wb") as fh:
            pickle.dump(self.config, fh)

        self.data_engineer.save_scaler(d / "data_engineer_scaler.pkl")
        self.som_core.save(d / "som_core.pkl")

        # Save y_train reference (needed to rebuild forecaster on load)
        np.save(str(d / "y_train.npy"), self.forecaster.forward_returns)

        logger.info("Pipeline saved → %s", d)

    @classmethod
    def load(cls, directory: str | Path) -> "SOMTradingPipeline":
        """
        Restore a fitted pipeline from *directory*.

        Parameters
        ----------
        directory : str or Path

        Returns
        -------
        SOMTradingPipeline
        """
        d = Path(directory)

        with open(d / "config.pkl", "rb") as fh:
            config = pickle.load(fh)

        pipeline = cls(config)

        pipeline.data_engineer = DataEngineer(window_size=config.window_size)
        pipeline.data_engineer.load_scaler(d / "data_engineer_scaler.pkl")

        pipeline.som_core = SOMCore.load(d / "som_core.pkl")

        y_train = np.load(str(d / "y_train.npy"))

        threshold = (
            config.anomaly_threshold
            if config.anomaly_threshold is not None
            else LiveInference.set_threshold_from_training(
                pipeline.data_engineer,
                pipeline.som_core,
                percentile=config.anomaly_percentile,
            )
        )

        pipeline.live_inference = LiveInference(
            data_engineer=pipeline.data_engineer,
            som_core=pipeline.som_core,
            anomaly_threshold=threshold,
        )

        pipeline.forecaster = ForecastingEngine(
            cluster_map=pipeline.som_core.cluster_map,
            forward_returns=y_train,
            min_cluster_samples=config.min_cluster_samples,
            win_rate_threshold=config.win_rate_threshold,
            expected_return_threshold=config.expected_return_threshold,
        )

        pipeline.is_fitted = True
        logger.info("Pipeline loaded ← %s", d)
        return pipeline

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"SOMTradingPipeline(status={status}, config={self.config!r})"
