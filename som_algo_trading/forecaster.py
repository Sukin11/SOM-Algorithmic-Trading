"""
forecaster.py  —  Phase 4: Statistical Forecasting Engine
----------------------------------------------------------
Responsibilities
    1. Look up the BMU coordinate in the cluster_map.
    2. Check minimum sample significance.
    3. Compute expected return and win rate over matched forward returns.
    4. Emit a deterministic BUY / SELL / HOLD signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Signal enum-like constants                                          #
# ------------------------------------------------------------------ #

class Signal:
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# ------------------------------------------------------------------ #
#  Result container                                                    #
# ------------------------------------------------------------------ #

@dataclass
class ForecastResult:
    """
    Output of :meth:`ForecastingEngine.predict`.

    Attributes
    ----------
    signal : str
        One of ``"BUY"``, ``"SELL"``, ``"HOLD"``.
    bmu : tuple
        BMU coordinate queried.
    n_matches : int
        Number of historical windows in the matched cluster.
    expected_return : float or None
        Mean T+1 forward return of matched windows.
    win_rate : float or None
        Fraction of matched windows with positive T+1 return.
    reason : str
        Human-readable explanation of the signal.
    matched_returns : np.ndarray or None
        The raw forward returns of matched windows (for analysis).
    """
    signal: str
    bmu: Tuple[int, int]
    n_matches: int
    expected_return: Optional[float]
    win_rate: Optional[float]
    reason: str
    matched_returns: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        lines = [
            f"Signal          : {self.signal}",
            f"BMU             : {self.bmu}",
            f"Cluster matches : {self.n_matches}",
        ]
        if self.expected_return is not None:
            lines.append(f"Expected return : {self.expected_return:+.4%}")
        if self.win_rate is not None:
            lines.append(f"Win rate        : {self.win_rate:.2%}")
        lines.append(f"Reason          : {self.reason}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ForecastResult(signal={self.signal!r}, bmu={self.bmu}, "
            f"matches={self.n_matches}, "
            f"E[r]={self.expected_return}, win_rate={self.win_rate})"
        )


# ------------------------------------------------------------------ #
#  Main class                                                          #
# ------------------------------------------------------------------ #

class ForecastingEngine:
    """
    Phase-4 statistical forecasting based on historical cluster forward returns.

    Parameters
    ----------
    cluster_map : dict
        ``{(row, col): [window_index, ...]}``.  From :attr:`SOMCore.cluster_map`.
    forward_returns : np.ndarray, shape (n_train,)
        T+1 returns aligned with training windows (``y_train`` from Phase 1).
    min_cluster_samples : int
        Minimum matches required to generate a non-HOLD signal.
    win_rate_threshold : float
        P(return > 0) required for a directional signal.
    expected_return_threshold : float
        |E[r]| required to confirm a directional signal.

    Examples
    --------
    >>> fe = ForecastingEngine(cluster_map, y_train,
    ...                        min_cluster_samples=30,
    ...                        win_rate_threshold=0.60,
    ...                        expected_return_threshold=0.001)
    >>> result = fe.predict(bmu=(3, 7))
    >>> print(result.summary())
    """

    def __init__(
        self,
        cluster_map: Dict[Tuple[int, int], List[int]],
        forward_returns: np.ndarray,
        min_cluster_samples: int = 30,
        win_rate_threshold: float = 0.60,
        expected_return_threshold: float = 0.001,
    ) -> None:
        self.cluster_map = cluster_map
        self.forward_returns = np.asarray(forward_returns, dtype=float)
        self.min_cluster_samples = min_cluster_samples
        self.win_rate_threshold = win_rate_threshold
        self.expected_return_threshold = expected_return_threshold

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def predict(self, bmu: Tuple[int, int]) -> ForecastResult:
        """
        Generate a trading signal for *bmu*.

        Parameters
        ----------
        bmu : (row, col) tuple
            BMU coordinate from Phase 3.

        Returns
        -------
        ForecastResult
        """
        # Step 1: Cluster lookup
        indices = self.cluster_map.get(bmu, [])
        n_matches = len(indices)

        # Step 2: Significance check
        if n_matches < self.min_cluster_samples:
            reason = (
                f"Cluster {bmu} has only {n_matches} historical matches "
                f"(minimum required: {self.min_cluster_samples})."
            )
            logger.info("HOLD — %s", reason)
            return ForecastResult(
                signal=Signal.HOLD,
                bmu=bmu,
                n_matches=n_matches,
                expected_return=None,
                win_rate=None,
                reason=reason,
            )

        # Step 3: Metrics
        matched_returns = self.forward_returns[indices]
        expected_return = float(np.mean(matched_returns))
        win_rate = float(np.sum(matched_returns > 0) / n_matches)

        # Step 4: Signal logic
        signal, reason = self._determine_signal(expected_return, win_rate, bmu, n_matches)

        logger.info(
            "%s | BMU=%s | E[r]=%.4f | WR=%.2f | n=%d",
            signal, bmu, expected_return, win_rate, n_matches,
        )

        return ForecastResult(
            signal=signal,
            bmu=bmu,
            n_matches=n_matches,
            expected_return=expected_return,
            win_rate=win_rate,
            reason=reason,
            matched_returns=matched_returns,
        )

    def cluster_statistics(self) -> Dict[Tuple[int, int], dict]:
        """
        Compute summary statistics for every occupied cluster.

        Useful for post-hoc analysis or heatmap visualisations.

        Returns
        -------
        dict
            ``{bmu: {"n": int, "mean_return": float, "win_rate": float,
               "std_return": float, "signal": str}}``.
        """
        stats = {}
        for bmu, indices in self.cluster_map.items():
            ret = self.forward_returns[indices]
            win_rate = float(np.sum(ret > 0) / len(ret))
            expected_return = float(np.mean(ret))
            _, signal_label = self._determine_signal(
                expected_return, win_rate, bmu, len(indices)
            )
            stats[bmu] = {
                "n": len(indices),
                "mean_return": expected_return,
                "std_return": float(np.std(ret)),
                "win_rate": win_rate,
                "signal": self._determine_signal(expected_return, win_rate, bmu, len(indices))[0],
            }
        return stats

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _determine_signal(
        self,
        expected_return: float,
        win_rate: float,
        bmu: Tuple[int, int],
        n: int,
    ) -> Tuple[str, str]:
        wrt = self.win_rate_threshold
        ert = self.expected_return_threshold

        if win_rate >= wrt and expected_return >= ert:
            signal = Signal.BUY
            reason = (
                f"Cluster {bmu}: WR={win_rate:.2%} >= {wrt:.0%} and "
                f"E[r]={expected_return:+.4%} >= {ert:.3%} ({n} matches)."
            )
        elif win_rate <= (1 - wrt) and expected_return <= -ert:
            signal = Signal.SELL
            reason = (
                f"Cluster {bmu}: WR={win_rate:.2%} <= {1-wrt:.0%} and "
                f"E[r]={expected_return:+.4%} <= {-ert:.3%} ({n} matches)."
            )
        else:
            signal = Signal.HOLD
            reason = (
                f"Cluster {bmu}: WR={win_rate:.2%}, E[r]={expected_return:+.4%} — "
                f"insufficient directional conviction ({n} matches)."
            )
        return signal, reason

    def __repr__(self) -> str:
        return (
            f"ForecastingEngine("
            f"clusters={len(self.cluster_map)}, "
            f"win_rate_thresh={self.win_rate_threshold}, "
            f"er_thresh={self.expected_return_threshold}, "
            f"min_samples={self.min_cluster_samples})"
        )
