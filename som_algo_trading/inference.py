"""
inference.py  —  Phase 3: Live Inference & Risk Management
-----------------------------------------------------------
Responsibilities
    1. Accept a live price series, extract and scale the current window.
    2. Find its BMU on the trained SOM.
    3. Compute the Quantization Error (QE).
    4. Apply the anomaly (risk) gate — halt if QE exceeds the threshold.
    5. Return the BMU coordinate to Phase 4 if the gate passes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .data import DataEngineer
from .som import SOMCore

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Result container                                                    #
# ------------------------------------------------------------------ #

@dataclass
class InferenceResult:
    """
    Output of :meth:`LiveInference.run`.

    Attributes
    ----------
    status : str
        ``"OK"`` if QE is within limits, ``"HALTED"`` otherwise.
    bmu : tuple or None
        Best Matching Unit coordinate (row, col).  None when halted.
    quantization_error : float
        Euclidean distance of the live vector to its BMU weight.
    anomaly_threshold : float
        Threshold used for the risk gate.
    live_vector : np.ndarray
        The scaled live window passed to the SOM.
    message : str
        Human-readable status message.
    """
    status: str
    bmu: Optional[Tuple[int, int]]
    quantization_error: float
    anomaly_threshold: float
    live_vector: np.ndarray
    message: str

    @property
    def is_ok(self) -> bool:
        return self.status == "OK"

    def __repr__(self) -> str:
        return (
            f"InferenceResult(status={self.status!r}, bmu={self.bmu}, "
            f"QE={self.quantization_error:.4f}, threshold={self.anomaly_threshold:.4f})"
        )


# ------------------------------------------------------------------ #
#  Main class                                                          #
# ------------------------------------------------------------------ #

class LiveInference:
    """
    Phase-3 live inference and risk management module.

    Parameters
    ----------
    data_engineer : DataEngineer
        Fitted Phase-1 object (carries the locked StandardScaler).
    som_core : SOMCore
        Trained Phase-2 SOM with a built cluster_map.
    anomaly_threshold : float
        Maximum acceptable quantization error.  Typically the 95th
        percentile of training QEs (see :meth:`set_threshold_from_training`).

    Examples
    --------
    >>> li = LiveInference(data_engineer, som_core, anomaly_threshold=0.85)
    >>> result = li.run(live_prices)
    >>> if result.is_ok:
    ...     signal = forecaster.predict(result.bmu)
    """

    def __init__(
        self,
        data_engineer: DataEngineer,
        som_core: SOMCore,
        anomaly_threshold: float,
    ) -> None:
        if not data_engineer.is_fitted:
            raise RuntimeError("data_engineer must be fitted before use.")
        if not som_core.is_trained:
            raise RuntimeError("som_core must be trained before use.")

        self.data_engineer = data_engineer
        self.som_core = som_core
        self.anomaly_threshold = anomaly_threshold

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(self, live_prices: np.ndarray) -> InferenceResult:
        """
        Execute the full Phase-3 pipeline on *live_prices*.

        Parameters
        ----------
        live_prices : np.ndarray
            Recent closing prices.  Must contain at least
            ``window_size + 1`` values so one return-window can be formed.

        Returns
        -------
        InferenceResult
        """
        # Step 1 & 2: extract + scale live window
        live_vec = self.data_engineer.transform_live(live_prices)   # (1, ws)
        flat_vec = live_vec.flatten()

        # Step 3: BMU
        bmu = self.som_core.get_bmu(flat_vec)

        # Step 4: QE
        qe = float(self.som_core.quantization_error(flat_vec))

        # Step 5: Risk gate
        if qe > self.anomaly_threshold:
            msg = (
                f"SYSTEM HALTED: Unprecedented Market Regime (High QE). "
                f"QE={qe:.4f} > threshold={self.anomaly_threshold:.4f}"
            )
            logger.warning(msg)
            return InferenceResult(
                status="HALTED",
                bmu=None,
                quantization_error=qe,
                anomaly_threshold=self.anomaly_threshold,
                live_vector=flat_vec,
                message=msg,
            )

        msg = (
            f"BMU={bmu}, QE={qe:.4f} (threshold={self.anomaly_threshold:.4f}). "
            "Proceeding to forecast."
        )
        logger.info(msg)
        return InferenceResult(
            status="OK",
            bmu=bmu,
            quantization_error=qe,
            anomaly_threshold=self.anomaly_threshold,
            live_vector=flat_vec,
            message=msg,
        )

    @classmethod
    def set_threshold_from_training(
        cls,
        data_engineer: DataEngineer,
        som_core: SOMCore,
        percentile: float = 95.0,
    ) -> float:
        """
        Auto-calculate the anomaly threshold from the distribution of
        training quantization errors.

        Parameters
        ----------
        data_engineer : DataEngineer
        som_core : SOMCore
            Must have ``training_qes`` populated (i.e., after
            :meth:`SOMCore.build_cluster_map`).
        percentile : float
            Which percentile of training QEs to use as the threshold.
            Default 95.0 (reject the worst 5 % of regimes seen historically).

        Returns
        -------
        float
            The computed threshold value.
        """
        if som_core.training_qes is None:
            raise RuntimeError(
                "som_core.training_qes is None. "
                "Call SOMCore.build_cluster_map() before auto-thresholding."
            )
        threshold = float(np.percentile(som_core.training_qes, percentile))
        logger.info(
            "Anomaly threshold set to %.4f (%.0fth percentile of training QEs).",
            threshold, percentile,
        )
        return threshold

    def __repr__(self) -> str:
        return (
            f"LiveInference(window_size={self.data_engineer.window_size}, "
            f"anomaly_threshold={self.anomaly_threshold:.4f})"
        )
