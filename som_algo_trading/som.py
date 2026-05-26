"""
som.py  —  Phase 2: SOM Core (Training & Clustering)
------------------------------------------------------
Implements a minisom-backed Self-Organizing Map with:
    * PCA-based weight initialisation for fast convergence.
    * Training on the standardised window matrix from Phase 1.
    * Historical mapping: every training window → its BMU coordinate.
    * cluster_map dict: {(row, col): [list of window indices]}.

Dependencies
------------
    minisom >= 2.3        pip install minisom
"""

from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from minisom import MiniSom

logger = logging.getLogger(__name__)


class SOMCore:
    """
    Phase-2 SOM: training and historical-regime clustering.

    Parameters
    ----------
    grid_rows : int
    grid_cols : int
    window_size : int
        Input vector dimensionality (must match DataEngineer.window_size).
    sigma : float
        Initial neighbourhood radius.
    learning_rate : float
        Initial learning rate.
    num_iterations : int
        Training epochs.
    random_state : int or None

    Attributes
    ----------
    som : minisom.MiniSom
        The trained SOM object.
    cluster_map : dict
        ``{(row, col): [idx, idx, ...]}``.  Populated after :meth:`build_cluster_map`.
    training_qes : np.ndarray
        Quantization errors of every training window (set after mapping).
    is_trained : bool

    Examples
    --------
    >>> core = SOMCore(grid_rows=10, grid_cols=10, window_size=30)
    >>> core.fit(X_train)
    >>> core.build_cluster_map(X_train)
    >>> bmu = core.get_bmu(live_vector)
    """

    def __init__(
        self,
        grid_rows: int = 10,
        grid_cols: int = 10,
        window_size: int = 30,
        sigma: float = 3.0,
        learning_rate: float = 0.5,
        num_iterations: int = 10_000,
        random_state: Optional[int] = 42,
    ) -> None:
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.window_size = window_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.random_state = random_state

        self.som: Optional[MiniSom] = None
        self.cluster_map: Dict[Tuple[int, int], List[int]] = {}
        self.training_qes: Optional[np.ndarray] = None
        self.is_trained: bool = False

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "SOMCore":
        """
        Initialise (PCA) and train the SOM on *X*.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, window_size)
            Standardised training windows from Phase 1.

        Returns
        -------
        self
        """
        X = self._validate_X(X)
        seed = self.random_state if self.random_state is not None else 0

        self.som = MiniSom(
            x=self.grid_rows,
            y=self.grid_cols,
            input_len=self.window_size,
            sigma=self.sigma,
            learning_rate=self.learning_rate,
            neighborhood_function="gaussian",
            random_seed=seed,
        )

        logger.info(
            "Initialising SOM weights via PCA (%dx%d, sigma=%.2f, lr=%.3f) ...",
            self.grid_rows, self.grid_cols, self.sigma, self.learning_rate,
        )
        self.som.pca_weights_init(X)

        logger.info("Training SOM for %d iterations ...", self.num_iterations)
        self.som.train_batch(X, self.num_iterations, verbose=False)

        self.is_trained = True
        logger.info("SOM training complete.")
        return self

    def build_cluster_map(self, X: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
        """
        Map every training window to its BMU and build the cluster look-up.

        Also records per-window quantization errors in ``self.training_qes``.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, window_size)
            The same standardised training windows used to train the SOM.

        Returns
        -------
        cluster_map : dict
            ``{(row, col): [window_index, ...]}``.
        """
        self._check_trained()
        X = self._validate_X(X)

        cluster_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        qes = np.empty(len(X))

        for idx, vec in enumerate(X):
            bmu = self.som.winner(vec)           # (row, col)
            cluster_map[bmu].append(idx)
            qes[idx] = self._quantization_error(vec, bmu)

        self.cluster_map = dict(cluster_map)
        self.training_qes = qes

        n_clusters_used = len(self.cluster_map)
        total_nodes = self.grid_rows * self.grid_cols
        logger.info(
            "Cluster map built | %d windows → %d / %d nodes used",
            len(X), n_clusters_used, total_nodes,
        )
        return self.cluster_map

    def get_bmu(self, vec: np.ndarray) -> Tuple[int, int]:
        """
        Return the Best Matching Unit coordinate for a single vector.

        Parameters
        ----------
        vec : np.ndarray, shape (window_size,) or (1, window_size)

        Returns
        -------
        (row, col) tuple
        """
        self._check_trained()
        vec = np.asarray(vec, dtype=float).flatten()
        return self.som.winner(vec)

    def quantization_error(self, vec: np.ndarray) -> float:
        """
        Euclidean distance between *vec* and its BMU weight vector.

        Parameters
        ----------
        vec : np.ndarray, shape (window_size,) or (1, window_size)

        Returns
        -------
        float
        """
        self._check_trained()
        vec = np.asarray(vec, dtype=float).flatten()
        bmu = self.som.winner(vec)
        return self._quantization_error(vec, bmu)

    def get_weight_vector(self, bmu: Tuple[int, int]) -> np.ndarray:
        """Return the weight vector of a given BMU node."""
        self._check_trained()
        return self.som.get_weights()[bmu[0], bmu[1]]

    def u_matrix(self) -> np.ndarray:
        """
        Return the Unified Distance Matrix (U-matrix) for visualisation.

        Shape: (grid_rows, grid_cols).
        """
        self._check_trained()
        return self.som.distance_map()

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Pickle the entire SOMCore (including the trained MiniSom)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("SOMCore saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SOMCore":
        """Load a previously saved SOMCore from *path*."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SOMCore file not found: {path}")
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected SOMCore, got {type(obj)}")
        logger.info("SOMCore loaded ← %s", path)
        return obj

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _quantization_error(
        self, vec: np.ndarray, bmu: Tuple[int, int]
    ) -> float:
        weight = self.som.get_weights()[bmu[0], bmu[1]]
        return float(np.linalg.norm(vec - weight))

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.ndim}-D.")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or Inf values.")
        return X

    def _check_trained(self):
        if not self.is_trained or self.som is None:
            raise RuntimeError(
                "SOMCore is not trained. Call fit() first."
            )

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        n_clusters = len(self.cluster_map) if self.cluster_map else 0
        return (
            f"SOMCore(grid={self.grid_rows}x{self.grid_cols}, "
            f"window_size={self.window_size}, status={status}, "
            f"clusters_occupied={n_clusters})"
        )
