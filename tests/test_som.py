"""Unit tests for SOMCore (Phase 2)."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from som_algo_trading.som import SOMCore
from som_algo_trading.data import DataEngineer
from som_algo_trading.utils import generate_synthetic_prices


@pytest.fixture(scope="module")
def fitted_som():
    prices = generate_synthetic_prices(n_days=600, random_state=1)
    de = DataEngineer(window_size=20)
    X_train, y_train, _, _ = de.fit_transform(prices)
    core = SOMCore(grid_rows=5, grid_cols=5, window_size=20, num_iterations=500, random_state=1)
    core.fit(X_train)
    core.build_cluster_map(X_train)
    return core, X_train


class TestSOMCore:

    def test_trained_flag(self, fitted_som):
        core, _ = fitted_som
        assert core.is_trained

    def test_cluster_map_keys_are_tuples(self, fitted_som):
        core, _ = fitted_som
        for key in core.cluster_map:
            assert isinstance(key, tuple) and len(key) == 2

    def test_cluster_map_all_indices_present(self, fitted_som):
        core, X_train = fitted_som
        all_idx = []
        for v in core.cluster_map.values():
            all_idx.extend(v)
        assert sorted(all_idx) == list(range(len(X_train)))

    def test_bmu_in_grid(self, fitted_som):
        core, X_train = fitted_som
        bmu = core.get_bmu(X_train[0])
        assert 0 <= bmu[0] < core.grid_rows
        assert 0 <= bmu[1] < core.grid_cols

    def test_qe_non_negative(self, fitted_som):
        core, X_train = fitted_som
        qe = core.quantization_error(X_train[0])
        assert qe >= 0

    def test_training_qes_shape(self, fitted_som):
        core, X_train = fitted_som
        assert core.training_qes is not None
        assert len(core.training_qes) == len(X_train)

    def test_u_matrix_shape(self, fitted_som):
        core, _ = fitted_som
        u = core.u_matrix()
        assert u.shape == (core.grid_rows, core.grid_cols)

    def test_save_load(self, fitted_som, tmp_path):
        core, _ = fitted_som
        path = tmp_path / "som.pkl"
        core.save(path)
        loaded = SOMCore.load(path)
        assert loaded.is_trained
        assert loaded.cluster_map.keys() == core.cluster_map.keys()

    def test_untrained_raises(self):
        core = SOMCore(grid_rows=3, grid_cols=3, window_size=5)
        with pytest.raises(RuntimeError):
            core.get_bmu(np.zeros(5))
