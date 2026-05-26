"""Unit tests for DataEngineer (Phase 1)."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from som_algo_trading.data import DataEngineer
from som_algo_trading.utils import generate_synthetic_prices


@pytest.fixture
def prices():
    return generate_synthetic_prices(n_days=500, random_state=0)


class TestDataEngineer:

    def test_returns_length(self, prices):
        de = DataEngineer(window_size=30)
        returns = de.get_returns(prices)
        assert len(returns) == len(prices) - 1

    def test_returns_finite(self, prices):
        de = DataEngineer(window_size=30)
        returns = de.get_returns(prices)
        assert np.all(np.isfinite(returns))

    def test_fit_transform_shapes(self, prices):
        de = DataEngineer(window_size=30)
        X_train, y_train, X_test, y_test = de.fit_transform(prices, train_ratio=0.8)
        assert X_train.ndim == 2
        assert X_train.shape[1] == 30
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_train_test_split_ratio(self, prices):
        de = DataEngineer(window_size=30)
        X_tr, y_tr, X_te, y_te = de.fit_transform(prices, train_ratio=0.8)
        total = len(X_tr) + len(X_te)
        assert abs(len(X_tr) / total - 0.8) < 0.05

    def test_scaler_fitted(self, prices):
        de = DataEngineer(window_size=30)
        de.fit_transform(prices)
        assert de.is_fitted
        assert de.scaler is not None

    def test_train_data_standardised(self, prices):
        de = DataEngineer(window_size=30)
        X_tr, *_ = de.fit_transform(prices)
        assert np.allclose(X_tr.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(X_tr.std(axis=0), 1, atol=1e-3)

    def test_transform_live(self, prices):
        de = DataEngineer(window_size=30)
        de.fit_transform(prices)
        live = de.transform_live(prices[-40:])
        assert live.shape == (1, 30)

    def test_unfitted_raises(self, prices):
        de = DataEngineer(window_size=30)
        with pytest.raises(RuntimeError):
            de.transform_live(prices[-40:])

    def test_invalid_prices(self):
        de = DataEngineer(window_size=5)
        with pytest.raises(ValueError):
            de.fit_transform(np.array([-1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))

    def test_scaler_save_load(self, tmp_path, prices):
        de = DataEngineer(window_size=30)
        de.fit_transform(prices)
        path = tmp_path / "scaler.pkl"
        de.save_scaler(path)
        de2 = DataEngineer(window_size=30)
        de2.load_scaler(path)
        assert de2.is_fitted
        live = de2.transform_live(prices[-40:])
        assert live.shape == (1, 30)
