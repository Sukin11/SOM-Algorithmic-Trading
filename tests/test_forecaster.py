"""Unit tests for ForecastingEngine (Phase 4)."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from som_algo_trading.forecaster import ForecastingEngine, Signal


def make_engine(seed=0):
    """Build a ForecastingEngine with synthetic cluster data."""
    rng = np.random.default_rng(seed)
    n = 300
    forward_returns = rng.normal(0.001, 0.01, size=n)
    # Put indices 0-49 in cluster (0,0) — strongly bullish
    forward_returns[:50] = np.abs(rng.normal(0.005, 0.005, size=50))
    # Put indices 50-99 in cluster (0,1) — strongly bearish
    forward_returns[50:100] = -np.abs(rng.normal(0.005, 0.005, size=50))

    cluster_map = {
        (0, 0): list(range(50)),
        (0, 1): list(range(50, 100)),
        (1, 0): list(range(100, 120)),   # only 20 matches → HOLD
    }
    engine = ForecastingEngine(
        cluster_map=cluster_map,
        forward_returns=forward_returns,
        min_cluster_samples=30,
        win_rate_threshold=0.60,
        expected_return_threshold=0.001,
    )
    return engine


class TestForecastingEngine:

    def test_buy_signal(self):
        engine = make_engine()
        result = engine.predict((0, 0))
        assert result.signal == Signal.BUY

    def test_sell_signal(self):
        engine = make_engine()
        result = engine.predict((0, 1))
        assert result.signal == Signal.SELL

    def test_hold_insufficient_samples(self):
        engine = make_engine()
        result = engine.predict((1, 0))
        assert result.signal == Signal.HOLD

    def test_hold_unknown_cluster(self):
        engine = make_engine()
        result = engine.predict((9, 9))
        assert result.signal == Signal.HOLD

    def test_result_has_matched_returns(self):
        engine = make_engine()
        result = engine.predict((0, 0))
        assert result.matched_returns is not None
        assert len(result.matched_returns) == 50

    def test_cluster_statistics_keys(self):
        engine = make_engine()
        stats = engine.cluster_statistics()
        for bmu in [(0, 0), (0, 1), (1, 0)]:
            assert bmu in stats
            assert "n" in stats[bmu]
            assert "mean_return" in stats[bmu]
            assert "win_rate" in stats[bmu]
