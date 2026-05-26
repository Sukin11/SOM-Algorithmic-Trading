"""Integration tests for SOMTradingPipeline."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from som_algo_trading.pipeline import SOMTradingPipeline
from som_algo_trading.config import TradingConfig
from som_algo_trading.forecaster import Signal
from som_algo_trading.utils import generate_synthetic_prices, generate_regime_prices


@pytest.fixture(scope="module")
def fitted_pipeline():
    prices = generate_regime_prices(n_days=800, random_state=42)
    cfg = TradingConfig(
        window_size=20,
        grid_rows=5,
        grid_cols=5,
        num_iterations=300,
        min_cluster_samples=5,   # small for test speed
        random_state=42,
    )
    pipe = SOMTradingPipeline(cfg)
    pipe.fit(prices)
    return pipe, prices


class TestSOMTradingPipeline:

    def test_is_fitted(self, fitted_pipeline):
        pipe, _ = fitted_pipeline
        assert pipe.is_fitted

    def test_predict_returns_forecast_result(self, fitted_pipeline):
        pipe, prices = fitted_pipeline
        result = pipe.predict(prices[-30:])
        assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)

    def test_predict_signal_is_string(self, fitted_pipeline):
        pipe, prices = fitted_pipeline
        result = pipe.predict(prices[-30:])
        assert isinstance(result.signal, str)

    def test_backtest_keys(self, fitted_pipeline):
        pipe, _ = fitted_pipeline
        bt = pipe.backtest()
        for key in ("signals", "actual_returns", "accuracy",
                    "buy_win_rate", "sell_win_rate", "n_buy", "n_sell",
                    "n_hold", "n_halted"):
            assert key in bt

    def test_backtest_signal_counts_add_up(self, fitted_pipeline):
        pipe, _ = fitted_pipeline
        bt = pipe.backtest()
        total = bt["n_buy"] + bt["n_sell"] + bt["n_hold"]
        assert total == len(bt["signals"])

    def test_save_load(self, fitted_pipeline, tmp_path):
        pipe, prices = fitted_pipeline
        pipe.save(tmp_path / "model")
        loaded = SOMTradingPipeline.load(tmp_path / "model")
        assert loaded.is_fitted
        result = loaded.predict(prices[-30:])
        assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)

    def test_unfitted_predict_raises(self):
        pipe = SOMTradingPipeline()
        with pytest.raises(RuntimeError):
            pipe.predict(np.random.rand(50))


class TestTradingConfig:

    def test_recommend_grid(self):
        rows, cols = TradingConfig.recommend_grid(n_windows=500)
        assert rows >= 2 and cols >= 2

    def test_validate_bad_window(self):
        cfg = TradingConfig(window_size=1)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_bad_lr(self):
        cfg = TradingConfig(learning_rate=1.5)
        with pytest.raises(ValueError):
            cfg.validate()
