import numpy as np
import pandas as pd

from pairs_trading.backtest.engine import run_backtest
from pairs_trading.config import ExecutionConfig, SignalConfig
from pairs_trading.data.validation import validate_market_data
from pairs_trading.research.cointegration import analyze_pair


def market_fixture(test_residuals: list[float]):
    rng = np.random.default_rng(19)
    train_size = 100
    residual = np.concatenate(
        [rng.normal(0, 0.001, train_size), np.asarray(test_residuals)]
    )
    log_b = np.cumsum(rng.normal(0, 0.01, len(residual))) + 4
    log_a = 0.2 + 0.7 * log_b + residual
    close_a, close_b = np.exp(log_a), np.exp(log_b)
    timestamps = pd.date_range("2022-01-03", periods=len(residual), freq="B")
    market = validate_market_data(
        pd.DataFrame(
            {
                "timestamp": np.repeat(timestamps, 2),
                "symbol": ["A", "B"] * len(timestamps),
                "adjusted_close": np.column_stack((close_a, close_b)).reshape(-1),
                "open": np.column_stack((close_a, close_b)).reshape(-1),
            }
        ),
        min_observations=50,
    )
    training = market.adjusted_close.iloc[:train_size]
    pair = analyze_pair(training, "A", "B", min_observations=50)
    return market, pair, timestamps[train_size]


def test_out_of_sample_engine_executes_next_open_and_applies_costs() -> None:
    residuals = [
        0.0,
        0.0002,
        0.0003,
        0.0001,
        0.02,
        0.015,
        0.009,
        0.0005,
        0.0,
        0.0001,
        0.0002,
    ]
    market, pair, start = market_fixture(residuals)
    signals = SignalConfig(
        zscore_window_days=5,
        entry_threshold=1.5,
        exit_threshold=0.2,
        stop_threshold=3.0,
        max_holding_days=10,
    )
    execution = ExecutionConfig(
        commission_bps_per_side=2,
        slippage_bps_per_side=5,
        initial_capital=10_000,
    )

    result = run_backtest(market, pair, signals, execution, start_date=start)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side.value == "short_spread"
    assert trade.exit_reason == "mean_reversion"
    assert trade.entry_timestamp > start
    assert trade.total_costs > 0
    assert len(result.fills) == 4
    assert result.equity_curve.index.is_monotonic_increasing
    assert (result.equity_curve["net_equity"] <= result.equity_curve["gross_equity"]).all()
    assert (
        result.equity_curve["net_equity"].iloc[-1]
        < result.equity_curve["gross_equity"].iloc[-1]
    )


def test_open_position_is_liquidated_at_end_of_data() -> None:
    market, pair, start = market_fixture([0.0, 0.0002, 0.0003, 0.0001, 0.02, 0.015])
    signals = SignalConfig(
        zscore_window_days=5,
        entry_threshold=1.5,
        exit_threshold=0.1,
        stop_threshold=4.0,
        max_holding_days=10,
    )

    result = run_backtest(
        market,
        pair,
        signals,
        ExecutionConfig(initial_capital=10_000),
        start_date=start,
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "end_of_data"
    assert result.trades[0].exit_timestamp == market.adjusted_close.index[-1]
    assert not result.cancelled_orders
