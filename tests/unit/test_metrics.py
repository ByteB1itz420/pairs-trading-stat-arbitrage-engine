import pandas as pd
import pytest

from pairs_trading.backtest.engine import TradeRecord
from pairs_trading.backtest.events import Fill
from pairs_trading.reporting.metrics import calculate_metrics
from pairs_trading.signals.zscore import PositionSide


def test_metrics_calculate_returns_drawdown_trade_statistics_and_costs() -> None:
    dates = pd.date_range("2025-01-01", periods=4, freq="B")
    equity = pd.DataFrame(
        {
            "net_equity": [1_000.0, 1_100.0, 990.0, 1_089.0],
            "gross_equity": [1_000.0, 1_110.0, 1_000.0, 1_100.0],
            "gross_exposure": [0.0, 500.0, 500.0, 0.0],
            "borrow_costs": [0.0, 0.0, 1.0, 1.0],
        },
        index=dates,
    )
    trades = [
        TradeRecord(
            entry_timestamp=dates[1],
            exit_timestamp=dates[3],
            side=PositionSide.LONG_SPREAD,
            net_pnl=89.0,
            total_costs=11.0,
            holding_bars=2,
            exit_reason="mean_reversion",
        )
    ]
    fills = [
        Fill(dates[1], "A", 1.0, 100.0, 100.1, 0.1, 0.1, "entry"),
        Fill(dates[3], "A", -1.0, 109.0, 108.9, 0.1, 0.1, "exit"),
    ]

    metrics = calculate_metrics(equity, trades, fills, initial_capital=1_000)

    assert metrics["cumulative_net_return"] == pytest.approx(0.089)
    assert metrics["cumulative_gross_return"] == pytest.approx(0.1)
    assert metrics["maximum_drawdown"] == pytest.approx(-0.1)
    assert metrics["trade_count"] == 1
    assert metrics["win_rate"] == 1.0
    assert metrics["average_holding_bars"] == 2.0
    assert metrics["total_commissions"] == pytest.approx(0.2)
    assert metrics["total_slippage_costs"] == pytest.approx(0.2)
    assert metrics["total_borrow_costs"] == 1.0
    assert metrics["time_in_market"] == 0.5


def test_metrics_reject_empty_or_non_finite_equity() -> None:
    columns = ["net_equity", "gross_equity", "gross_exposure"]
    empty = pd.DataFrame(columns=columns)
    with pytest.raises(ValueError, match="empty equity"):
        calculate_metrics(empty, [], [], initial_capital=1_000)

    invalid = pd.DataFrame(
        {"net_equity": [float("inf")], "gross_equity": [1_000], "gross_exposure": [0]}
    )
    with pytest.raises(ValueError, match="non-finite"):
        calculate_metrics(invalid, [], [], initial_capital=1_000)
