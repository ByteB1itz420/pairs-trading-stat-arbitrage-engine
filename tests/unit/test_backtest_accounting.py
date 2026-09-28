import pandas as pd
import pytest

from pairs_trading.backtest.events import Fill, OrderIntent
from pairs_trading.backtest.execution import ExecutionModel
from pairs_trading.backtest.portfolio import PortfolioLedger
from pairs_trading.config import ExecutionConfig
from pairs_trading.research.cointegration import PairAnalysis
from pairs_trading.signals.zscore import PositionSide


def test_pair_sizing_respects_gross_notional_and_adverse_execution_costs() -> None:
    config = ExecutionConfig(
        initial_capital=10_000,
        gross_exposure_fraction=1.0,
        commission_bps_per_side=10,
        slippage_bps_per_side=20,
    )
    ledger = PortfolioLedger(config.initial_capital)
    pair = PairAnalysis("A", "B", 100, intercept=0.0, hedge_ratio=0.5)
    order = OrderIntent(
        "A",
        "B",
        PositionSide.LONG_SPREAD,
        pd.Timestamp("2025-01-01"),
        "test",
        -2.0,
        2.0,
    )

    fills = ExecutionModel(config).create_fills(
        order,
        pd.Timestamp("2025-01-02"),
        {"A": 100.0, "B": 50.0},
        pair,
        ledger,
    )

    assert [fill.symbol for fill in fills] == ["A", "B"]
    assert fills[0].quantity == pytest.approx(10_000 / 3 / 100)
    assert fills[1].quantity == pytest.approx(-2 * 10_000 / 3 / 50)
    assert fills[0].execution_price > fills[0].reference_price
    assert fills[1].execution_price < fills[1].reference_price
    gross = sum(abs(fill.quantity * fill.reference_price) for fill in fills)
    assert gross == pytest.approx(10_000)
    assert all(fill.commission > 0 and fill.slippage_cost > 0 for fill in fills)


def test_cash_positions_and_borrow_costs_reconcile() -> None:
    ledger = PortfolioLedger(10_000)
    ledger.apply_fill(
        # Selling short produces positive cash and a negative position.
        Fill(
            pd.Timestamp("2025-01-02"),
            "A",
            -10,
            100,
            100,
            0,
            0,
            "test",
        )
    )

    assert ledger.cash == 11_000
    assert ledger.position("A") == -10
    assert ledger.accrue_borrow({"A": 100}, annual_borrow_rate=0.252) == pytest.approx(1.0)
    valuation = ledger.mark_to_market(pd.Timestamp("2025-01-02"), {"A": 105})

    assert valuation.net_equity == pytest.approx(9_949)
    assert valuation.gross_exposure == pytest.approx(1_050)
    assert valuation.borrow_costs == pytest.approx(1.0)


def test_missing_open_for_active_leg_fails_before_partial_execution() -> None:
    config = ExecutionConfig(initial_capital=10_000)
    ledger = PortfolioLedger(config.initial_capital)
    order = OrderIntent(
        "A",
        "B",
        PositionSide.LONG_SPREAD,
        pd.Timestamp("2025-01-01"),
        "test",
        -2.0,
        1.0,
    )

    with pytest.raises(ValueError, match="missing or invalid open for B"):
        ExecutionModel(config).create_fills(
            order,
            pd.Timestamp("2025-01-02"),
            {"A": 100.0, "B": float("nan")},
            PairAnalysis("A", "B", 100, intercept=0.0, hedge_ratio=1.0),
            ledger,
        )
    assert ledger.positions == {}
