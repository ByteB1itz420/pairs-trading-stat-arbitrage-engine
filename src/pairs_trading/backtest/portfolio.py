"""Cash, position, transaction-cost, and mark-to-market accounting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from pairs_trading.backtest.events import Fill


@dataclass(frozen=True)
class Valuation:
    timestamp: pd.Timestamp
    cash: float
    net_equity: float
    gross_equity: float
    gross_exposure: float
    net_exposure: float
    commissions: float
    slippage_costs: float
    borrow_costs: float


class PortfolioLedger:
    """Single-pair cash ledger with explicit marking and no synthetic price fills."""

    def __init__(self, initial_capital: float) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, float] = {}
        self.last_prices: dict[str, float] = {}
        self.fills: list[Fill] = []
        self.commission_total = 0.0
        self.slippage_total = 0.0
        self.borrow_total = 0.0

    @property
    def total_costs(self) -> float:
        return self.commission_total + self.slippage_total + self.borrow_total

    def position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def apply_fill(self, fill: Fill) -> None:
        if not math.isfinite(fill.quantity) or not math.isfinite(fill.execution_price):
            raise ValueError("Fill quantity and execution price must be finite.")
        if fill.execution_price <= 0 or fill.commission < 0 or fill.slippage_cost < 0:
            raise ValueError("Fill prices must be positive and costs cannot be negative.")
        self.cash -= fill.quantity * fill.execution_price + fill.commission
        new_position = self.position(fill.symbol) + fill.quantity
        if abs(new_position) < 1e-10:
            self.positions.pop(fill.symbol, None)
        else:
            self.positions[fill.symbol] = new_position
        self.commission_total += fill.commission
        self.slippage_total += fill.slippage_cost
        self.fills.append(fill)

    def accrue_borrow(
        self,
        previous_close: dict[str, float] | None,
        annual_borrow_rate: float,
        annualization_days: int = 252,
    ) -> float:
        if annual_borrow_rate < 0 or annualization_days < 1:
            raise ValueError("Borrow rate cannot be negative and annualization must be positive.")
        if annual_borrow_rate == 0 or not self.positions:
            return 0.0
        if previous_close is None:
            raise ValueError("Previous close is required to accrue borrow costs.")
        daily_cost = 0.0
        for symbol, quantity in self.positions.items():
            if quantity >= 0:
                continue
            price = previous_close.get(symbol, float("nan"))
            if not math.isfinite(price) or price <= 0:
                raise ValueError(f"Cannot accrue borrow for {symbol}: missing previous close.")
            daily_cost += abs(quantity) * price * annual_borrow_rate / annualization_days
        self.cash -= daily_cost
        self.borrow_total += daily_cost
        return daily_cost

    def mark_to_market(
        self,
        timestamp: pd.Timestamp,
        close_prices: dict[str, float],
    ) -> Valuation:
        gross_exposure = 0.0
        net_exposure = 0.0
        for symbol, quantity in self.positions.items():
            price = close_prices.get(symbol, float("nan"))
            if not math.isfinite(price) or price <= 0:
                raise ValueError(
                    f"Cannot mark open position in {symbol} at {timestamp}: "
                    "missing or invalid close."
                )
            self.last_prices[symbol] = price
            notional = quantity * price
            gross_exposure += abs(notional)
            net_exposure += notional
        net_equity = self.cash + sum(
            quantity * self.last_prices[symbol]
            for symbol, quantity in self.positions.items()
        )
        return Valuation(
            timestamp=timestamp,
            cash=self.cash,
            net_equity=net_equity,
            gross_equity=net_equity + self.total_costs,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            commissions=self.commission_total,
            slippage_costs=self.slippage_total,
            borrow_costs=self.borrow_total,
        )
