"""Typed events for the chronological market/execution/valuation loop."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import count
from typing import TypeAlias

import pandas as pd

from pairs_trading.signals.zscore import PositionSide


@dataclass(frozen=True)
class OrderIntent:
    symbol_a: str
    symbol_b: str
    target_side: PositionSide
    decision_timestamp: pd.Timestamp
    reason: str
    zscore: float | None


@dataclass(frozen=True)
class Fill:
    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    reference_price: float
    execution_price: float
    commission: float
    slippage_cost: float
    reason: str


@dataclass(frozen=True)
class MarketOpenEvent:
    timestamp: pd.Timestamp
    prices: dict[str, float]


@dataclass(frozen=True)
class ExecuteOrderEvent:
    timestamp: pd.Timestamp
    order: OrderIntent
    reference_prices: dict[str, float]


@dataclass(frozen=True)
class MarketCloseEvent:
    timestamp: pd.Timestamp
    prices: dict[str, float]


@dataclass(frozen=True)
class StrategyEvent:
    timestamp: pd.Timestamp


@dataclass(frozen=True)
class EndOfDataEvent:
    timestamp: pd.Timestamp
    close_prices: dict[str, float]


BacktestEvent: TypeAlias = (
    MarketOpenEvent
    | ExecuteOrderEvent
    | MarketCloseEvent
    | StrategyEvent
    | EndOfDataEvent
)

_PRIORITY = {
    MarketOpenEvent: 0,
    ExecuteOrderEvent: 1,
    MarketCloseEvent: 2,
    StrategyEvent: 3,
    EndOfDataEvent: 4,
}


class EventQueue:
    """Stable priority queue; same-time events follow the trading-session phases."""

    def __init__(self) -> None:
        self._events: list[tuple[pd.Timestamp, int, int, BacktestEvent]] = []
        self._sequence = count()

    def push(self, event: BacktestEvent) -> None:
        heapq.heappush(
            self._events,
            (event.timestamp, _PRIORITY[type(event)], next(self._sequence), event),
        )

    def pop(self) -> BacktestEvent:
        return heapq.heappop(self._events)[3]

    def __bool__(self) -> bool:
        return bool(self._events)
