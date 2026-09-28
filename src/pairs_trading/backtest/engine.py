"""Event-driven single-pair backtest with next-open execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

import pandas as pd

from pairs_trading.backtest.events import (
    BacktestEvent,
    EndOfDataEvent,
    EventQueue,
    ExecuteOrderEvent,
    Fill,
    MarketCloseEvent,
    MarketOpenEvent,
    OrderIntent,
    StrategyEvent,
)
from pairs_trading.backtest.execution import ExecutionModel
from pairs_trading.backtest.portfolio import PortfolioLedger, Valuation
from pairs_trading.config import ExecutionConfig, SignalConfig
from pairs_trading.data.validation import MarketData
from pairs_trading.research.cointegration import PairAnalysis
from pairs_trading.research.spread import calculate_spread, rolling_zscore
from pairs_trading.signals.zscore import (
    PositionSide,
    Signal,
    SignalAction,
    ZScoreStrategy,
)


@dataclass(frozen=True)
class TradeRecord:
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    side: PositionSide
    net_pnl: float
    total_costs: float
    holding_bars: int
    exit_reason: str


@dataclass(frozen=True)
class CancelledOrder:
    order: OrderIntent
    reason: str


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    signals: tuple[Signal, ...]
    fills: tuple[Fill, ...]
    trades: tuple[TradeRecord, ...]
    cancelled_orders: tuple[CancelledOrder, ...]


class BacktestEngine:
    """Run one frozen pair model over a declared evaluation date range."""

    def __init__(
        self,
        market: MarketData,
        pair: PairAnalysis,
        signal_config: SignalConfig,
        execution_config: ExecutionConfig,
        start_date: str | date | pd.Timestamp | None = None,
        end_date: str | date | pd.Timestamp | None = None,
    ) -> None:
        if market.open is None:
            raise ValueError("Next-open execution requires open prices in the market data.")
        if pair.intercept is None or pair.hedge_ratio is None:
            raise ValueError("Backtesting requires a fitted pair model.")
        if pair.symbol_a not in market.adjusted_close or pair.symbol_b not in market.adjusted_close:
            raise ValueError("Market data does not contain both fitted pair symbols.")
        all_dates = market.adjusted_close.index
        start = pd.Timestamp(start_date) if start_date is not None else all_dates.min()
        end = pd.Timestamp(end_date) if end_date is not None else all_dates.max()
        if start > end:
            raise ValueError("start_date must not be after end_date.")
        self.market = market
        self.pair = pair
        self.signal_config = signal_config
        self.execution_config = execution_config
        self.dates = all_dates[(all_dates >= start) & (all_dates <= end)]
        if len(self.dates) == 0:
            raise ValueError("The requested backtest date range has no observations.")

    def run(self) -> BacktestResult:
        spread = calculate_spread(self.market.adjusted_close, self.pair)
        zscores = rolling_zscore(spread, self.signal_config.zscore_window_days)
        close_prices = self.market.adjusted_close
        open_prices = self.market.open
        if open_prices is None:
            raise ValueError("Next-open execution requires open prices in the market data.")

        queue = EventQueue()
        date_positions = {timestamp: index for index, timestamp in enumerate(self.dates)}
        for timestamp in self.dates:
            queue.push(
                MarketOpenEvent(
                    timestamp,
                    {symbol: float(value) for symbol, value in open_prices.loc[timestamp].items()},
                )
            )
            queue.push(
                MarketCloseEvent(
                    timestamp,
                    {symbol: float(value) for symbol, value in close_prices.loc[timestamp].items()},
                )
            )
            value = zscores.loc[timestamp]
            queue.push(
                StrategyEvent(
                    timestamp,
                    float(value) if pd.notna(value) and math.isfinite(float(value)) else None,
                )
            )
        final_timestamp = self.dates[-1]
        queue.push(
            EndOfDataEvent(
                final_timestamp,
                {
                    symbol: float(value)
                    for symbol, value in close_prices.loc[final_timestamp].items()
                },
            )
        )

        strategy = ZScoreStrategy(self.signal_config)
        execution_model = ExecutionModel(self.execution_config)
        portfolio = PortfolioLedger(self.execution_config.initial_capital)
        previous_close: dict[str, float] | None = None
        signals: list[Signal] = []
        trades: list[TradeRecord] = []
        cancelled_orders: list[CancelledOrder] = []
        valuations: list[Valuation] = []
        trade_start_equity: float | None = None
        trade_start_costs = 0.0
        trade_side = PositionSide.FLAT
        trade_holding_bars = 0
        trade_entry_timestamp: pd.Timestamp | None = None
        pending_exit_reason = "strategy_exit"

        while queue:
            event = queue.pop()
            if isinstance(event, MarketOpenEvent):
                portfolio.accrue_borrow(
                    previous_close,
                    self.execution_config.annual_borrow_rate,
                )
            elif isinstance(event, ExecuteOrderEvent):
                order = event.order
                if order.target_side == PositionSide.FLAT:
                    if trade_start_equity is None or trade_entry_timestamp is None:
                        raise ValueError("Cannot close a pair position without an open trade.")
                    if order.reason not in {"end_of_data", "forced_end_of_data"}:
                        pending_exit_reason = order.reason
                else:
                    if strategy.side != PositionSide.FLAT:
                        raise ValueError("A new pair position was ordered while one is already open.")
                    trade_start_equity = portfolio.mark_to_market(
                        event.timestamp,
                        event.reference_prices,
                    ).net_equity
                    trade_start_costs = portfolio.total_costs
                    trade_side = order.target_side
                    trade_holding_bars = 0

                fills = execution_model.create_fills(
                    order,
                    event.timestamp,
                    event.reference_prices,
                    self.pair,
                    portfolio,
                )
                if not fills:
                    raise ValueError("Pair order produced no executable leg fills.")
                for fill in fills:
                    portfolio.apply_fill(fill)
                strategy.on_fill(order.target_side)

                if order.target_side == PositionSide.FLAT:
                    if trade_start_equity is None:
                        raise ValueError("Trade accounting lost its entry equity.")
                    trades.append(
                        TradeRecord(
                            entry_timestamp=trade_entry_timestamp,
                            exit_timestamp=event.timestamp,
                            side=trade_side,
                            net_pnl=portfolio.cash - trade_start_equity,
                            total_costs=portfolio.total_costs - trade_start_costs,
                            holding_bars=trade_holding_bars,
                            exit_reason=pending_exit_reason,
                        )
                    )
                    trade_start_equity = None
                    trade_entry_timestamp = None
                    trade_side = PositionSide.FLAT
                    trade_holding_bars = 0
            elif isinstance(event, MarketCloseEvent):
                valuation = portfolio.mark_to_market(event.timestamp, event.prices)
                valuations.append(valuation)
                previous_close = event.prices
                if strategy.side != PositionSide.FLAT:
                    trade_holding_bars += 1
            elif isinstance(event, StrategyEvent):
                signal = strategy.on_close(event.timestamp, event.zscore)
                signals.append(signal)
                if signal.action == SignalAction.HOLD:
                    continue
                target_side = {
                    SignalAction.ENTER_LONG: PositionSide.LONG_SPREAD,
                    SignalAction.ENTER_SHORT: PositionSide.SHORT_SPREAD,
                    SignalAction.EXIT: PositionSide.FLAT,
                }[signal.action]
                order = OrderIntent(
                    symbol_a=self.pair.symbol_a,
                    symbol_b=self.pair.symbol_b,
                    target_side=target_side,
                    decision_timestamp=event.timestamp,
                    reason=signal.reason,
                    zscore=signal.zscore,
                )
                current_index = date_positions[event.timestamp]
                if current_index + 1 >= len(self.dates):
                    cancelled_orders.append(
                        CancelledOrder(order, "no next bar available for next-open execution")
                    )
                    continue
                next_timestamp = self.dates[current_index + 1]
                next_open = {
                    symbol: float(value)
                    for symbol, value in open_prices.loc[next_timestamp].items()
                }
                if target_side != PositionSide.FLAT:
                    trade_entry_timestamp = next_timestamp
                    pending_exit_reason = "strategy_exit"
                queue.push(ExecuteOrderEvent(next_timestamp, order, next_open))
            elif isinstance(event, EndOfDataEvent):
                if strategy.side != PositionSide.FLAT:
                    order = OrderIntent(
                        symbol_a=self.pair.symbol_a,
                        symbol_b=self.pair.symbol_b,
                        target_side=PositionSide.FLAT,
                        decision_timestamp=event.timestamp,
                        reason="end_of_data",
                        zscore=None,
                    )
                    fills = execution_model.create_fills(
                        order,
                        event.timestamp,
                        event.close_prices,
                        self.pair,
                        portfolio,
                    )
                    for fill in fills:
                        portfolio.apply_fill(fill)
                    strategy.on_fill(PositionSide.FLAT)
                    if trade_start_equity is None or trade_entry_timestamp is None:
                        raise ValueError("End-of-data liquidation has no matching open trade.")
                    trades.append(
                        TradeRecord(
                            entry_timestamp=trade_entry_timestamp,
                            exit_timestamp=event.timestamp,
                            side=trade_side,
                            net_pnl=portfolio.cash - trade_start_equity,
                            total_costs=portfolio.total_costs - trade_start_costs,
                            holding_bars=trade_holding_bars,
                            exit_reason="end_of_data",
                        )
                    )
                    trade_start_equity = None
                    trade_entry_timestamp = None
                    trade_side = PositionSide.FLAT
                    trade_holding_bars = 0
                    valuations[-1] = portfolio.mark_to_market(event.timestamp, event.close_prices)

        equity_curve = pd.DataFrame(
            [
                {
                    "timestamp": valuation.timestamp,
                    "cash": valuation.cash,
                    "net_equity": valuation.net_equity,
                    "gross_equity": valuation.gross_equity,
                    "gross_exposure": valuation.gross_exposure,
                    "net_exposure": valuation.net_exposure,
                    "commissions": valuation.commissions,
                    "slippage_costs": valuation.slippage_costs,
                    "borrow_costs": valuation.borrow_costs,
                }
                for valuation in valuations
            ]
        ).set_index("timestamp")
        fill_snapshot = tuple(portfolio.fills)
        return BacktestResult(
            equity_curve=equity_curve,
            signals=tuple(signals),
            fills=fill_snapshot,
            trades=tuple(trades),
            cancelled_orders=tuple(cancelled_orders),
        )


def run_backtest(
    market: MarketData,
    pair: PairAnalysis,
    signal_config: SignalConfig,
    execution_config: ExecutionConfig,
    start_date: str | date | pd.Timestamp | None = None,
    end_date: str | date | pd.Timestamp | None = None,
) -> BacktestResult:
    """Public helper for a frozen pair model over a chosen chronological period."""
    return BacktestEngine(
        market,
        pair,
        signal_config,
        execution_config,
        start_date=start_date,
        end_date=end_date,
    ).run()
