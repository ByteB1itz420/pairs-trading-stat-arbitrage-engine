"""Pair-level target exposure sizing and adverse-price execution."""

from __future__ import annotations

import math

from pairs_trading.backtest.events import Fill, OrderIntent
from pairs_trading.backtest.portfolio import PortfolioLedger
from pairs_trading.config import ExecutionConfig
from pairs_trading.research.cointegration import PairAnalysis
from pairs_trading.signals.zscore import PositionSide


class ExecutionModel:
    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config

    def create_fills(
        self,
        order: OrderIntent,
        timestamp,
        reference_prices: dict[str, float],
        pair: PairAnalysis,
        portfolio: PortfolioLedger,
    ) -> list[Fill]:
        if pair.hedge_ratio is None:
            raise ValueError("A fitted hedge ratio is required for execution.")
        if (order.symbol_a, order.symbol_b) != (pair.symbol_a, pair.symbol_b):
            raise ValueError("Order symbols do not match the fitted pair.")
        budget = self.config.initial_capital * self.config.gross_exposure_fraction
        side_sign = {
            PositionSide.FLAT: 0.0,
            PositionSide.LONG_SPREAD: 1.0,
            PositionSide.SHORT_SPREAD: -1.0,
        }[order.target_side]
        denominator = 1.0 + abs(pair.hedge_ratio)
        target_notionals = {
            pair.symbol_a: side_sign * budget / denominator,
            pair.symbol_b: -side_sign * pair.hedge_ratio * budget / denominator,
        }

        fills: list[Fill] = []
        for symbol, target_notional in target_notionals.items():
            target_quantity = 0.0
            if target_notional != 0:
                reference_price = reference_prices.get(symbol, float("nan"))
                if not math.isfinite(reference_price) or reference_price <= 0:
                    raise ValueError(
                        f"Cannot execute pair order at {timestamp}: "
                        f"missing or invalid open for {symbol}."
                    )
                target_quantity = target_notional / reference_price
            quantity = target_quantity - portfolio.position(symbol)
            if abs(quantity) < 1e-10:
                continue
            reference_price = reference_prices.get(symbol, float("nan"))
            if not math.isfinite(reference_price) or reference_price <= 0:
                raise ValueError(
                    f"Cannot execute pair order at {timestamp}: missing or invalid open for {symbol}."
                )
            slippage_rate = self.config.slippage_bps_per_side / 10_000
            execution_price = reference_price * (1 + math.copysign(slippage_rate, quantity))
            commission = (
                abs(quantity * execution_price)
                * self.config.commission_bps_per_side
                / 10_000
            )
            slippage_cost = abs(quantity * (execution_price - reference_price))
            fills.append(
                Fill(
                    timestamp=timestamp,
                    symbol=symbol,
                    quantity=quantity,
                    reference_price=reference_price,
                    execution_price=execution_price,
                    commission=commission,
                    slippage_cost=slippage_cost,
                    reason=order.reason,
                )
            )
        return fills
