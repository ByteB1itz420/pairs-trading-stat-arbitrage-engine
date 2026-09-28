"""Out-of-sample portfolio and trade statistics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pairs_trading.backtest.engine import TradeRecord
from pairs_trading.backtest.events import Fill


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: tuple[TradeRecord, ...] | list[TradeRecord],
    fills: tuple[Fill, ...] | list[Fill],
    initial_capital: float,
    risk_free_rate_annual: float = 0.0,
    annualization_days: int = 252,
) -> dict[str, float | int | None]:
    """Compute metrics on net results while retaining gross and cost diagnostics."""
    required = {"net_equity", "gross_equity", "gross_exposure"}
    missing = required - set(equity_curve.columns)
    if missing:
        raise ValueError(f"Equity curve is missing columns: {', '.join(sorted(missing))}.")
    if equity_curve.empty:
        raise ValueError("Cannot calculate performance metrics for an empty equity curve.")
    if initial_capital <= 0 or annualization_days < 1:
        raise ValueError("Initial capital and annualization days must be positive.")
    net_equity = pd.to_numeric(equity_curve["net_equity"], errors="raise")
    gross_equity = pd.to_numeric(equity_curve["gross_equity"], errors="raise")
    if not np.isfinite(net_equity.to_numpy()).all() or not np.isfinite(
        gross_equity.to_numpy()
    ).all():
        raise ValueError("Equity curve contains non-finite values.")

    net_with_initial = pd.concat(
        [pd.Series([initial_capital]), net_equity.reset_index(drop=True)],
        ignore_index=True,
    )
    gross_with_initial = pd.concat(
        [pd.Series([initial_capital]), gross_equity.reset_index(drop=True)],
        ignore_index=True,
    )
    returns = net_with_initial.pct_change().dropna()
    gross_returns = gross_with_initial.pct_change().dropna()
    periods = max(len(equity_curve), 1)
    end_equity = float(net_equity.iloc[-1])
    gross_end_equity = float(gross_equity.iloc[-1])
    net_return = end_equity / initial_capital - 1
    gross_return = gross_end_equity / initial_capital - 1
    annualized_return = (
        (end_equity / initial_capital) ** (annualization_days / periods) - 1
        if end_equity > 0
        else None
    )

    volatility = float(returns.std(ddof=1) * math.sqrt(annualization_days)) if len(returns) > 1 else None
    return_std = float(returns.std(ddof=1)) if len(returns) > 1 else None
    sharpe = (
        float(
            (returns - risk_free_rate_annual / annualization_days).mean()
            / return_std
            * math.sqrt(annualization_days)
        )
        if return_std is not None and return_std > 0
        else None
    )
    drawdown = net_with_initial / net_with_initial.cummax() - 1
    underwater = drawdown.lt(0)
    drawdown_duration = 0
    current_duration = 0
    for is_underwater in underwater:
        current_duration = current_duration + 1 if is_underwater else 0
        drawdown_duration = max(drawdown_duration, current_duration)

    trade_pnls = [trade.net_pnl for trade in trades]
    winners = [pnl for pnl in trade_pnls if pnl > 0]
    losers = [pnl for pnl in trade_pnls if pnl < 0]
    total_traded_notional = sum(
        abs(fill.quantity * fill.execution_price)
        for fill in fills
    )
    average_equity = float(net_equity.mean())
    maximum_exposure = float(equity_curve["gross_exposure"].max())
    borrow_costs = (
        float(equity_curve["borrow_costs"].iloc[-1])
        if "borrow_costs" in equity_curve
        else 0.0
    )

    return {
        "starting_capital": float(initial_capital),
        "ending_net_equity": end_equity,
        "ending_gross_equity": gross_end_equity,
        "cumulative_net_return": float(net_return),
        "cumulative_gross_return": float(gross_return),
        "cash_benchmark_return": 0.0,
        "gross_minus_net_return": float(gross_return - net_return),
        "annualized_net_return": float(annualized_return) if annualized_return is not None else None,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": float(drawdown.min()),
        "maximum_drawdown_duration_bars": drawdown_duration,
        "trade_count": len(trades),
        "win_rate": len(winners) / len(trade_pnls) if trade_pnls else None,
        "average_trade_pnl": float(np.mean(trade_pnls)) if trade_pnls else None,
        "average_winning_trade": float(np.mean(winners)) if winners else None,
        "average_losing_trade": float(np.mean(losers)) if losers else None,
        "average_holding_bars": (
            float(np.mean([trade.holding_bars for trade in trades])) if trades else None
        ),
        "turnover_on_initial_capital": total_traded_notional / initial_capital,
        "total_commissions": float(sum(fill.commission for fill in fills)),
        "total_slippage_costs": float(sum(fill.slippage_cost for fill in fills)),
        "total_borrow_costs": borrow_costs,
        "time_in_market": float((equity_curve["gross_exposure"] > 0).mean()),
        "maximum_gross_exposure_fraction": (
            maximum_exposure / average_equity if average_equity > 0 else None
        ),
        "annualized_gross_volatility": (
            float(gross_returns.std(ddof=1) * math.sqrt(annualization_days))
            if len(gross_returns) > 1
            else None
        ),
    }
