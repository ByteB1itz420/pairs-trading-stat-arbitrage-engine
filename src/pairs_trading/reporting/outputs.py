"""CSV/JSON result exports and optional diagnostic plots."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from pairs_trading.backtest.engine import BacktestResult


def _records(values: tuple[object, ...]) -> pd.DataFrame:
    return pd.DataFrame([asdict(value) for value in values])


def export_results(
    result: BacktestResult,
    metrics: dict[str, float | int | None],
    output_directory: str | Path,
) -> None:
    """Write deterministic machine-readable outputs for a completed backtest."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(output / "equity_curve.csv")
    result.spread.to_frame().to_csv(output / "spread.csv")
    result.zscore.to_frame().to_csv(output / "zscore.csv")
    _records(result.fills).to_csv(output / "fills.csv", index=False)
    _records(result.trades).to_csv(output / "trades.csv", index=False)
    _records(result.signals).to_csv(output / "signals.csv", index=False)
    _records(result.cancelled_orders).to_csv(output / "cancelled_orders.csv", index=False)
    _records(result.refits).to_csv(output / "pair_refits.csv", index=False)
    with (output / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, allow_nan=False)


def plot_backtest_report(
    result: BacktestResult,
    output_path: str | Path,
) -> None:
    """Save equity/drawdown and spread/z-score diagnostics with signal markers."""
    import matplotlib.pyplot as plt

    if result.equity_curve.empty:
        raise ValueError("Cannot plot an empty backtest result.")
    figure, (equity_axis, drawdown_axis, signal_axis) = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 2]},
    )
    equity = result.equity_curve
    equity_axis.plot(equity.index, equity["net_equity"], label="Net equity")
    equity_axis.plot(equity.index, equity["gross_equity"], label="Gross equity", alpha=0.7)
    equity_axis.plot(
        equity.index,
        equity["cash_benchmark"],
        label="Cash benchmark",
        linestyle="--",
    )
    equity_axis.set_ylabel("Portfolio value")
    equity_axis.legend()

    drawdown = equity["net_equity"] / equity["net_equity"].cummax() - 1
    drawdown_axis.fill_between(drawdown.index, drawdown, 0, alpha=0.35)
    drawdown_axis.set_ylabel("Drawdown")

    signal_axis.plot(result.spread.index, result.spread, label="Spread", alpha=0.6)
    signal_axis.set_ylabel("Spread")
    zscore_axis = signal_axis.twinx()
    zscore_axis.plot(result.zscore.index, result.zscore, label="Z-score", color="tab:orange")
    zscore_axis.axhline(0, color="black", linewidth=0.7)
    zscore_axis.set_ylabel("Z-score")
    for signal in result.signals:
        if signal.zscore is None:
            continue
        marker = {
            "enter_long": "^",
            "enter_short": "v",
            "exit": "x",
        }.get(signal.action.value)
        if marker is not None:
            zscore_axis.scatter(signal.timestamp, signal.zscore, marker=marker, color="tab:red")
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
