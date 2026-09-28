"""Command-line entry point for reproducible out-of-sample experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
import yaml

from pairs_trading.backtest.engine import run_backtest
from pairs_trading.config import ExperimentConfig, load_config
from pairs_trading.data.loader import load_prices_csv
from pairs_trading.research.cointegration import PairAnalysis, analyze_pair
from pairs_trading.research.screening import screen_pairs
from pairs_trading.reporting.metrics import calculate_metrics
from pairs_trading.reporting.outputs import export_results, plot_backtest_report


def run_experiment(
    config: ExperimentConfig,
    output_directory: str | Path,
) -> tuple[PairAnalysis, dict[str, float | int | None]]:
    """Screen on training history, freeze one pair, then evaluate out of sample."""
    evaluation = config.evaluation
    if evaluation.train_end is None or evaluation.test_start is None:
        raise ValueError(
            "Set evaluation.train_end and evaluation.test_start to freeze "
            "chronological training and test periods."
        )

    market = load_prices_csv(
        config.data.path,
        min_observations=config.data.min_observations,
    )
    training_close = market.adjusted_close.loc[
        market.adjusted_close.index <= pd.Timestamp(evaluation.train_end)
    ]
    if training_close.empty:
        raise ValueError("No training prices occur on or before evaluation.train_end.")

    symbols = list(config.universe) if config.universe else list(training_close.columns)
    screening = screen_pairs(
        training_close,
        symbols=symbols,
        formation_window_days=config.screening.formation_window_days,
        min_observations=config.data.min_observations,
        significance=config.screening.engle_granger_significance,
        adf_significance=config.screening.adf_significance,
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(candidate) for candidate in screening]).to_csv(
        output / "pair_screening.csv",
        index=False,
    )

    selected = [candidate for candidate in screening if candidate.cointegrated]
    if not selected:
        raise ValueError(
            "No pair passed Engle-Granger significance and Benjamini-Hochberg "
            "correction; inspect pair_screening.csv for diagnostics."
        )
    selected.sort(
        key=lambda candidate: (
            candidate.adjusted_pvalue if candidate.adjusted_pvalue is not None else 1.0,
            candidate.symbol_a,
            candidate.symbol_b,
        )
    )
    chosen = selected[0]
    formation = training_close.tail(config.screening.formation_window_days)
    pair = analyze_pair(
        formation,
        chosen.symbol_a,
        chosen.symbol_b,
        min_observations=config.data.min_observations,
        adf_significance=config.screening.adf_significance,
    )

    result = run_backtest(
        market,
        pair,
        config.signals,
        config.execution,
        start_date=evaluation.test_start,
        end_date=evaluation.test_end,
        refit_frequency_days=config.screening.refit_frequency_days,
        formation_window_days=config.screening.formation_window_days,
        min_refit_observations=config.data.min_observations,
    )
    metrics = calculate_metrics(
        result.equity_curve,
        result.trades,
        result.fills,
        initial_capital=config.execution.initial_capital,
        risk_free_rate_annual=evaluation.risk_free_rate_annual,
        annualization_days=evaluation.annualization_days,
    )
    with (output / "experiment_config.yaml").open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(asdict(config), config_file, sort_keys=False)
    export_results(result, metrics, output)
    plot_backtest_report(result, output / "diagnostics.png")
    return pair, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run training-only pair screening and a frozen out-of-sample backtest."
    )
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to the experiment YAML configuration.",
    )
    parser.add_argument(
        "--output",
        default="results/baseline",
        help="Directory for screening, metrics, fills, trades, and plot outputs.",
    )
    args = parser.parse_args()
    pair, metrics = run_experiment(load_config(args.config), args.output)
    print(
        json.dumps(
            {
                "selected_pair": [pair.symbol_a, pair.symbol_b],
                "hedge_ratio": pair.hedge_ratio,
                "metrics": metrics,
            },
            indent=2,
            allow_nan=False,
        )
    )
