"""Typed experiment configuration and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
import math
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def _require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")


@dataclass(frozen=True)
class DataConfig:
    frequency: str = "daily"
    price_field: str = "adjusted_close"
    min_observations: int = 252
    path: str = "data/raw/prices.csv"

    def __post_init__(self) -> None:
        if isinstance(self.min_observations, bool) or not isinstance(self.min_observations, int):
            raise ValueError("min_observations must be an integer.")
        if self.frequency != "daily":
            raise ValueError("Only daily data is supported by the initial engine.")
        if self.price_field != "adjusted_close":
            raise ValueError("The initial data contract requires adjusted_close prices.")
        if self.min_observations < 20:
            raise ValueError("min_observations must be at least 20 for statistical screening.")


@dataclass(frozen=True)
class ScreeningConfig:
    formation_window_days: int = 252
    engle_granger_significance: float = 0.05
    adf_significance: float = 0.05
    multiple_testing_method: str = "benjamini_hochberg"
    refit_frequency_days: int = 21

    def __post_init__(self) -> None:
        for name in ("engle_granger_significance", "adf_significance"):
            value = getattr(self, name)
            _require_finite(name, value)
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.formation_window_days, self.refit_frequency_days)
        ):
            raise ValueError("Formation and refit windows must be integers.")
        if self.formation_window_days < 20 or self.refit_frequency_days < 1:
            raise ValueError("Formation and refit windows must be at least 20 and positive.")
        if self.multiple_testing_method != "benjamini_hochberg":
            raise ValueError("Only benjamini_hochberg multiple-testing correction is supported.")


@dataclass(frozen=True)
class SignalConfig:
    zscore_window_days: int = 60
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    stop_threshold: float = 4.0
    max_holding_days: int = 30

    def __post_init__(self) -> None:
        for name in ("entry_threshold", "exit_threshold", "stop_threshold"):
            _require_finite(name, getattr(self, name))
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.zscore_window_days, self.max_holding_days)
        ):
            raise ValueError("Signal windows and holding period must be integers.")
        if self.zscore_window_days < 2 or self.max_holding_days < 1:
            raise ValueError("Signal windows and holding period must be positive.")
        if not 0 <= self.exit_threshold < self.entry_threshold:
            raise ValueError("exit_threshold must be non-negative and below entry_threshold.")
        if self.stop_threshold <= self.entry_threshold:
            raise ValueError("stop_threshold must exceed entry_threshold.")


@dataclass(frozen=True)
class ExecutionConfig:
    decision_time: str = "close"
    fill_time: str = "next_open"
    commission_bps_per_side: float = 2.0
    slippage_bps_per_side: float = 5.0
    annual_borrow_rate: float = 0.0
    initial_capital: float = 100_000.0
    gross_exposure_fraction: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "commission_bps_per_side",
            "slippage_bps_per_side",
            "annual_borrow_rate",
            "initial_capital",
            "gross_exposure_fraction",
        ):
            _require_finite(name, getattr(self, name))
        if self.decision_time != "close" or self.fill_time != "next_open":
            raise ValueError("Execution must decide at close and fill at the next open.")
        if min(
            self.commission_bps_per_side,
            self.slippage_bps_per_side,
            self.annual_borrow_rate,
        ) < 0:
            raise ValueError("Execution costs cannot be negative.")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if not 0 < self.gross_exposure_fraction <= 1:
            raise ValueError("gross_exposure_fraction must be in (0, 1].")


@dataclass(frozen=True)
class EvaluationConfig:
    split_method: str = "chronological"
    report_gross_and_net: bool = True
    risk_free_rate_annual: float = 0.0
    annualization_days: int = 252
    train_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None

    def __post_init__(self) -> None:
        _require_finite("risk_free_rate_annual", self.risk_free_rate_annual)
        if isinstance(self.annualization_days, bool) or not isinstance(self.annualization_days, int):
            raise ValueError("annualization_days must be an integer.")
        if self.split_method != "chronological":
            raise ValueError("Only chronological evaluation splits are supported.")
        if self.annualization_days < 1:
            raise ValueError("annualization_days must be at least 1.")
        dates: dict[str, date] = {}
        for name in ("train_end", "test_start", "test_end"):
            value = getattr(self, name)
            if value is None:
                continue
            try:
                parsed = date.fromisoformat(value)
            except ValueError as error:
                raise ValueError(f"{name} must use YYYY-MM-DD format.") from error
            if parsed.isoformat() != value:
                raise ValueError(f"{name} must use YYYY-MM-DD format.")
            dates[name] = parsed
        if "train_end" in dates and "test_start" in dates:
            if dates["train_end"] >= dates["test_start"]:
                raise ValueError("test_start must be later than train_end.")
        if "test_end" in dates and "test_start" in dates:
            if dates["test_end"] < dates["test_start"]:
                raise ValueError("test_end cannot be earlier than test_start.")


@dataclass(frozen=True)
class ExperimentConfig:
    universe: tuple[str, ...] = ()
    data: DataConfig = DataConfig()
    screening: ScreeningConfig = ScreeningConfig()
    signals: SignalConfig = SignalConfig()
    execution: ExecutionConfig = ExecutionConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    def __post_init__(self) -> None:
        if any(not isinstance(symbol, str) for symbol in self.universe):
            raise ValueError("Universe symbols must be strings.")
        if self.screening.formation_window_days < self.data.min_observations:
            raise ValueError("formation_window_days cannot be below data.min_observations.")
        normalized = tuple(symbol.strip().upper() for symbol in self.universe)
        if any(not symbol for symbol in normalized):
            raise ValueError("Universe symbols cannot be empty.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Universe symbols must be unique.")
        object.__setattr__(self, "universe", normalized)


def _build_section(section_type: type[T], values: Any, name: str) -> T:
    if values is None:
        return section_type()
    if not isinstance(values, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping.")
    allowed = {field.name for field in fields(section_type)}
    unexpected = values.keys() - allowed
    if unexpected:
        raise ValueError(f"Unknown keys in '{name}': {', '.join(sorted(unexpected))}.")
    return section_type(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from a YAML file."""
    with Path(path).open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError("Experiment configuration must be a YAML mapping.")

    allowed = {"universe", "data", "screening", "signals", "execution", "evaluation"}
    unexpected = values.keys() - allowed
    if unexpected:
        raise ValueError(f"Unknown experiment configuration keys: {', '.join(sorted(unexpected))}.")

    universe = values.get("universe", ())
    if not isinstance(universe, (list, tuple)):
        raise ValueError("'universe' must be a list of symbols.")
    return ExperimentConfig(
        universe=tuple(universe),
        data=_build_section(DataConfig, values.get("data"), "data"),
        screening=_build_section(ScreeningConfig, values.get("screening"), "screening"),
        signals=_build_section(SignalConfig, values.get("signals"), "signals"),
        execution=_build_section(ExecutionConfig, values.get("execution"), "execution"),
        evaluation=_build_section(EvaluationConfig, values.get("evaluation"), "evaluation"),
    )
