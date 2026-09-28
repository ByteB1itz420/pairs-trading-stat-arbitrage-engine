from pathlib import Path

import pytest

from pairs_trading.config import (
    EvaluationConfig,
    ExecutionConfig,
    ExperimentConfig,
    SignalConfig,
    load_config,
)


def test_baseline_config_loads_with_validated_defaults() -> None:
    config = load_config(Path(__file__).parents[2] / "configs" / "baseline.yaml")

    assert config.data.min_observations == 252
    assert config.signals.entry_threshold == 2.0
    assert config.execution.fill_time == "next_open"


def test_universe_is_normalized_and_duplicate_symbols_are_rejected() -> None:
    assert ExperimentConfig(universe=(" aapl ", "msft")).universe == ("AAPL", "MSFT")

    with pytest.raises(ValueError, match="unique"):
        ExperimentConfig(universe=("AAPL", "aapl"))


def test_invalid_signal_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError, match="below entry_threshold"):
        SignalConfig(entry_threshold=1.0, exit_threshold=1.0)


def test_unknown_configuration_keys_are_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("signals:\n  enter_threshold: 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown keys"):
        load_config(config_path)


def test_evaluation_dates_must_be_chronological() -> None:
    with pytest.raises(ValueError, match="later than train_end"):
        EvaluationConfig(train_end="2025-01-02", test_start="2025-01-02")


@pytest.mark.parametrize(
    ("config_type", "kwargs"),
    [
        (SignalConfig, {"entry_threshold": float("nan")}),
        (SignalConfig, {"stop_threshold": float("inf")}),
        (ExecutionConfig, {"commission_bps_per_side": float("nan")}),
        (ExecutionConfig, {"initial_capital": float("inf")}),
        (EvaluationConfig, {"risk_free_rate_annual": float("nan")}),
    ],
)
def test_non_finite_configuration_values_are_rejected(config_type, kwargs) -> None:
    with pytest.raises(ValueError, match="finite"):
        config_type(**kwargs)
