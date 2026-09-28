import numpy as np
import pandas as pd

from pairs_trading.research.cointegration import analyze_pair
from pairs_trading.research.screening import screen_pairs


def make_cointegrated_prices(seed: int = 7, count: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_b = np.cumsum(rng.normal(0, 0.01, count)) + 4.0
    residual = np.zeros(count)
    shocks = rng.normal(0, 0.01, count)
    for index in range(1, count):
        residual[index] = 0.65 * residual[index - 1] + shocks[index]
    log_a = 1.2 + 0.8 * log_b + residual
    return pd.DataFrame(
        {"B": np.exp(log_b), "A": np.exp(log_a)},
        index=pd.date_range("2020-01-01", periods=count, freq="B"),
    )


def test_ols_recovers_known_hedge_ratio_and_engle_granger_finds_cointegration() -> None:
    result = analyze_pair(make_cointegrated_prices(), "A", "B")

    assert result.eligible
    assert result.symbol_a == "A"
    assert result.symbol_b == "B"
    assert abs(result.hedge_ratio - 0.8) < 0.08
    assert result.engle_granger_pvalue < 0.01
    assert result.adf_statistic is not None
    assert result.adf_lags is not None
    assert result.adf_diagnostic_pass


def test_screening_is_deterministic_and_adjusts_all_candidate_tests() -> None:
    prices = make_cointegrated_prices()
    prices["C"] = np.exp(np.cumsum(np.random.default_rng(31).normal(0, 0.01, len(prices))) + 4)

    results = screen_pairs(prices[["C", "B", "A"]], formation_window_days=500)

    assert [(item.symbol_a, item.symbol_b) for item in results] == [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
    ]
    assert results[0].cointegrated
    assert all(item.adjusted_pvalue is not None for item in results)


def test_insufficient_history_is_retained_as_an_excluded_candidate() -> None:
    prices = make_cointegrated_prices(count=100)
    prices.loc[prices.index[:20], "A"] = np.nan

    results = screen_pairs(prices, formation_window_days=100, min_observations=90)

    assert len(results) == 1
    assert not results[0].eligible
    assert results[0].exclusion_reason is not None
