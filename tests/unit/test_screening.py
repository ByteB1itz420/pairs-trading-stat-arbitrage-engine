import pytest

from pairs_trading.research.screening import adjust_pvalues_bh


def test_benjamini_hochberg_adjustment_preserves_input_order() -> None:
    assert adjust_pvalues_bh([0.01, 0.04, 0.03, 0.002]) == pytest.approx(
        [0.02, 0.04, 0.04, 0.008]
    )


def test_benjamini_hochberg_rejects_invalid_pvalues() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        adjust_pvalues_bh([0.1, 1.1])


def test_formation_window_is_calendar_sessions_before_pair_alignment(monkeypatch) -> None:
    """A missing last-session price must not pull an older row into the window."""
    import pandas as pd
    from pairs_trading.research.screening import screen_pairs

    dates = pd.date_range("2025-01-01", periods=35, freq="B")
    prices = pd.DataFrame({"A": range(1, 36), "B": range(2, 37)}, index=dates, dtype=float)
    prices.loc[dates[-1], "B"] = float("nan")
    # The final 20 sessions contain only 19 aligned rows: excluded, not
    # silently extended to an earlier date to make the pair eligible.
    candidates = screen_pairs(prices, formation_window_days=20, min_observations=20)
    assert len(candidates) == 1
    assert candidates[0].observations == 19
    assert candidates[0].eligible is False


def test_constant_pair_is_excluded_without_aborting_other_candidates() -> None:
    """Degenerate prices should not crash a multi-pair research screen."""
    import numpy as np
    import pandas as pd
    from pairs_trading.research.screening import screen_pairs

    rng = np.random.default_rng(37)
    dates = pd.bdate_range("2020-01-01", periods=120)
    steps = rng.normal(0, 0.01, len(dates))
    prices = pd.DataFrame(
        {"A": np.full(len(dates), 100.0),
         "B": np.exp(4 + np.cumsum(steps)),
         "C": np.exp(4.2 + np.cumsum(steps) + rng.normal(0, 0.003, len(dates)))},
        index=dates,
    )
    result = screen_pairs(prices, formation_window_days=120, min_observations=100)
    assert len(result) == 3
    constant_pairs = [pair for pair in result if "A" in (pair.symbol_a, pair.symbol_b)]
    assert len(constant_pairs) == 2
    assert all(not pair.eligible and "constant" in pair.exclusion_reason for pair in constant_pairs)
    assert next(pair for pair in result if (pair.symbol_a, pair.symbol_b) == ("B", "C")).eligible
