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
