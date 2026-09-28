import pytest

from pairs_trading.research.screening import adjust_pvalues_bh


def test_benjamini_hochberg_adjustment_preserves_input_order() -> None:
    assert adjust_pvalues_bh([0.01, 0.04, 0.03, 0.002]) == pytest.approx(
        [0.02, 0.04, 0.04, 0.008]
    )


def test_benjamini_hochberg_rejects_invalid_pvalues() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        adjust_pvalues_bh([0.1, 1.1])
