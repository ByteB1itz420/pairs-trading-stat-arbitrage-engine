import pandas as pd
import pytest

from pairs_trading.data.loader import load_prices_csv
from pairs_trading.data.validation import align_pair, validate_market_data


@pytest.fixture
def observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "symbol": [" a ", "B", "A", "B"],
            "adjusted_close": [10.0, 20.0, 11.0, 21.0],
            "open": [9.5, 19.5, 10.5, 20.5],
        }
    )


def test_validation_normalizes_symbols_and_builds_price_panels(
    observations: pd.DataFrame,
) -> None:
    market = validate_market_data(observations)

    assert list(market.adjusted_close.columns) == ["A", "B"]
    assert market.adjusted_close.loc["2025-01-02", "A"] == 10.0
    assert market.open is not None
    assert market.open.loc["2025-01-03", "B"] == 20.5
    assert market.quality.observations == 4
    assert market.quality.missing_adjusted_close == {"A": 0, "B": 0}


def test_duplicate_observations_and_invalid_prices_are_rejected(
    observations: pd.DataFrame,
) -> None:
    duplicate = pd.concat([observations, observations.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_market_data(duplicate)

    observations.loc[0, "adjusted_close"] = 0
    with pytest.raises(ValueError, match="non-positive"):
        validate_market_data(observations)

    invalid_numeric = observations.copy()
    invalid_numeric["adjusted_close"] = invalid_numeric["adjusted_close"].astype(object)
    invalid_numeric.loc[0, "adjusted_close"] = "not-a-price"
    with pytest.raises(ValueError):
        validate_market_data(invalid_numeric)


def test_pair_alignment_drops_missing_intersection_without_forward_fill() -> None:
    prices = pd.DataFrame(
        {"A": [10.0, None, 12.0], "B": [20.0, 21.0, None]},
        index=pd.date_range("2025-01-01", periods=3),
    )

    pair = align_pair(prices, "a", " b ")

    assert len(pair) == 1
    assert pair.iloc[0].tolist() == [10.0, 20.0]


def test_minimum_history_and_unknown_symbols_are_rejected(
    observations: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="Insufficient"):
        validate_market_data(observations, min_observations=3)

    market = validate_market_data(observations)
    with pytest.raises(ValueError, match="Unknown pair symbols"):
        align_pair(market.adjusted_close, "A", "UNKNOWN")


def test_csv_loader_reports_missing_prices_without_imputing(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "timestamp": ["2025-01-02", "2025-01-02", "2025-01-03"],
            "symbol": ["A", "B", "A"],
            "adjusted_close": [10.0, 20.0, 11.0],
        }
    ).to_csv(csv_path, index=False)

    market = load_prices_csv(csv_path)

    assert market.quality.missing_adjusted_close == {"A": 0, "B": 1}
    assert pd.isna(market.adjusted_close.loc["2025-01-03", "B"])
    with pytest.raises(FileNotFoundError):
        load_prices_csv(tmp_path / "absent.csv")
