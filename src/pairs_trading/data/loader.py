"""Local CSV market-data loaders."""

from pathlib import Path

import pandas as pd

from pairs_trading.data.validation import MarketData, validate_market_data


def load_prices_csv(
    path: str | Path,
    min_observations: int = 1,
) -> MarketData:
    """Load long-format CSV data with timestamp, symbol, adjusted_close and optional open."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Market-data CSV does not exist: {source}")
    observations = pd.read_csv(source)
    return validate_market_data(observations, min_observations=min_observations)
