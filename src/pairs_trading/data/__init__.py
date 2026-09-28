"""Market-data ingestion and validation."""

from pairs_trading.data.loader import load_prices_csv
from pairs_trading.data.validation import (
    DataQualityReport,
    MarketData,
    align_pair,
    validate_market_data,
)

__all__ = [
    "DataQualityReport",
    "MarketData",
    "align_pair",
    "load_prices_csv",
    "validate_market_data",
]
