"""Validation and normalization for long-format market price observations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "symbol", "adjusted_close"}


@dataclass(frozen=True)
class MarketData:
    """Validated wide price panels; missing observations are never forward-filled."""

    adjusted_close: pd.DataFrame
    open: pd.DataFrame | None
    quality: "DataQualityReport"


@dataclass(frozen=True)
class DataQualityReport:
    observations: int
    symbols: tuple[str, ...]
    start: pd.Timestamp
    end: pd.Timestamp
    missing_adjusted_close: dict[str, int]
    missing_open: dict[str, int]


def validate_market_data(
    observations: pd.DataFrame,
    min_observations: int = 1,
) -> MarketData:
    """Normalize and validate long-format data into symbol-indexed price panels."""
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1.")
    missing_columns = REQUIRED_COLUMNS - set(observations.columns)
    if missing_columns:
        raise ValueError(f"Missing required market-data columns: {', '.join(sorted(missing_columns))}.")
    if observations.empty:
        raise ValueError("Market data contains no observations.")

    frame = observations.copy()
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError("Market data contains invalid timestamps.")
    frame["timestamp"] = timestamps.dt.tz_convert(None)
    frame["symbol"] = frame["symbol"].astype("string").str.strip().str.upper()
    if frame["symbol"].isna().any() or frame["symbol"].eq("").any():
        raise ValueError("Market data contains empty symbols.")
    if frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("Market data contains duplicate timestamp/symbol observations.")

    numeric_fields = ["adjusted_close"]
    if "open" in frame.columns:
        numeric_fields.append("open")
    for field in numeric_fields:
        frame[field] = pd.to_numeric(frame[field], errors="raise")
        present = frame[field].dropna()
        if not present.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all():
            raise ValueError(f"Market data field '{field}' contains infinite prices.")
        if (present <= 0).any():
            raise ValueError(f"Market data field '{field}' contains non-positive prices.")

    close = frame.pivot(index="timestamp", columns="symbol", values="adjusted_close").sort_index()
    close.columns.name = None
    counts = close.count()
    insufficient = counts[counts < min_observations]
    if not insufficient.empty:
        symbols = ", ".join(f"{symbol} ({count})" for symbol, count in insufficient.items())
        raise ValueError(f"Insufficient adjusted-close history: {symbols}.")

    open_prices: pd.DataFrame | None = None
    if "open" in frame.columns:
        open_prices = frame.pivot(index="timestamp", columns="symbol", values="open").sort_index()
        open_prices = open_prices.reindex(index=close.index, columns=close.columns)
        open_prices.columns.name = None

    report = DataQualityReport(
        observations=len(frame),
        symbols=tuple(close.columns),
        start=close.index.min(),
        end=close.index.max(),
        missing_adjusted_close=close.isna().sum().astype(int).to_dict(),
        missing_open=(
            open_prices.isna().sum().astype(int).to_dict()
            if open_prices is not None
            else {}
        ),
    )
    return MarketData(adjusted_close=close, open=open_prices, quality=report)


def align_pair(
    adjusted_close: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    min_observations: int = 1,
) -> pd.DataFrame:
    """Return common, non-missing pair observations without filling calendar gaps."""
    symbol_a = symbol_a.strip().upper()
    symbol_b = symbol_b.strip().upper()
    if symbol_a == symbol_b:
        raise ValueError("A pair must contain two distinct symbols.")
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1.")
    missing_symbols = {symbol_a, symbol_b} - set(adjusted_close.columns)
    if missing_symbols:
        raise ValueError(f"Unknown pair symbols: {', '.join(sorted(missing_symbols))}.")
    pair = adjusted_close.loc[:, [symbol_a, symbol_b]].dropna(how="any")
    if len(pair) < min_observations:
        raise ValueError(
            f"Pair {symbol_a}/{symbol_b} has {len(pair)} aligned observations; "
            f"{min_observations} required."
        )
    return pair
