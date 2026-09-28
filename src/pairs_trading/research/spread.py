"""Residual spread construction from frozen, training-period pair parameters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs_trading.research.cointegration import PairAnalysis


def calculate_spread(
    adjusted_close: pd.DataFrame,
    pair: PairAnalysis,
) -> pd.Series:
    """Apply a previously estimated log-price relationship without refitting it."""
    if pair.intercept is None or pair.hedge_ratio is None:
        raise ValueError("Pair analysis must contain a fitted intercept and hedge ratio.")
    missing = {pair.symbol_a, pair.symbol_b} - set(adjusted_close.columns)
    if missing:
        raise ValueError(f"Missing pair prices: {', '.join(sorted(missing))}.")
    prices = adjusted_close[[pair.symbol_a, pair.symbol_b]]
    valid = prices.notna().all(axis=1)
    if ((prices.loc[valid] <= 0).any().any()
            or not np.isfinite(prices.loc[valid].to_numpy(dtype=float)).all()):
        raise ValueError("Spread input prices must be finite and positive.")
    spread = pd.Series(np.nan, index=prices.index, name="spread", dtype=float)
    spread.loc[valid] = (
        np.log(prices.loc[valid, pair.symbol_a])
        - pair.hedge_ratio * np.log(prices.loc[valid, pair.symbol_b])
        - pair.intercept
    )
    return spread


def rolling_zscore(spread: pd.Series, window: int, epsilon: float = 1e-12) -> pd.Series:
    """Compute a trailing z-score; incomplete or degenerate windows remain unavailable."""
    if window < 2:
        raise ValueError("window must be at least 2.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    rolling = spread.rolling(window=window, min_periods=window)
    mean = rolling.mean()
    standard_deviation = rolling.std(ddof=1)
    valid = standard_deviation > epsilon
    return ((spread - mean) / standard_deviation.where(valid)).rename("zscore")
