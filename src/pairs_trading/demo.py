"""Deterministic synthetic market data for safely trying the research workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs_trading.data.validation import MarketData, validate_market_data


def make_demo_market(
    training_observations: int = 360,
    test_observations: int = 120,
    seed: int = 731,
) -> tuple[MarketData, pd.Timestamp]:
    """Build a synthetic cointegrated pair with a reverting test-period dislocation."""
    if training_observations < 100 or test_observations < 20:
        raise ValueError("Demo data requires at least 100 training and 20 test observations.")
    count = training_observations + test_observations
    rng = np.random.default_rng(seed)
    log_b = 4.5 + np.cumsum(rng.normal(0, 0.01, count))
    residual = np.zeros(count)
    residual_shocks = rng.normal(0, 0.008, count)
    for index in range(1, count):
        residual[index] = 0.72 * residual[index - 1] + residual_shocks[index]
    residual[training_observations + 5 : training_observations + 9] += [
        0.055,
        0.042,
        0.028,
        0.015,
    ]
    log_a = 0.35 + 0.8 * log_b + residual
    close = np.column_stack((np.exp(log_a), np.exp(log_b)))
    overnight_moves = rng.normal(0, 0.001, (count, 2))
    open_prices = close * np.exp(overnight_moves)
    timestamps = pd.date_range("2022-01-03", periods=count, freq="B")
    observations = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, 2),
            "symbol": ["ALPHA", "BETA"] * count,
            "adjusted_close": close.reshape(-1),
            "open": open_prices.reshape(-1),
        }
    )
    market = validate_market_data(
        observations,
        min_observations=training_observations,
    )
    return market, timestamps[training_observations]
