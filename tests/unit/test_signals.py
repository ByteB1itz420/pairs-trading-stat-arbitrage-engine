import numpy as np
import pandas as pd
import pytest

from pairs_trading.config import SignalConfig
from pairs_trading.research.cointegration import PairAnalysis
from pairs_trading.research.spread import calculate_spread, rolling_zscore
from pairs_trading.signals.zscore import PositionSide, SignalAction, ZScoreStrategy


def test_rolling_zscore_has_warmup_and_uses_trailing_values() -> None:
    spread = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])

    zscore = rolling_zscore(spread, window=3)

    assert zscore.iloc[:2].isna().all()
    assert zscore.iloc[2] == 1.0
    assert zscore.iloc[3] == 1.0
    assert zscore.iloc[4] == 1.0


def test_zero_variance_and_invalid_prices_do_not_create_signals() -> None:
    zscore = rolling_zscore(pd.Series([2.0] * 5), window=3)
    assert zscore.isna().all()

    pair = PairAnalysis("A", "B", 10, intercept=0.0, hedge_ratio=1.0)
    prices = pd.DataFrame({"A": [10.0, 0.0], "B": [8.0, 8.0]})
    with pytest.raises(ValueError, match="finite and positive"):
        calculate_spread(prices, pair)


def test_long_and_short_spread_entries_and_mean_reversion_exits() -> None:
    strategy = ZScoreStrategy(SignalConfig())
    timestamp = pd.Timestamp("2025-01-01")

    assert strategy.on_close(timestamp, -2.0).action == SignalAction.ENTER_LONG
    strategy.on_fill(PositionSide.LONG_SPREAD)
    assert strategy.on_close(timestamp, 0.0).reason == "mean_reversion"
    strategy.on_fill(PositionSide.FLAT)
    assert strategy.on_close(timestamp, 2.0).action == SignalAction.ENTER_SHORT
    strategy.on_fill(PositionSide.SHORT_SPREAD)
    assert strategy.on_close(timestamp, 0.5).reason == "mean_reversion"


def test_stops_maximum_holding_and_invalid_values_are_handled() -> None:
    config = SignalConfig(max_holding_days=2)
    strategy = ZScoreStrategy(config)
    timestamp = pd.Timestamp("2025-01-01")
    strategy.on_fill(PositionSide.LONG_SPREAD)
    assert strategy.on_close(timestamp, None).action == SignalAction.HOLD
    signal = strategy.on_close(timestamp, np.nan)
    assert signal.action == SignalAction.EXIT
    assert signal.reason == "maximum_holding_period"

    strategy.on_fill(PositionSide.FLAT)
    strategy.on_fill(PositionSide.SHORT_SPREAD)
    assert strategy.on_close(timestamp, 4.0).reason == "stop_threshold"
