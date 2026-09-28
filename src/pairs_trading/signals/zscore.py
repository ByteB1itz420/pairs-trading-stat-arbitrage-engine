"""Rolling z-score features and a long/short spread state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

import pandas as pd

from pairs_trading.config import SignalConfig


class PositionSide(StrEnum):
    FLAT = "flat"
    LONG_SPREAD = "long_spread"
    SHORT_SPREAD = "short_spread"


class SignalAction(StrEnum):
    HOLD = "hold"
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT = "exit"


@dataclass(frozen=True)
class Signal:
    timestamp: pd.Timestamp
    action: SignalAction
    zscore: float | None
    reason: str


class ZScoreStrategy:
    """Emit close-time intents; position state changes only after an execution fill."""

    def __init__(self, config: SignalConfig) -> None:
        self.config = config
        self.side = PositionSide.FLAT
        self.holding_bars = 0

    def on_close(self, timestamp: pd.Timestamp, zscore: float | None) -> Signal:
        valid_zscore = zscore is not None and math.isfinite(zscore)
        normalized_zscore = float(zscore) if valid_zscore else None
        if self.side == PositionSide.FLAT:
            if normalized_zscore is None:
                return Signal(timestamp, SignalAction.HOLD, None, "invalid_or_warmup_zscore")
            if normalized_zscore >= self.config.entry_threshold:
                return Signal(timestamp, SignalAction.ENTER_SHORT, normalized_zscore, "entry_threshold")
            if normalized_zscore <= -self.config.entry_threshold:
                return Signal(timestamp, SignalAction.ENTER_LONG, normalized_zscore, "entry_threshold")
            return Signal(timestamp, SignalAction.HOLD, normalized_zscore, "inside_entry_band")

        self.holding_bars += 1
        if self.holding_bars >= self.config.max_holding_days:
            return Signal(timestamp, SignalAction.EXIT, normalized_zscore, "maximum_holding_period")
        if normalized_zscore is None:
            return Signal(timestamp, SignalAction.HOLD, None, "invalid_zscore_while_position_open")

        if self.side == PositionSide.LONG_SPREAD:
            if normalized_zscore <= -self.config.stop_threshold:
                return Signal(timestamp, SignalAction.EXIT, normalized_zscore, "stop_threshold")
            if normalized_zscore >= -self.config.exit_threshold:
                return Signal(timestamp, SignalAction.EXIT, normalized_zscore, "mean_reversion")
        elif self.side == PositionSide.SHORT_SPREAD:
            if normalized_zscore >= self.config.stop_threshold:
                return Signal(timestamp, SignalAction.EXIT, normalized_zscore, "stop_threshold")
            if normalized_zscore <= self.config.exit_threshold:
                return Signal(timestamp, SignalAction.EXIT, normalized_zscore, "mean_reversion")
        return Signal(timestamp, SignalAction.HOLD, normalized_zscore, "position_open")

    def on_fill(self, new_side: PositionSide) -> None:
        if self.side == PositionSide.FLAT and new_side in {
            PositionSide.LONG_SPREAD,
            PositionSide.SHORT_SPREAD,
        }:
            self.side = new_side
            self.holding_bars = 0
            return
        if self.side in {PositionSide.LONG_SPREAD, PositionSide.SHORT_SPREAD} and new_side == PositionSide.FLAT:
            self.side = PositionSide.FLAT
            self.holding_bars = 0
            return
        raise ValueError(f"Invalid position transition: {self.side} -> {new_side}.")
