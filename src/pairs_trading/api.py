"""HTTP API for bounded, stateless out-of-sample pair backtests."""

from __future__ import annotations

from collections import deque
from datetime import date
import hmac
import os

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from pairs_trading.backtest.engine import run_backtest
from pairs_trading.config import ExecutionConfig, SignalConfig
from pairs_trading.data.validation import validate_market_data
from pairs_trading.reporting.metrics import calculate_metrics
from pairs_trading.research.screening import screen_pairs

MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_PRICE_ROWS = 40_000
TOO_LARGE_BODY = b'{"detail":"Request body exceeds 8 MiB."}'
API_TOKEN = os.getenv("API_TOKEN", "")


class RequestSizeLimitMiddleware:
    """Reject oversized bodies even when the request omits Content-Length."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared_length = next(
            (value for name, value in scope["headers"] if name == b"content-length"),
            None,
        )
        if declared_length is not None:
            try:
                too_large = int(declared_length) > MAX_REQUEST_BYTES
            except ValueError:
                too_large = True
            if too_large:
                await self._reject(send)
                return

        messages: deque[Message] = deque()
        received_bytes = 0
        while True:
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > MAX_REQUEST_BYTES:
                    await self._reject(send)
                    return
                messages.append(message)
                if not message.get("more_body", False):
                    break
            else:
                messages.append(message)
                break

        async def replay_body() -> Message:
            if messages:
                return messages.popleft()
            return await receive()

        await self.app(scope, replay_body, send)

    @staticmethod
    async def _reject(send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(TOO_LARGE_BODY)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": TOO_LARGE_BODY})

class PriceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    timestamp: date
    symbol: str = Field(min_length=1, max_length=24)
    adjusted_close: float = Field(gt=0)
    open: float = Field(gt=0)


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[PriceObservation] = Field(min_length=40, max_length=MAX_PRICE_ROWS)
    symbol_a: str = Field(min_length=1, max_length=24)
    symbol_b: str = Field(min_length=1, max_length=24)
    train_end: date
    test_start: date
    test_end: date | None = None
    formation_window_days: int = Field(default=252, ge=20, le=10_000)
    min_observations: int = Field(default=100, ge=20, le=10_000)
    refit_frequency_days: int = Field(default=21, ge=0, le=10_000)
    zscore_window_days: int = Field(default=60, ge=2, le=10_000)
    entry_threshold: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    exit_threshold: float = Field(default=0.5, ge=0, allow_inf_nan=False)
    stop_threshold: float = Field(default=4.0, gt=0, allow_inf_nan=False)
    max_holding_days: int = Field(default=30, ge=1, le=10_000)
    initial_capital: float = Field(default=100_000, gt=0, allow_inf_nan=False)
    commission_bps_per_side: float = Field(default=2, ge=0, allow_inf_nan=False)
    slippage_bps_per_side: float = Field(default=5, ge=0, allow_inf_nan=False)
    annual_borrow_rate: float = Field(default=0, ge=0, allow_inf_nan=False)
    gross_exposure_fraction: float = Field(default=1, gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_split_and_symbols(self) -> BacktestRequest:
        self.symbol_a = self.symbol_a.strip().upper()
        self.symbol_b = self.symbol_b.strip().upper()
        if not self.symbol_a or not self.symbol_b or self.symbol_a == self.symbol_b:
            raise ValueError("Select two different non-empty symbols.")
        if self.train_end >= self.test_start:
            raise ValueError("test_start must be later than train_end.")
        if self.test_end is not None and self.test_end < self.test_start:
            raise ValueError("test_end cannot be earlier than test_start.")
        if self.min_observations > self.formation_window_days:
            raise ValueError("min_observations cannot exceed formation_window_days.")
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError("exit_threshold must be below entry_threshold.")
        if self.stop_threshold <= self.entry_threshold:
            raise ValueError("stop_threshold must exceed entry_threshold.")
        symbols = {row.symbol.strip().upper() for row in self.rows}
        if symbols != {self.symbol_a, self.symbol_b}:
            raise ValueError("Submit observations for the selected pair only.")
        return self


app = FastAPI(
    title="Pairwise Statistical Arbitrage API",
    version="1.0.0",
    description="Stateless research API for chronological pair screening and backtests.",
)
app.add_middleware(RequestSizeLimitMiddleware)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/backtest")
def backtest(
    request: BacktestRequest,
    x_api_key: str | None = Header(default=None),
) -> dict[str, object]:
    if API_TOKEN and not hmac.compare_digest(x_api_key or "", API_TOKEN):
        raise HTTPException(status_code=401, detail="A valid API key is required.")
    try:
        records = [row.model_dump() for row in request.rows]
        market = validate_market_data(pd.DataFrame.from_records(records), min_observations=20)
        if market.open is None:
            raise ValueError("Next-open backtesting requires open prices.")
        train_end = pd.Timestamp(request.train_end)
        test_start = pd.Timestamp(request.test_start)
        test_end = pd.Timestamp(request.test_end) if request.test_end else None
        training_prices = market.adjusted_close.loc[:train_end]
        candidates = screen_pairs(
            training_prices,
            symbols=[request.symbol_a, request.symbol_b],
            formation_window_days=request.formation_window_days,
            min_observations=request.min_observations,
        )
        pair = candidates[0]
        if not pair.eligible:
            raise ValueError(f"Training data is ineligible: {pair.exclusion_reason}.")
        if not pair.cointegrated:
            return {
                "status": "not_cointegrated",
                "pair": _pair_result(pair),
                "message": "The training-period Engle-Granger screen did not pass.",
            }

        signal_config = SignalConfig(
            zscore_window_days=request.zscore_window_days,
            entry_threshold=request.entry_threshold,
            exit_threshold=request.exit_threshold,
            stop_threshold=request.stop_threshold,
            max_holding_days=request.max_holding_days,
        )
        execution_config = ExecutionConfig(
            initial_capital=request.initial_capital,
            commission_bps_per_side=request.commission_bps_per_side,
            slippage_bps_per_side=request.slippage_bps_per_side,
            annual_borrow_rate=request.annual_borrow_rate,
            gross_exposure_fraction=request.gross_exposure_fraction,
        )
        result = run_backtest(
            market,
            pair,
            signal_config,
            execution_config,
            start_date=test_start,
            end_date=test_end,
            refit_frequency_days=request.refit_frequency_days,
            formation_window_days=request.formation_window_days,
            min_refit_observations=request.min_observations,
        )
        metrics = calculate_metrics(
            result.equity_curve,
            result.trades,
            result.fills,
            execution_config.initial_capital,
        )
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "status": "complete",
        "pair": _pair_result(pair),
        "out_of_sample": {
            "start": result.equity_curve.index.min().date().isoformat(),
            "end": result.equity_curve.index.max().date().isoformat(),
            "observations": len(result.equity_curve),
        },
        "metrics": metrics,
        "equity_curve": [
            {
                "timestamp": timestamp.date().isoformat(),
                "net_equity": float(row["net_equity"]),
                "gross_equity": float(row["gross_equity"]),
                "gross_exposure": float(row["gross_exposure"]),
            }
            for timestamp, row in result.equity_curve.iterrows()
        ],
        "trades": [
            {
                "entry_timestamp": trade.entry_timestamp.date().isoformat(),
                "exit_timestamp": trade.exit_timestamp.date().isoformat(),
                "side": trade.side.value,
                "net_pnl": trade.net_pnl,
                "total_costs": trade.total_costs,
                "holding_bars": trade.holding_bars,
                "exit_reason": trade.exit_reason,
            }
            for trade in result.trades
        ],
    }


def _pair_result(pair) -> dict[str, object]:
    return {
        "symbol_a": pair.symbol_a,
        "symbol_b": pair.symbol_b,
        "observations": pair.observations,
        "hedge_ratio": pair.hedge_ratio,
        "engle_granger_pvalue": pair.engle_granger_pvalue,
        "adf_pvalue": pair.adf_pvalue,
        "adjusted_pvalue": pair.adjusted_pvalue,
        "cointegrated": pair.cointegrated,
    }
