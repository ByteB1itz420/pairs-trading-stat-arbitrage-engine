import asyncio
from datetime import date

from httpx import ASGITransport, AsyncClient
import numpy as np

from pairs_trading import api
from pairs_trading.api import app


def _request(method: str, url: str, **kwargs):
    async def send():
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(send())


def _backtest_payload() -> dict[str, object]:
    rng = np.random.default_rng(29)
    count = 150
    log_b = np.cumsum(rng.normal(0, 0.01, count)) + 4
    residual = np.zeros(count)
    innovations = rng.normal(0, 0.003, count)
    for index in range(1, count):
        residual[index] = 0.45 * residual[index - 1] + innovations[index]
    close_a = np.exp(0.3 + 0.75 * log_b + residual)
    close_b = np.exp(log_b)
    timestamps = np.busday_offset(
        np.datetime64("2024-01-02"),
        np.arange(count),
        roll="forward",
    )
    rows = []
    for timestamp, price_a, price_b in zip(timestamps, close_a, close_b, strict=True):
        day = str(timestamp)
        rows.extend(
            (
                {
                    "timestamp": day,
                    "symbol": "A",
                    "adjusted_close": float(price_a),
                    "open": float(price_a * 0.999),
                },
                {
                    "timestamp": day,
                    "symbol": "B",
                    "adjusted_close": float(price_b),
                    "open": float(price_b * 0.999),
                },
            )
        )
    return {
        "rows": rows,
        "symbol_a": "A",
        "symbol_b": "B",
        "train_end": str(date.fromisoformat(str(timestamps[109]))),
        "test_start": str(date.fromisoformat(str(timestamps[110]))),
        "formation_window_days": 110,
        "min_observations": 100,
        "zscore_window_days": 20,
        "refit_frequency_days": 0,
    }


def test_healthcheck() -> None:
    response = _request("GET", "/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_backtest_screens_training_data_and_returns_out_of_sample_results() -> None:
    response = _request("POST", "/api/v1/backtest", json=_backtest_payload())

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "complete"
    assert result["pair"]["cointegrated"] is True
    assert result["out_of_sample"]["observations"] == 40
    assert result["out_of_sample"]["start"] == _backtest_payload()["test_start"]
    assert len(result["equity_curve"]) == 40
    assert "cumulative_net_return" in result["metrics"]


def test_backtest_rejects_improper_training_test_order() -> None:
    payload = _backtest_payload()
    payload["train_end"] = payload["test_start"]

    response = _request("POST", "/api/v1/backtest", json=payload)

    assert response.status_code == 422


def test_configured_api_key_is_required(monkeypatch) -> None:
    monkeypatch.setattr(api, "API_TOKEN", "local-test-token")

    response = _request("POST", "/api/v1/backtest", json=_backtest_payload())

    assert response.status_code == 401


def test_request_body_size_is_limited(monkeypatch) -> None:
    monkeypatch.setattr(api, "MAX_REQUEST_BYTES", 4)

    response = _request("POST", "/healthz", content=b"12345")

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body exceeds 8 MiB."
