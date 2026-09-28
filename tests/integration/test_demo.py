from pairs_trading.config import ExecutionConfig, SignalConfig
from pairs_trading.demo import make_demo_market
from pairs_trading.research.cointegration import analyze_pair
from pairs_trading.research.screening import screen_pairs
from pairs_trading.backtest.engine import run_backtest
from pairs_trading.reporting.metrics import calculate_metrics


def test_synthetic_demo_can_be_screened_and_backtested_end_to_end() -> None:
    market, test_start = make_demo_market()
    training = market.adjusted_close.loc[
        market.adjusted_close.index < test_start
    ]
    screening = screen_pairs(
        training,
        formation_window_days=252,
        min_observations=100,
    )

    assert len(screening) == 1
    assert screening[0].cointegrated
    pair = analyze_pair(
        training.tail(252),
        screening[0].symbol_a,
        screening[0].symbol_b,
        min_observations=100,
    )
    result = run_backtest(
        market,
        pair,
        SignalConfig(zscore_window_days=20),
        ExecutionConfig(initial_capital=100_000),
        start_date=test_start,
        refit_frequency_days=21,
        formation_window_days=252,
        min_refit_observations=100,
    )
    metrics = calculate_metrics(
        result.equity_curve,
        result.trades,
        result.fills,
        initial_capital=100_000,
    )

    assert len(result.equity_curve) == 120
    assert result.zscore.notna().any()
    assert metrics["trade_count"] >= 1
