"""Local visual workbench for trying pair screening and backtests."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from io import BytesIO
import json

import pandas as pd
import streamlit as st

from pairs_trading.backtest.engine import run_backtest
from pairs_trading.config import ExecutionConfig, SignalConfig
from pairs_trading.data.validation import MarketData, validate_market_data
from pairs_trading.demo import make_demo_market
from pairs_trading.reporting.metrics import calculate_metrics
from pairs_trading.research.cointegration import PairAnalysis, analyze_pair
from pairs_trading.research.screening import screen_pairs


def _load_uploaded_market(uploaded_file, min_observations: int) -> MarketData:
    observations = pd.read_csv(uploaded_file)
    return validate_market_data(observations, min_observations=min_observations)


def _dataframe_download(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=True).encode("utf-8")


def _display_backtest(
    market: MarketData,
    pair: PairAnalysis,
    signal_config: SignalConfig,
    execution_config: ExecutionConfig,
    test_start: date,
    test_end: date,
    refit_frequency: int,
    formation_window: int,
    min_observations: int,
) -> None:
    result = run_backtest(
        market,
        pair,
        signal_config,
        execution_config,
        start_date=test_start,
        end_date=test_end,
        refit_frequency_days=refit_frequency,
        formation_window_days=formation_window,
        min_refit_observations=min_observations,
    )
    metrics = calculate_metrics(
        result.equity_curve,
        result.trades,
        result.fills,
        initial_capital=execution_config.initial_capital,
    )
    st.subheader(f"Out-of-sample results: {pair.symbol_a} / {pair.symbol_b}")
    st.caption(
        f"OLS hedge ratio: {pair.hedge_ratio:.5f} · "
        f"Engle–Granger p-value: {pair.engle_granger_pvalue:.5g}"
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Net return", f"{metrics['cumulative_net_return']:.2%}")
    metric_columns[1].metric("Gross return", f"{metrics['cumulative_gross_return']:.2%}")
    metric_columns[2].metric("Maximum drawdown", f"{metrics['maximum_drawdown']:.2%}")
    metric_columns[3].metric("Trades", str(metrics["trade_count"]))
    st.caption(
        f"Sharpe: {metrics['sharpe_ratio'] if metrics['sharpe_ratio'] is not None else 'n/a'} · "
        f"Win rate: {metrics['win_rate'] if metrics['win_rate'] is not None else 'n/a'} · "
        f"Costs: ${metrics['total_commissions'] + metrics['total_slippage_costs'] + metrics['total_borrow_costs']:,.2f}"
    )

    st.markdown("#### Equity and exposure")
    st.line_chart(
        result.equity_curve[
            ["net_equity", "gross_equity", "cash_benchmark"]
        ],
        height=280,
    )
    st.markdown("#### Spread z-score")
    zscore_view = result.zscore.to_frame()
    zscore_view["entry_high"] = signal_config.entry_threshold
    zscore_view["entry_low"] = -signal_config.entry_threshold
    zscore_view["exit_high"] = signal_config.exit_threshold
    zscore_view["exit_low"] = -signal_config.exit_threshold
    st.line_chart(zscore_view, height=260)

    result_columns = st.columns(2)
    with result_columns[0]:
        st.markdown("#### Completed trades")
        st.dataframe(
            pd.DataFrame([asdict(trade) for trade in result.trades]),
            width="stretch",
        )
    with result_columns[1]:
        st.markdown("#### Fills")
        st.dataframe(
            pd.DataFrame([asdict(fill) for fill in result.fills]),
            width="stretch",
        )

    csv_columns = st.columns(3)
    csv_columns[0].download_button(
        "Download equity CSV",
        _dataframe_download(result.equity_curve),
        file_name="equity_curve.csv",
        mime="text/csv",
    )
    csv_columns[1].download_button(
        "Download trades CSV",
        pd.DataFrame([asdict(trade) for trade in result.trades]).to_csv(index=False),
        file_name="trades.csv",
        mime="text/csv",
    )
    csv_columns[2].download_button(
        "Download metrics JSON",
        json.dumps(metrics, indent=2, allow_nan=False),
        file_name="metrics.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(
        page_title="Pairs Trading Lab",
        page_icon="📈",
        layout="wide",
    )
    st.title("Pairs Trading Lab")
    st.caption(
        "Local visual test workbench for synthetic examples or your own historical CSV. "
        "Research only — no live orders are sent."
    )
    st.info(
        "Synthetic demo prices are generated locally and are not market data. "
        "For real experiments, upload data you are licensed to use."
    )

    source = st.sidebar.radio("Data source", ["Synthetic demo", "Upload CSV"])
    if source == "Synthetic demo":
        market, suggested_test_start = make_demo_market()
    else:
        uploaded_file = st.sidebar.file_uploader("Long-format adjusted-price CSV", type=["csv"])
        if uploaded_file is None:
            st.warning("Upload a CSV with timestamp, symbol, adjusted_close, and open columns.")
            st.stop()
        try:
            market = _load_uploaded_market(uploaded_file, min_observations=20)
        except (ValueError, FileNotFoundError) as error:
            st.error(f"Could not load market data: {error}")
            st.stop()
        suggested_test_start = market.adjusted_close.index[
            max(1, int(len(market.adjusted_close) * 0.7))
        ]

    index = market.adjusted_close.index
    st.sidebar.caption(
        f"{len(index)} sessions · {len(market.quality.symbols)} symbols · "
        f"{index.min().date()} to {index.max().date()}"
    )
    symbols = st.sidebar.multiselect(
        "Universe",
        options=list(market.quality.symbols),
        default=list(market.quality.symbols),
    )
    suggested_train_end = (pd.Timestamp(suggested_test_start) - pd.offsets.BDay(1)).date()
    train_end = st.sidebar.date_input(
        "Training data through",
        value=suggested_train_end,
        min_value=index.min().date(),
        max_value=index[-2].date(),
    )
    available_training = market.adjusted_close.loc[
        market.adjusted_close.index <= pd.Timestamp(train_end)
    ]
    if available_training.empty:
        st.error("The training cutoff has no price observations.")
        st.stop()
    max_window = len(available_training)
    if max_window < 20:
        st.error("Choose a training cutoff with at least 20 observations.")
        st.stop()
    formation_window = st.sidebar.slider(
        "Formation window (sessions)",
        min_value=20,
        max_value=max_window,
        value=min(252, max_window),
    )
    min_observations = st.sidebar.slider(
        "Minimum aligned observations",
        min_value=20,
        max_value=formation_window,
        value=min(100, formation_window),
    )
    test_min = (pd.Timestamp(train_end) + pd.Timedelta(days=1)).date()
    test_start = st.sidebar.date_input(
        "Test starts",
        value=max(test_min, pd.Timestamp(suggested_test_start).date()),
        min_value=test_min,
        max_value=index[-1].date(),
    )
    test_end = st.sidebar.date_input(
        "Test ends",
        value=index[-1].date(),
        min_value=test_start,
        max_value=index[-1].date(),
    )

    with st.sidebar.expander("Signal and execution settings"):
        zscore_window = st.number_input("Z-score window", min_value=2, max_value=252, value=60)
        entry_threshold = st.number_input("Entry |z|", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        exit_threshold = st.number_input("Exit |z|", min_value=0.0, max_value=4.9, value=0.5, step=0.1)
        stop_threshold = st.number_input("Stop |z|", min_value=0.6, max_value=8.0, value=4.0, step=0.1)
        max_holding_days = st.number_input("Maximum holding sessions", min_value=1, max_value=252, value=30)
        initial_capital = st.number_input("Initial capital", min_value=100.0, value=100_000.0, step=10_000.0)
        commission_bps = st.number_input("Commission (bps/side)", min_value=0.0, value=2.0, step=0.5)
        slippage_bps = st.number_input("Slippage (bps/side)", min_value=0.0, value=5.0, step=0.5)
        borrow_rate = st.number_input("Annual borrow rate", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
        refit_frequency = st.number_input("OLS refit frequency", min_value=1, max_value=252, value=21)

    if exit_threshold >= entry_threshold:
        st.error("Exit threshold must be below entry threshold.")
        st.stop()
    if stop_threshold <= entry_threshold:
        st.error("Stop threshold must exceed entry threshold.")
        st.stop()
    if not symbols or len(symbols) < 2:
        st.error("Select at least two symbols.")
        st.stop()

    if st.button("Screen pairs and run out-of-sample test", type="primary"):
        try:
            results = screen_pairs(
                available_training,
                symbols=symbols,
                formation_window_days=formation_window,
                min_observations=min_observations,
            )
        except ValueError as error:
            st.error(f"Pair screening failed: {error}")
            st.stop()
        screen_table = pd.DataFrame([asdict(result) for result in results])
        st.subheader("Pair screening diagnostics")
        st.dataframe(screen_table, use_container_width=True)
        passing_pairs = [result for result in results if result.cointegrated]
        if not passing_pairs:
            st.warning(
                "No pairs passed the Engle–Granger and multiple-testing filters. "
                "Inspect the diagnostics, extend training history, or review data quality."
            )
            st.stop()
        passing_pairs.sort(
            key=lambda result: (
                result.adjusted_pvalue if result.adjusted_pvalue is not None else 1.0,
                result.symbol_a,
                result.symbol_b,
            )
        )
        selected = passing_pairs[0]
        pair_model = analyze_pair(
            available_training.tail(formation_window),
            selected.symbol_a,
            selected.symbol_b,
            min_observations=min_observations,
        )
        signal_config = SignalConfig(
            zscore_window_days=int(zscore_window),
            entry_threshold=float(entry_threshold),
            exit_threshold=float(exit_threshold),
            stop_threshold=float(stop_threshold),
            max_holding_days=int(max_holding_days),
        )
        execution_config = ExecutionConfig(
            initial_capital=float(initial_capital),
            commission_bps_per_side=float(commission_bps),
            slippage_bps_per_side=float(slippage_bps),
            annual_borrow_rate=float(borrow_rate),
        )
        _display_backtest(
            market,
            pair_model,
            signal_config,
            execution_config,
            test_start,
            test_end,
            int(refit_frequency),
            formation_window,
            min_observations,
        )


if __name__ == "__main__":
    main()
