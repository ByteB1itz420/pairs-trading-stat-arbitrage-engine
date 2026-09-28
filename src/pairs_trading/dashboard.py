"""Local visual workbench for trying pair screening and backtests."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pairs_trading.backtest.engine import BacktestResult, run_backtest
from pairs_trading.config import ExecutionConfig, SignalConfig
from pairs_trading.data.validation import MarketData, validate_market_data
from pairs_trading.demo import make_demo_market
from pairs_trading.reporting.metrics import calculate_metrics
from pairs_trading.research.cointegration import PairAnalysis, analyze_pair
from pairs_trading.research.screening import screen_pairs


def _load_uploaded_market(uploaded_file, min_observations: int) -> MarketData:
    observations = pd.read_csv(uploaded_file)
    if "open" not in observations.columns:
        raise ValueError(
            "CSV is missing the required 'open' column. "
            "Backtests need next-session open prices for realistic signal execution."
        )
    return validate_market_data(observations, min_observations=min_observations)


def _dataframe_download(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=True).encode("utf-8")


def _style() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #18332f;
          --muted: #71817b;
          --paper: #f5f7f4;
          --card: #ffffff;
          --line: #e6ebe6;
          --green: #087f65;
          --mint: #d9f2e9;
          --coral: #e87963;
          --gold: #d8a94f;
        }
        .stApp { background: var(--paper); color: var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { background: #edf2ee; border-right: 1px solid var(--line); }
        .block-container { max-width: 1440px; padding: 2rem clamp(1rem, 4vw, 3.5rem) 4rem; }
        h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.035em; }
        h1 { font-weight: 720; }
        [data-testid="stMetric"] {
          background: var(--card); border: 1px solid var(--line); border-radius: 18px;
          padding: 1rem 1.1rem; box-shadow: 0 4px 18px rgba(24, 51, 47, .035);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--ink); }
        [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
          background: var(--card); border: 1px solid var(--line); border-radius: 18px;
          padding: .5rem; overflow: hidden;
        }
        [data-testid="stFileUploader"], [data-testid="stExpander"] {
          background: var(--card); border: 1px solid var(--line); border-radius: 16px;
        }
        .hero {
          padding: clamp(1.4rem, 4vw, 2.7rem); border-radius: 28px;
          background: linear-gradient(125deg, #e0f1e8 0%, #f5eee0 100%);
          margin-bottom: 1.5rem; border: 1px solid rgba(8, 127, 101, .08);
        }
        .eyebrow { color: var(--green); font-size: .76rem; font-weight: 700;
          letter-spacing: .12em; text-transform: uppercase; }
        .hero p { color: var(--muted); max-width: 760px; margin-bottom: 0; }
        div.stButton > button[kind="primary"] {
          border: 0; border-radius: 14px; background: var(--green); color: white;
          min-height: 3rem; font-weight: 650; padding-inline: 1.3rem;
        }
        div.stButton > button[kind="primary"]:hover { background: #066b56; color: white; }
        @media (max-width: 760px) {
          .block-container { padding: 1rem .8rem 2.5rem; }
          .hero { border-radius: 20px; }
          [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .65rem; }
          [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            flex: 1 1 100% !important; min-width: min(100%, 280px) !important;
          }
          [data-testid="stMetric"] { padding: .75rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _plot_layout(figure: go.Figure, height: int = 320) -> go.Figure:
    figure.update_layout(
        height=height,
        margin={"l": 18, "r": 18, "t": 28, "b": 16},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, ui-sans-serif, system-ui, sans-serif", "color": "#71817b"},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
        xaxis={"showgrid": False, "zeroline": False},
        yaxis={"gridcolor": "#edf0ed", "zerolinecolor": "#dfe6e0"},
    )
    return figure


def _draw_charts(
    market: MarketData,
    pair: PairAnalysis,
    result: BacktestResult,
    signal_config: SignalConfig,
) -> None:
    dates = result.equity_curve.index
    palette = {
        "net": "#087f65",
        "gross": "#88b8a7",
        "cash": "#c5cfc8",
        "coral": "#e87963",
        "gold": "#d8a94f",
        "blue": "#6688bd",
    }

    st.markdown("#### Portfolio")
    portfolio_figure = go.Figure()
    portfolio_figure.add_trace(
        go.Scatter(
            x=dates,
            y=result.equity_curve["net_equity"],
            name="Net",
            mode="lines",
            line={"color": palette["net"], "width": 3},
            fill="tozeroy",
            fillcolor="rgba(8,127,101,.07)",
        )
    )
    portfolio_figure.add_trace(
        go.Scatter(
            x=dates,
            y=result.equity_curve["gross_equity"],
            name="Gross, before costs",
            mode="lines",
            line={"color": palette["gross"], "width": 2, "dash": "dot"},
        )
    )
    portfolio_figure.add_trace(
        go.Scatter(
            x=dates,
            y=result.equity_curve["cash_benchmark"],
            name="Cash benchmark",
            mode="lines",
            line={"color": palette["cash"], "width": 1.5},
        )
    )
    _plot_layout(portfolio_figure, 340)
    st.plotly_chart(portfolio_figure, width="stretch", config={"displayModeBar": False})

    columns = st.columns(2)
    with columns[0]:
        st.markdown("#### Drawdown")
        high_water = result.equity_curve["net_equity"].cummax()
        drawdown = result.equity_curve["net_equity"] / high_water - 1
        figure = go.Figure(
            go.Scatter(
                x=dates,
                y=drawdown,
                name="Drawdown",
                mode="lines",
                line={"color": palette["coral"], "width": 2},
                fill="tozeroy",
                fillcolor="rgba(232,121,99,.14)",
            )
        )
        _plot_layout(figure, 280)
        figure.update_yaxes(tickformat=".0%")
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    with columns[1]:
        st.markdown("#### Gross exposure")
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=result.equity_curve["gross_exposure"],
                name="Gross exposure",
                mode="lines",
                line={"color": palette["blue"], "width": 2},
                fill="tozeroy",
                fillcolor="rgba(102,136,189,.12)",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=result.equity_curve["net_exposure"],
                name="Net exposure",
                mode="lines",
                line={"color": palette["gold"], "width": 1.5},
            )
        )
        _plot_layout(figure, 280)
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    st.markdown("#### Pair prices (rebased to 100)")
    test_close = market.adjusted_close.loc[dates[0] : dates[-1], [pair.symbol_a, pair.symbol_b]]
    if not test_close.empty:
        rebased = test_close.div(test_close.iloc[0]).mul(100)
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=rebased.index,
                y=rebased[pair.symbol_a],
                name=pair.symbol_a,
                line={"color": palette["net"], "width": 2},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=rebased.index,
                y=rebased[pair.symbol_b],
                name=pair.symbol_b,
                line={"color": palette["blue"], "width": 2},
            )
        )
        _plot_layout(figure, 300)
        figure.update_yaxes(title="Index")
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    st.markdown("#### Spread and signal bands")
    zscore_figure = make_subplots(specs=[[{"secondary_y": True}]])
    zscore_figure.add_trace(
        go.Scatter(
            x=result.spread.index,
            y=result.spread,
            name="Spread",
            line={"color": palette["blue"], "width": 1.5},
        ),
        secondary_y=False,
    )
    zscore_figure.add_trace(
        go.Scatter(
            x=result.zscore.index,
            y=result.zscore,
            name="Z-score",
            line={"color": palette["net"], "width": 2},
        ),
        secondary_y=True,
    )
    for threshold, color, label in (
        (signal_config.entry_threshold, palette["coral"], "Entry"),
        (-signal_config.entry_threshold, palette["coral"], None),
        (signal_config.exit_threshold, palette["gold"], "Exit"),
        (-signal_config.exit_threshold, palette["gold"], None),
    ):
        zscore_figure.add_hline(
            y=threshold,
            line={"color": color, "dash": "dot", "width": 1},
            annotation_text=label,
            secondary_y=True,
        )
    action_markers = {
        "enter_long": ("triangle-up", palette["net"]),
        "enter_short": ("triangle-down", palette["coral"]),
        "exit": ("x", palette["gold"]),
    }
    for action, (marker, color) in action_markers.items():
        points = [signal for signal in result.signals if signal.action.value == action and signal.zscore is not None]
        if points:
            zscore_figure.add_trace(
                go.Scatter(
                    x=[signal.timestamp for signal in points],
                    y=[signal.zscore for signal in points],
                    name=action.replace("_", " ").title(),
                    mode="markers",
                    marker={"symbol": marker, "color": color, "size": 10},
                ),
                secondary_y=True,
            )
    _plot_layout(zscore_figure, 330)
    zscore_figure.update_yaxes(title_text="Spread", secondary_y=False, showgrid=False)
    zscore_figure.update_yaxes(title_text="Z-score", secondary_y=True, gridcolor="#edf0ed")
    st.plotly_chart(zscore_figure, width="stretch", config={"displayModeBar": False})

    st.markdown("#### Monthly net returns")
    equity = result.equity_curve["net_equity"].resample("ME").last()
    monthly_returns = equity.pct_change().dropna()
    if len(monthly_returns):
        monthly = pd.DataFrame(
            {
                "year": monthly_returns.index.year,
                "month": monthly_returns.index.month,
                "return": monthly_returns.to_numpy(),
            }
        )
        pivot = monthly.pivot(index="year", columns="month", values="return").reindex(
            columns=range(1, 13)
        )
        figure = go.Figure(
            go.Heatmap(
                z=pivot.to_numpy() * 100,
                x=[date(2000, month, 1).strftime("%b") for month in range(1, 13)],
                y=[str(year) for year in pivot.index],
                colorscale=[
                    [0, "#e87963"],
                    [0.5, "#f5f7f4"],
                    [1, "#087f65"],
                ],
                zmid=0,
                colorbar={"title": "%"},
                hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
            )
        )
        _plot_layout(figure, max(150, 60 + len(pivot.index) * 42))
        figure.update_xaxes(showgrid=False)
        figure.update_yaxes(showgrid=False, autorange="reversed")
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    else:
        st.caption("Monthly return heatmap appears once the test period includes a completed month.")


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

    _draw_charts(market, pair, result, signal_config)

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
    _style()
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">Research workspace · local only</div>
          <h1>Pairs Trading Lab</h1>
          <p>Explore adjusted-price relationships, test mean-reversion signals, and inspect
          costs and drawdowns. No live orders are sent.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Data source: locally generated synthetic demo or the CSV you explicitly upload. "
        "This app does not automatically fetch market data or read files from your computer."
    )

    source = st.sidebar.radio("Data source", ["Synthetic demo", "Upload CSV"])
    if source == "Synthetic demo":
        market, suggested_test_start = make_demo_market()
    else:
        uploaded_file = st.sidebar.file_uploader("Long-format adjusted-price CSV", type=["csv"])
        if uploaded_file is None:
            st.warning(
                "No dataset is loaded yet. Upload a CSV with one row per symbol/date and "
                "the columns timestamp, symbol, adjusted_close, and open."
            )
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
    missing_close = sum(market.quality.missing_adjusted_close.values())
    missing_open = sum(market.quality.missing_open.values())
    st.success(
        f"Loaded {market.quality.observations:,} rows · "
        f"{len(market.quality.symbols)} symbols · {len(index):,} sessions · "
        f"{index.min().date()} to {index.max().date()}"
    )
    if missing_close or missing_open:
        st.warning(
            f"Data quality: {missing_close} missing adjusted-close cells and "
            f"{missing_open} missing open cells. Pair observations are not forward-filled."
        )
    with st.expander("Preview loaded prices"):
        st.dataframe(
            market.adjusted_close.tail(8).reset_index(),
            width="stretch",
            hide_index=True,
        )
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
        st.dataframe(screen_table, width="stretch")
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
