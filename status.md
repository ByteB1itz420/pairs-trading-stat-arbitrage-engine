# Project Status

**Updated:** 2026-09-29  
**Repository:** [pairs-trading-stat-arbitrage-engine](https://github.com/ByteB1itz420/pairs-trading-stat-arbitrage-engine)  
**State:** Local research engine and visual test workbench implemented; no production website deployed.

## What is implemented

- Installable Python package with YAML experiment configuration.
- Long-format CSV ingestion, timestamp/symbol normalization, duplicate and invalid-price checks, missing-data reports, and pair alignment without forward filling.
- OLS hedge ratios, Engle–Granger cointegration screening, ADF residual diagnostics, deterministic pair ordering, and Benjamini–Hochberg adjusted p-values.
- Trailing rolling spread z-scores and a long/short entry, mean-reversion exit, stop, and max-holding state machine.
- Chronological event-driven backtesting with next-session-open fills, fixed gross-notional sizing, commissions, slippage, borrow charges, cash/position tracking, and explicit final-close liquidation.
- Scheduled trailing OLS refits while flat; refits do not silently rebalance an open trade.
- Net/gross performance metrics, a cash baseline, CSV/JSON exports, and diagnostic plots.
- CLI experiment runner and a local Streamlit test workbench.
- Deterministic synthetic cointegrated data for trying the full flow without downloading market data.
- Unit and integration tests for the statistical core, signal rules, accounting, CLI workflow, refits, and synthetic demo.

## Confirmed bug found and fixed

| Severity | Bug | Evidence / impact | Fix and regression coverage |
|---|---|---|---|
| High | A scheduled OLS refit changed the spread and signal hedge ratio, but target position sizing still used the original formation-period hedge ratio. | The refit path updated `current_pair`, while `ExecutionModel.create_fills` received `self.pair`; the strategy and actual portfolio therefore represented different hedge ratios. | Each `OrderIntent` now snapshots the active hedge ratio. Execution sizes both legs from that snapshot. Unit and integration tests verify sized notionals use the order/refitted ratio. |
| Medium | `NaN` and infinite values could pass configuration checks for signal thresholds, costs, capital, and risk-free rate. | Parameterized tests showed non-finite settings could be accepted, allowing invalid or poisoned sizing/performance values. | Configuration validation now rejects non-finite numeric settings before execution. Five parameterized cases cover thresholds, costs, capital, and evaluation settings. |

The audit is ongoing; these are the confirmed reproducible defects found and fixed so far. Remaining model/data limitations are not asserted to be code bugs.

## Open limitations and follow-up

- Add a separate validation period and parameter-sensitivity/walk-forward research protocol.
- Obtain point-in-time universe membership and delisting-aware data to address survivorship bias.
- Extend from one selected pair per run to portfolio-level capital allocation, pair overlap, leverage, and exposure controls.
- Add borrow availability, margin rules, realistic partial fills, and market-impact modeling before interpreting results as executable performance.
- Perform the broader targeted bug-audit task described in the next section of [plan.md](./plan.md).
- Design, secure, deploy, and smoke-test a production website only after local-workbench feedback and explicit hosting/privacy decisions. The Streamlit app is local-only.

## Run the project locally

Install development and dashboard dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,plots,dashboard]"
```

Run all tests:

```bash
python -m pytest -q
```

Launch the visual test workbench:

```bash
streamlit run src/pairs_trading/dashboard.py
```

Choose **Synthetic demo** for a deterministic end-to-end example. For actual historical data, select **Upload CSV** and provide licensed long-format data containing `timestamp`, `symbol`, `adjusted_close`, and `open`. The test period starts after the training cutoff. No market dataset or credentials are bundled with the repository.

To use the CLI, set `evaluation.train_end` and `evaluation.test_start` in `configs/baseline.yaml`, configure the data path and universe, then run:

```bash
python -m pairs_trading --config configs/baseline.yaml --output results/baseline
```

## Verification performed for this update

- Full suite: **35 unit/integration/UI tests pass** after the dashboard and regression tests were added.
- Streamlit `AppTest` rendered the workbench, ran the synthetic test, and displayed all four headline metrics without app exceptions.
- Browser verification: the local dashboard displayed screening results, 3 synthetic-demo trades, gross/net curves, drawdown, and downloadable artifacts. The displayed synthetic results are a UI smoke test, not evidence of strategy performance.
- Regression checks: refitted hedge ratios drive actual execution sizing; `NaN` and infinite configuration values are rejected.
- Package dependency check: `pip check` found no broken requirements.
- Local UI dependency: Streamlit installed and version command verified.

No real market data was used for performance claims.
