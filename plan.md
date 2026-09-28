# Pairs Trading & Statistical Arbitrage Engine — Project Plan

**Project window:** May 2026 – July 2026  
**Goal:** Build a modular Python research and backtesting engine for identifying candidate cointegrated asset pairs, generating rolling z-score signals, and evaluating those signals in a cost-aware, event-driven, out-of-sample simulation.

## Current implementation status

The repository now has an installable Python package, YAML configuration, CSV price validation, OLS/Engle–Granger/ADF screening with Benjamini–Hochberg correction, rolling signals, a single-pair event-driven next-open simulator, scheduled trailing OLS refits while flat, transaction/borrow costs, a chronological training/test runner, performance metrics, CSV/JSON exports, plots, a local Streamlit test workbench, and unit/integration tests. Run `python -m pytest -q` and consult the README for setup and the configured experiment command.

The initial release intentionally does not claim every research-hardening item in this roadmap is complete. Explicit follow-up work includes a separate validation split and parameter-sensitivity workflow, point-in-time survivorship/delisting data, and portfolio-level handling for multiple overlapping pairs. Use a frozen, untouched test period for any reported strategy result.

## 1. Project outcomes

At completion, the project should provide:

1. A reproducible data-to-results workflow with explicit assumptions and configuration.
2. A pair screening pipeline using Engle–Granger cointegration testing, ADF stationarity checks, and OLS hedge-ratio estimation.
3. A rolling spread and z-score signal model with entry, exit, and risk-control rules.
4. An event-driven backtester that processes market data in time order and accounts for trades, positions, cash, commissions, and slippage.
5. An out-of-sample evaluation report, including benchmark comparisons and diagnostics that help distinguish a robust result from an artifact.
6. Unit and integration tests for statistical calculations, signal transitions, and portfolio accounting.

This is a research and engineering project, not a promise of profitability or investment advice. Statistical significance and historical returns do not guarantee future performance.

## 2. Scope and assumptions

### In scope

- Historical adjusted price data for liquid equities or ETFs.
- A configurable universe and sampling frequency (start with daily bars).
- Pair discovery and periodic re-estimation using data available only at the time of each decision.
- Long/short spread positions, position tracking, transaction costs, and out-of-sample backtests.
- Reproducible experiment configuration and machine-readable outputs.

### Out of scope for the first version

- Live brokerage connectivity or automated order placement.
- Intraday execution, market impact modeling beyond a configurable slippage approximation, and complex borrow availability feeds.
- Portfolio optimization across a large number of overlapping pairs.
- Claims that a pair is permanently cointegrated or that backtest returns will persist.

### Initial modeling choices

- Prices are adjusted for splits and distributions; data source, retrieval date, and symbol mapping are recorded.
- Use log prices for the Engle–Granger regression and define the spread consistently as `log(P_A) - hedge_ratio * log(P_B) - intercept`.
- Use a training-only calibration window for hedge ratios and z-score parameters. Do not fit or normalize using future observations.
- Begin with daily close-to-close bars. A signal computed after a close is eligible for execution no earlier than the next bar/open, according to the configured execution convention.
- Keep all key choices (lookback windows, thresholds, costs, re-fit cadence, and date splits) configurable rather than hard-coded.

## 3. Proposed technology and repository layout

Use a small, conventional Python stack:

- **Python 3.11+** for implementation.
- **pandas / NumPy** for time-series and numerical operations.
- **statsmodels** for OLS, ADF, and Engle–Granger (`coint`) tests.
- **pytest** for automated tests.
- **matplotlib** for diagnostic plots; optional reporting can use plain CSV/JSON and a notebook.
- **pyproject.toml** for package metadata, dependency declarations, and tool configuration.

Suggested layout:

```text
pairs-trading-stat-arbitrage-engine/
├── plan.md
├── pyproject.toml
├── README.md
├── configs/
│   └── baseline.yaml
├── data/
│   ├── raw/                 # ignored by version control
│   └── processed/           # ignored by version control
├── src/
│   └── pairs_trading/
│       ├── cli.py
│       ├── config.py
│       ├── data/
│       │   ├── loader.py
│       │   └── validation.py
│       ├── research/
│       │   ├── screening.py
│       │   ├── cointegration.py
│       │   └── spread.py
│       ├── signals/
│       │   └── zscore.py
│       ├── backtest/
│       │   ├── events.py
│       │   ├── portfolio.py
│       │   ├── execution.py
│       │   └── engine.py
│       └── reporting/
│           ├── metrics.py
│           └── outputs.py
├── tests/
│   ├── unit/
│   └── integration/
└── examples/
    └── run_baseline.py
```

Keep market-data acquisition behind a loader interface so the rest of the engine can use local fixtures or another provider without changing the strategy code. Do not commit licensed data, API keys, or credentials.

## 4. System design

### 4.1 Data layer

Responsibilities:

- Load adjusted OHLCV or adjusted close series from a documented source.
- Normalize timestamps and symbols, sort observations, and reject duplicate timestamps.
- Align pair observations on a common calendar without silently forward-filling missing prices.
- Report missingness, stale observations, non-positive prices, and insufficient history.
- Preserve a clear distinction between raw observations and transformed returns/log prices.

Outputs should include a validated price table indexed by timestamp and symbol, plus a data-quality summary. Every experiment should record its input date range, source, symbols, frequency, and configuration.

### 4.2 Pair screening and statistical estimation

For each eligible pair within the candidate universe:

1. Restrict the sample to the current training window and require a minimum number of aligned observations.
2. Estimate the OLS regression `log(P_A) = intercept + hedge_ratio * log(P_B) + residual`.
3. Apply the Engle–Granger two-step cointegration test to the pair, using a declared deterministic trend/intercept choice and significance level.
4. Run an ADF test on the estimated residual/spread and record the test statistic, p-value, lag choice, and sample length.
5. Apply configurable liquidity, price-history, and statistical filters. Correct for multiple testing when screening many pairs (for example, report adjusted p-values using Benjamini–Hochberg).
6. Save the full screening table, not only selected pairs, so the result is auditable.

The order of the Engle–Granger regression is not symmetric. Choose and document a deterministic ordering (for example, alphabetical symbol order) or explicitly evaluate both directions and account for the extra testing. Do not treat an ordinary ADF p-value on fitted residuals as a substitute for the Engle–Granger test's appropriate critical values.

Re-estimate the hedge ratio on a scheduled basis using past data only. Compare fixed and rolling estimation as separate experiments rather than changing the rule after seeing test-period performance.

### 4.3 Spread and signal model

For the fitted parameters at a given decision time:

- Compute spread observations using the defined log-price residual.
- Compute a rolling mean and standard deviation using trailing observations only.
- Calculate `z_t = (spread_t - rolling_mean_t) / rolling_std_t`.
- Mark z-scores as unavailable when the window is incomplete or standard deviation is zero/near-zero; do not generate trades from invalid values.

Baseline state machine:

- If flat and `z_t >= entry_threshold`, enter a short-spread position (short A, long hedge-ratio-weighted B).
- If flat and `z_t <= -entry_threshold`, enter a long-spread position (long A, short hedge-ratio-weighted B).
- Exit when the spread reverts through `exit_threshold` around zero.
- Apply a stop threshold, maximum holding period, and/or end-of-window liquidation rule as explicit configurable risk controls.
- Do not reverse directly from one side to the other without first closing the existing position.

Document whether the hedge ratio is interpreted in log-return exposure or translated to share quantities, and define gross exposure and capital allocation. Normalize pair sizing to a configurable gross-notional budget so pairs with different prices are comparable.

### 4.4 Event-driven backtesting

Represent the simulation as a chronological event loop. A minimal event flow is:

1. **Market event:** expose only the bar currently being processed.
2. **Strategy decision:** update eligible rolling statistics and emit an order intent from the current strategy state.
3. **Execution event:** fill orders according to the configured next-bar pricing rule and cost model.
4. **Portfolio update:** update cash, positions, entry prices, realized/unrealized P&L, and fees.
5. **Valuation event:** mark positions to market and append portfolio/equity records.

Keep signal generation separate from fill simulation so each can be tested independently. Record every signal, order, fill, fee, position transition, and rejected order with timestamps and reasons.

Minimum cost model:

- Per-side commission in basis points or per share.
- Slippage in basis points applied at execution.
- Optional short-borrow cost accrued by holding period.
- Costs applied on both entry and exit and deducted from portfolio P&L.

Never fill at a price that was unavailable when the signal was produced. State the convention for close signals (for example, decide on close `t`, execute on open `t+1`) and enforce it in tests. Handle delisted or missing assets explicitly rather than silently carrying a position at a fabricated price.

### 4.5 Evaluation and reporting

Report gross and net performance, at minimum:

- Cumulative return and annualized return.
- Annualized volatility and Sharpe ratio (with the risk-free-rate assumption stated).
- Maximum drawdown and drawdown duration.
- Turnover, trade count, average holding period, win rate, and average win/loss.
- Total commissions, slippage, and borrow costs.
- Exposure and time-in-market.
- Pair-level and portfolio-level results.

Include a buy-and-hold or cash baseline as appropriate, and compare net-of-cost results against a no-cost run to show cost sensitivity. Provide plots for equity, drawdown, rolling performance, spread/z-score, and trade markers. Export summary metrics and trade/fill logs to CSV or JSON.

## 5. Work plan and milestones

### Phase 1 — Requirements and reproducible setup (May 2026, week 1)

- Finalize universe, data source, frequency, date range, and baseline assumptions.
- Establish package layout, dependency management, configuration schema, and test runner.
- Define data schema, timestamp convention, and result artifact format.
- Add a small deterministic fixture dataset for tests.

**Deliverable:** runnable project skeleton, baseline config, documented assumptions, and CI-ready tests.

### Phase 2 — Data ingestion and validation (May 2026, weeks 2–3)

- Implement loader interface and local file loader.
- Validate calendars, duplicates, missing values, adjusted prices, and history length.
- Add quality summaries and deterministic train/validation/test date slicing.
- Ensure symbol membership and data availability are evaluated as of each date where possible.

**Deliverable:** validated aligned price data and tests for malformed or incomplete inputs.

### Phase 3 — Pair research and cointegration (May 2026, weeks 4–5)

- Implement OLS hedge-ratio estimation and residual construction.
- Add Engle–Granger and ADF statistics with complete result records.
- Build a candidate pair enumerator and configurable filters.
- Add multiple-testing correction for universe-wide screening.
- Test on synthetic data with known stationary and non-stationary relationships.

**Deliverable:** repeatable screening table with diagnostics and unit-tested statistical functions.

### Phase 4 — Rolling features and strategy state machine (June 2026, weeks 1–2)

- Implement trailing spread mean, standard deviation, and z-score.
- Add state transitions for entry, exit, stop, maximum holding period, and invalid-data handling.
- Define position sizing and translate hedge ratios into target quantities.
- Test signal timing, boundary conditions, and state transitions without using future observations.

**Deliverable:** deterministic signal stream and trade intents for a configured pair.

### Phase 5 — Event-driven simulator and costs (June 2026, weeks 3–4)

- Implement event types, order intents, execution policy, portfolio ledger, and mark-to-market.
- Apply commissions and slippage per fill; add configurable borrow costs.
- Enforce execution timing and record every event for auditability.
- Test cash/position accounting, long and short legs, partial/missing bars, and cost deductions.

**Deliverable:** end-to-end single-pair backtest with a complete trade and portfolio ledger.

### Phase 6 — Out-of-sample experiments and diagnostics (July 2026, weeks 1–2)

- Freeze a chronological train/validation/test split before evaluating strategy performance.
- Select pairs and fit parameters on training data only; use validation data for limited model selection.
- Evaluate the frozen strategy on the untouched test period.
- Add rolling or walk-forward evaluation as a separate robustness check; re-fit only at declared intervals.
- Run cost sensitivity, threshold sensitivity, and parameter-stability analyses without selecting on the final test set.

**Deliverable:** out-of-sample results, trade logs, plots, and a reproducible experiment configuration.

### Phase 7 — Hardening, documentation, and final review (July 2026, weeks 3–4)

- Run the complete test suite and resolve look-ahead, accounting, and reproducibility issues.
- Document installation, data format, configuration, command usage, limitations, and an example run.
- Verify that a clean run from saved configuration reproduces the same outputs.
- Review survivorship bias, multiple testing, overlapping positions, and execution assumptions.

**Deliverable:** documented, tested research engine and a concise results/limitations report.

## 6. Validation strategy

### Unit tests

- OLS recovers a known hedge ratio on generated data.
- Cointegrated synthetic series are distinguished from unrelated random walks within statistical-test limitations.
- Rolling calculations use only the trailing window and correctly handle warm-up periods and zero variance.
- Entry/exit/stop transitions occur at exact configured boundaries.
- Fees and slippage reduce net P&L by the expected amount.
- Portfolio cash, quantities, and marked value reconcile after buys, sells, and short sales.

### Integration tests

- A small synthetic dataset runs through ingestion, screening, signal generation, execution, and reporting.
- Execution occurs only after the signal timestamp under the chosen convention.
- Re-running a fixed configuration and fixture data produces identical trades and metrics.
- Missing data or invalid prices produce explicit validation results or errors rather than silently fabricated fills.

### Research checks

- Compare in-sample and out-of-sample behavior and report both.
- Assess sensitivity to transaction costs, entry/exit thresholds, estimation windows, and re-fit frequency.
- Check whether results are concentrated in a few pairs or a short interval.
- Track the number of pairs and parameter combinations tested; account for selection bias.
- Compare against simple baselines and inspect individual trades, not just aggregate Sharpe.

## 7. Risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Look-ahead bias | Inflates historical results when future prices or fitted parameters leak into decisions. | Chronological splits, trailing-only calculations, next-bar execution, and timestamp-focused tests. |
| Multiple testing / data snooping | Screening many pairs makes false discoveries likely. | Record all tests, apply multiple-testing correction, limit tuning, and reserve an untouched test period. |
| Survivorship and delisting bias | Current constituents omit failed or removed assets. | Prefer point-in-time universe membership and delisting-aware data; clearly disclose gaps if unavailable. |
| Non-stationary relationship | Cointegration can break after selection. | Re-test on a schedule, monitor residual behavior, impose risk limits, and report stability. |
| Unrealistic fills and costs | Close-price fills and zero costs can overstate returns. | Explicit next-bar execution, commission/slippage/borrow assumptions, and sensitivity analysis. |
| Short-sale constraints | Short legs may be unavailable or expensive. | Model borrow costs and availability when data exists; otherwise state the simplifying assumption. |
| Pair overlap and leverage | Multiple pairs can create concentrated, correlated exposure. | Track gross/net exposure and concentration; cap allocations and avoid assuming independent returns. |
| Data quality and calendar mismatch | Bad alignment can create artificial spreads or signals. | Validate calendars and corporate-action adjustments; never silently forward-fill prices. |

## 8. Definition of done

The project is complete when:

- A user can run the baseline workflow from a documented configuration.
- The same inputs and configuration produce reproducible screening results, trades, and metrics.
- Pair screening uses Engle–Granger/ADF/OLS results calculated only from eligible historical data.
- Rolling z-score signals follow the documented state machine and cannot execute on the signal bar unless explicitly configured.
- The event-driven backtester accounts for both legs, portfolio state, and configured transaction costs.
- An untouched chronological test period is reported separately from model development.
- Automated tests cover statistical functions, signal timing/state transitions, and accounting.
- Output artifacts and limitations are documented, including data-source and execution assumptions.

## 9. Suggested baseline configuration

Treat these as starting values for experiments, not optimized recommendations:

```yaml
data:
  frequency: daily
  price_field: adjusted_close
  min_observations: 252

screening:
  formation_window_days: 252
  engle_granger_significance: 0.05
  adf_significance: 0.05
  multiple_testing_method: benjamini_hochberg
  refit_frequency_days: 21

signals:
  zscore_window_days: 60
  entry_threshold: 2.0
  exit_threshold: 0.5
  stop_threshold: 4.0
  max_holding_days: 30

execution:
  decision_time: close
  fill_time: next_open
  commission_bps_per_side: 2.0
  slippage_bps_per_side: 5.0
  annual_borrow_rate: 0.0

evaluation:
  split_method: chronological
  report_gross_and_net: true
  risk_free_rate_annual: 0.0
```

Before using these defaults, verify that they match the data frequency and chosen price fields. Any change to the signal, sizing, or execution assumptions should be versioned in the experiment configuration.

## 10. Newly prioritized tasks: bug audit and deployable website

### Task A — Find, reproduce, and fix existing bugs

- Maintain a bug register in `status.md` with severity, reproduction evidence, fix, and regression test for each confirmed issue.
- Re-audit configuration validation, pair-screening edge cases, walk-forward parameter flow, event ordering, order rejection/missing data, portfolio/cost accounting, metrics, exports, and dashboard behavior.
- Add minimized regression tests for every confirmed defect and run the full suite after changes.
- Separate reproducible code defects from known model/data limitations; do not label an unverified risk as a confirmed bug.
- **Already found and fixed:** the backtest used the refitted hedge ratio for spread and signals but the original formation-period hedge ratio for sizing execution orders. The order now captures the ratio at signal time, and regression tests verify that the ratio used in the refit drives actual target notionals.
- **Already found and fixed:** non-finite values (`NaN` and infinity) could pass configuration checks for strategy thresholds, costs, capital, and evaluation. Configuration now rejects them; parameterized tests cover the affected settings.
- **Acceptance:** no known reproducible high-severity defects remain; all confirmed findings are either fixed with regression coverage or explicitly recorded as open with a reproduction case.

### Task B — Build and deploy a production-ready website

This follows successful local-workbench testing; the local Streamlit workbench is not itself a production deployment.

1. Confirm the intended hosting provider, access model, and whether uploaded data may leave the user's machine.
2. Define the service boundary: web UI, API/job runner, configuration validation, result storage, and a stateless/isolated execution model.
3. Add user-controlled data upload and deletion, strict size/type limits, non-executable data parsing, and no secret/data logging.
4. Add authentication/authorization if private projects or saved results are supported; otherwise clearly specify the public/demo-only behavior.
5. Add deploy configuration, health checks, dependency/build locking, automated deployment smoke tests, and an operational rollback procedure.
6. Run representative synthetic and user-authorized datasets through the deployed build and compare its results to the local test suite.
- **Acceptance:** a documented deployment can be reproduced from source, is reachable at the chosen host, passes health and end-to-end smoke tests, and has explicit privacy, retention, and failure behavior.
