# Project Status

**Updated:** 2026-09-29  
**Repository:** [pairs-trading-stat-arbitrage-engine](https://github.com/ByteB1itz420/pairs-trading-stat-arbitrage-engine)  
**State:** Local research engine and redesigned Streamlit workbench are complete. The static frontend is live on Vercel and the Python backtest API is live on Railway.

- **Vercel:** https://site-tan-nine-69.vercel.app
- **Railway health:** https://pairs-trading-api-production.up.railway.app/healthz
- **Railway API docs:** https://pairs-trading-api-production.up.railway.app/docs

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
- Responsive static Netlify/Vercel website with client-side CSV loading, four SVG charts, and an explicit optional API-backtest action.
- FastAPI `/healthz` and `/api/v1/backtest` endpoints for training-only Engle–Granger screening and cost-aware out-of-sample simulation.
- Railway `Dockerfile`/configuration, request body and row limits, optional API-token checks, allowlisted CORS, and no market-data file persistence.
- Vercel static output config, in addition to the Netlify publish configuration.
- Short branded 720p MP4 product tour and SVG storyboard linked from the README. The requested `brag` CLI was unavailable; the video was rendered with local FFmpeg.
- Synthetic static-site demo uses consecutive UTC weekdays, matching its daily-session chart labels.
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
- Railway API source is connected to GitHub `main`; configure the Vercel GitHub app's repository access for automatic frontend deploys.
- Add stronger rate limiting/abuse monitoring and review account quotas before broad public promotion.
- **Why data appeared not to load:** neither UI auto-downloads prices. Streamlit uses generated synthetic data until CSV selection; the static site starts with a synthetic business-day demo. Chart CSVs need `timestamp`, `symbol`, and `adjusted_close`; a Python backtest additionally requires valid `open` values. Browser-local data is sent to Railway only after pressing **Screen & backtest**.

## Run the project locally

Install development, dashboard, and API dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,plots,dashboard,backend]"
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

- Full Python suite: 42 tests passed.
- API coverage: deterministic cointegrated pair and chronological test split, API-key rejection when configured, health response, and oversized-body rejection.
- Live browser-to-local-API smoke test passed on generated data: Engle–Granger p-value `0.0000990`, 55 held-out sessions, and 3 completed trades.
- At a 390-pixel viewport, all charts and the API panel/results fit without horizontal overflow (375-pixel document width).
- `node --check site/app.js`, Vercel/Railway JSON parsing, `git diff --check`, `pip check`, and Pylance API diagnostics passed.
- MP4 verified as H.264 at 1280×720, 10.97 seconds, 203 KB.
- Production Vercel responds HTTP 200 over HTTPS; the website defaults to the Railway API base URL.
- Railway health returns `{"status":"ok"}`. Startup logs showed Railway assigned port 8080, so the public service domain was routed to 8080 instead of 8000.
- CORS preflight from the Vercel production origin succeeds with `X-API-Key` allowed.
- A valid production request without the API key returns HTTP 401. With the API key retrieved in-process from Railway (never printed), a production backtest completed HTTP 200 over 40 out-of-sample sessions and returned 2 synthetic trades.
- Production URL/default API changes were pushed to GitHub and redeployed to Vercel; Railway is linked to GitHub `main` for future deployment triggers.

No real market data was used for performance claims.

## Cloud deployment access

The user authorized Vercel and Railway CLI sessions. The API key remains only in Railway service variables; retrieve it through Railway when running a backtest on the public page. Never paste credentials into chat or commit them. Supabase is not configured or needed: the API is stateless and does not persist uploaded data or results.
