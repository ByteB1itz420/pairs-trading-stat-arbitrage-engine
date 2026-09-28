"""Reconcile the committed base and severe-cost test artifacts without raw quotes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parent / "real_data_snapshot"
SCENARIOS = ("test_cost1x_borrow0pct", "test_cost3x_borrow8pct")


def audit(root: Path) -> list[dict[str, object]]:
    """Check ledger, fills, costs, exposure dates and unchanged gross counterfactual."""
    outcomes: list[dict[str, object]] = []
    for name in SCENARIOS:
        folder = root / name
        metrics = json.loads((folder / "metrics.json").read_text())
        trades = pd.read_csv(folder / "trades.csv")
        fills = pd.read_csv(folder / "fills.csv")
        equity = pd.read_csv(folder / "equity_curve.csv")
        assert len(trades) == metrics["trade_count"] > 0, name
        assert len(fills) == 4 * len(trades), name
        assert len(equity) > 0 and equity.timestamp.is_unique, name
        assert equity.timestamp.is_monotonic_increasing, name
        assert equity.timestamp.iloc[0] == "2023-01-03", name
        assert equity.timestamp.iloc[-1] == "2025-12-31", name
        np.testing.assert_allclose(
            [trades.net_pnl.sum(), trades.total_costs.sum(),
             trades.gross_pnl_before_costs.sum()],
            [metrics["ending_net_equity"] - metrics["starting_capital"],
             metrics["total_commissions"] + metrics["total_slippage_costs"]
             + metrics["total_borrow_costs"],
             metrics["ending_gross_equity"] - metrics["starting_capital"]],
            rtol=0, atol=0.01,
        )
        np.testing.assert_allclose(
            [equity.net_equity.iloc[-1], equity.gross_equity.iloc[-1],
             fills.commission.sum(), fills.slippage_cost.sum(),
             equity.borrow_costs.iloc[-1]],
            [metrics["ending_net_equity"], metrics["ending_gross_equity"],
             metrics["total_commissions"], metrics["total_slippage_costs"],
             metrics["total_borrow_costs"]],
            rtol=0, atol=0.01,
        )
        outcomes.append({"scenario": name, "trades": len(trades),
                         "net_return": metrics["cumulative_net_return"],
                         "gross_return": metrics["cumulative_gross_return"],
                         "borrow_cost": metrics["total_borrow_costs"]})
    assert abs(outcomes[0]["gross_return"] - outcomes[1]["gross_return"]) < 1e-10
    assert outcomes[1]["net_return"] < outcomes[0]["net_return"]
    return outcomes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    for result in audit(parser.parse_args().root):
        print(json.dumps(result, sort_keys=True))
