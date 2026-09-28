"""Deterministic universe pair screening and false-discovery-rate adjustment."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np
import pandas as pd

from pairs_trading.research.cointegration import PairAnalysis, analyze_pair


def adjust_pvalues_bh(pvalues: list[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted p-values in their original order."""
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("pvalues must be a one-dimensional sequence of finite values.")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("pvalues must be between 0 and 1.")
    count = len(values)
    if count == 0:
        return []

    order = np.argsort(values, kind="stable")
    ranked = values[order] * count / np.arange(1, count + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def screen_pairs(
    adjusted_close: pd.DataFrame,
    symbols: list[str] | tuple[str, ...] | None = None,
    formation_window_days: int = 252,
    min_observations: int = 100,
    significance: float = 0.05,
    adf_significance: float = 0.05,
) -> list[PairAnalysis]:
    """Test each alphabetically ordered pair once and retain every candidate result."""
    if formation_window_days < 20 or min_observations < 20:
        raise ValueError("Formation and minimum-history windows must be at least 20.")
    if min_observations > formation_window_days:
        raise ValueError("min_observations cannot exceed formation_window_days.")
    if not 0 < significance < 1 or not 0 < adf_significance < 1:
        raise ValueError("Significance levels must be between 0 and 1.")

    candidates = sorted(
        {
            symbol.strip().upper()
            for symbol in (adjusted_close.columns if symbols is None else symbols)
        }
    )
    missing = set(candidates) - set(adjusted_close.columns)
    if missing:
        raise ValueError(f"Unknown screening symbols: {', '.join(sorted(missing))}.")

    # Fix the shared formation dates first. Dropping a missing pair observation
    # must not extend that pair's lookback farther into the past.
    formation = adjusted_close.tail(formation_window_days)
    results: list[PairAnalysis] = []
    for symbol_a, symbol_b in combinations(candidates, 2):
        pair = formation.loc[:, [symbol_a, symbol_b]].dropna(how="any")
        exclusion_reason = None
        if len(pair) < min_observations:
            exclusion_reason = f"requires {min_observations} aligned observations in formation window"
        elif any(np.log(pair[symbol]).std() < 1e-10 for symbol in (symbol_a, symbol_b)):
            exclusion_reason = "constant price series in formation window"
        if exclusion_reason is not None:
            results.append(
                PairAnalysis(symbol_a=symbol_a, symbol_b=symbol_b,
                             observations=len(pair), eligible=False,
                             exclusion_reason=exclusion_reason)
            )
            continue
        results.append(
            analyze_pair(
                pair,
                symbol_a,
                symbol_b,
                min_observations=min_observations,
                adf_significance=adf_significance,
            )
        )

    valid_indices = [
        index
        for index, result in enumerate(results)
        if result.eligible and result.engle_granger_pvalue is not None
    ]
    adjusted_values = adjust_pvalues_bh(
        [float(results[index].engle_granger_pvalue) for index in valid_indices]
    )
    for index, adjusted_pvalue in zip(valid_indices, adjusted_values, strict=True):
        result = results[index]
        results[index] = replace(
            result,
            adjusted_pvalue=adjusted_pvalue,
            cointegrated=(
                result.engle_granger_pvalue <= significance
                and adjusted_pvalue <= significance
            ),
        )
    return results
