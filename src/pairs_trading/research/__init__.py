"""Pair discovery and cointegration research."""

from pairs_trading.research.cointegration import (
    PairAnalysis,
    analyze_pair,
    estimate_ols_parameters,
)
from pairs_trading.research.screening import adjust_pvalues_bh, screen_pairs

__all__ = [
    "PairAnalysis",
    "adjust_pvalues_bh",
    "analyze_pair",
    "estimate_ols_parameters",
    "screen_pairs",
]
