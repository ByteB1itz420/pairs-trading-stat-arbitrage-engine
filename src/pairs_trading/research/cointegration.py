"""OLS hedge-ratio estimation and cointegration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

from pairs_trading.data.validation import align_pair


@dataclass(frozen=True)
class PairAnalysis:
    symbol_a: str
    symbol_b: str
    observations: int
    intercept: float | None = None
    hedge_ratio: float | None = None
    engle_granger_statistic: float | None = None
    engle_granger_pvalue: float | None = None
    engle_granger_critical_values: tuple[float, float, float] | None = None
    adf_statistic: float | None = None
    adf_pvalue: float | None = None
    adf_lags: int | None = None
    adf_diagnostic_pass: bool | None = None
    adjusted_pvalue: float | None = None
    cointegrated: bool = False
    eligible: bool = True
    exclusion_reason: str | None = None


def analyze_pair(
    adjusted_close: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    min_observations: int = 100,
    adf_significance: float = 0.05,
) -> PairAnalysis:
    """Analyze one direction of a pair; callers should choose order deterministically."""
    if min_observations < 20:
        raise ValueError("At least 20 observations are required for cointegration diagnostics.")
    if not 0 < adf_significance < 1:
        raise ValueError("adf_significance must be between 0 and 1.")
    pair = align_pair(adjusted_close, symbol_a, symbol_b)
    symbol_a, symbol_b = pair.columns
    nobs = len(pair)
    if nobs < min_observations:
        return PairAnalysis(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            observations=nobs,
            eligible=False,
            exclusion_reason=f"requires {min_observations} aligned observations",
        )

    if not np.isfinite(pair.to_numpy(dtype=float)).all() or (pair <= 0).any().any():
        raise ValueError(f"Pair {symbol_a}/{symbol_b} must have finite, positive prices.")
    log_a = np.log(pair[symbol_a])
    log_b = np.log(pair[symbol_b])
    design = sm.add_constant(log_b, has_constant="add")
    regression = sm.OLS(log_a, design, missing="raise").fit()
    intercept = float(regression.params.iloc[0])
    hedge_ratio = float(regression.params.iloc[1])
    residual = regression.resid

    eg_stat, eg_pvalue, eg_critical = coint(
        log_a,
        log_b,
        trend="c",
        method="aeg",
        maxlag=None,
        autolag="aic",
    )
    adf_stat, adf_pvalue, adf_lags, _, _, _ = adfuller(
        residual,
        regression="n",
        autolag="AIC",
    )

    return PairAnalysis(
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        observations=nobs,
        intercept=intercept,
        hedge_ratio=hedge_ratio,
        engle_granger_statistic=float(eg_stat),
        engle_granger_pvalue=float(eg_pvalue),
        engle_granger_critical_values=tuple(float(value) for value in eg_critical),
        adf_statistic=float(adf_stat),
        adf_pvalue=float(adf_pvalue),
        adf_lags=int(adf_lags),
        adf_diagnostic_pass=bool(adf_pvalue <= adf_significance),
    )
