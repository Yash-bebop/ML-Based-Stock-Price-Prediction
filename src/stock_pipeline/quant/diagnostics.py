"""Stationarity and residual diagnostics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


def stationarity_report(returns: pd.Series) -> Dict[str, float]:
    """Run the ADF stationarity test on log returns."""

    clean = returns.dropna()
    stat, pvalue, used_lag, nobs, *_ = adfuller(clean)
    return {
        "adf_statistic": float(stat),
        "adf_pvalue": float(pvalue),
        "used_lag": int(used_lag),
        "observations": int(nobs),
    }


def residual_report(actual, predicted, ljung_lags=(5, 10)) -> Dict[str, float]:
    """Summarize residual bias, dispersion, tails, and autocorrelation."""

    residuals = pd.Series(np.asarray(actual) - np.asarray(predicted)).dropna()
    report = {
        "mean": float(residuals.mean()),
        "std": float(residuals.std()),
        "skew": float(stats.skew(residuals)),
        "excess_kurtosis": float(stats.kurtosis(residuals)),
    }
    lb = acorr_ljungbox(residuals, lags=list(ljung_lags), return_df=True)
    for lag in ljung_lags:
        report[f"ljung_box_pvalue_lag_{lag}"] = float(lb.loc[lag, "lb_pvalue"])
    return report
