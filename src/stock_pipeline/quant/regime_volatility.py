"""HMM regime detection and GARCH volatility modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegimeVolatilityResult:
    frame: pd.DataFrame
    hmm_available: bool
    garch_available: bool
    bull_state: int = 1


def add_hmm_regimes(frame: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, bool, int]:
    """Fit a two-state Gaussian HMM on train returns and label all rows."""

    df = frame.copy()
    try:
        from hmmlearn.hmm import GaussianHMM

        split = int(len(df) * (1 - test_size))
        returns = df["Log_Return"].to_numpy().reshape(-1, 1)
        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=random_state)
        model.fit(returns[:split])
        state_means = model.means_.flatten()
        bull_state = int(np.argmax(state_means))
        df["HMM_Regime"] = model.predict(returns)
        df["HMM_Bull_Prob"] = model.predict_proba(returns)[:, bull_state]
        return df, True, bull_state
    except Exception:
        df["HMM_Regime"] = 0
        df["HMM_Bull_Prob"] = 0.5
        return df, False, 1


def add_garch_volatility(frame: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, bool]:
    """Fit train-only GARCH(1,1) and roll conditional volatility forward."""

    df = frame.copy()
    try:
        from arch import arch_model

        ret_pct = df["Log_Return"].to_numpy() * 100
        split = int(len(ret_pct) * (1 - test_size))
        result = arch_model(ret_pct[:split], mean="Zero", vol="Garch", p=1, q=1, dist="t").fit(
            disp="off", show_warning=False
        )
        omega = result.params["omega"]
        alpha = result.params["alpha[1]"]
        beta = result.params["beta[1]"]
        sigma2 = np.full(len(ret_pct), np.nan)
        sigma2[0] = np.var(ret_pct[:split])
        for idx in range(1, len(ret_pct)):
            sigma2[idx] = omega + alpha * (ret_pct[idx - 1] ** 2) + beta * sigma2[idx - 1]
        df["GARCH_Vol"] = np.sqrt(np.clip(sigma2, 0, None))
        return df, True
    except Exception:
        df["GARCH_Vol"] = df.get("Rolling_Std_20", pd.Series(index=df.index, data=0.0)) * 100
        return df, False


def add_regime_and_volatility(frame: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> RegimeVolatilityResult:
    """Apply the notebook's HMM and GARCH diagnostics in sequence."""

    with_regime, hmm_ok, bull_state = add_hmm_regimes(frame, test_size=test_size, random_state=random_state)
    with_vol, garch_ok = add_garch_volatility(with_regime, test_size=test_size)
    return RegimeVolatilityResult(with_vol, hmm_ok, garch_ok, bull_state)
