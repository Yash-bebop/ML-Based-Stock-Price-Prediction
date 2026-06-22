"""Portfolio and strategy risk metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    """Return maximum drawdown as a decimal."""

    curve = pd.Series(equity).dropna()
    return float((curve / curve.cummax() - 1).min())


def risk_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """Compute the notebook's Sharpe/Sortino/Calmar/VaR/CVaR risk block."""

    r = pd.Series(returns).dropna()
    if r.empty:
        return {}
    excess = r - risk_free_rate / 252
    downside = excess[excess < 0]
    equity = (1 + r).cumprod()
    annual_return = equity.iloc[-1] ** (252 / len(r)) - 1
    annual_vol = r.std() * np.sqrt(252)
    var_95 = np.percentile(r, 5)
    cvar_95 = r[r <= var_95].mean() if (r <= var_95).any() else var_95
    mdd = max_drawdown(equity)
    return {
        "ann_return_pct": float(annual_return * 100),
        "ann_vol_pct": float(annual_vol * 100),
        "sharpe": float(excess.mean() / max(excess.std(), 1e-12) * np.sqrt(252)),
        "sortino": float(excess.mean() / max(downside.std(), 1e-12) * np.sqrt(252)),
        "calmar": float(annual_return / abs(mdd)) if mdd else np.nan,
        "var_95_1d_pct": float(var_95 * 100),
        "cvar_95_1d_pct": float(cvar_95 * 100),
        "max_drawdown_pct": float(mdd * 100),
        "tail_ratio": float(abs(np.percentile(r, 95) / max(abs(np.percentile(r, 5)), 1e-12))),
    }


def long_short_strategy_returns(predicted_returns, actual_returns, transaction_cost: float = 0.001) -> pd.Series:
    """Create simple sign-based strategy returns from predicted log returns."""

    signal = np.where(np.asarray(predicted_returns) > 0, 1.0, -1.0)
    turnover = np.abs(np.diff(signal, prepend=0))
    gross = signal * np.asarray(actual_returns)
    net = gross - turnover * transaction_cost
    return pd.Series(net)
