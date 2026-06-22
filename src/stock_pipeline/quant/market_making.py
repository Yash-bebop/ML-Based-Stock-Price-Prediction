"""Avellaneda-Stoikov spread model helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def avellaneda_stoikov_spread(
    mid_price,
    volatility,
    gamma: float = 0.1,
    kappa: float = 1.5,
    horizon: float = 1.0,
    inventory: float = 0.0,
) -> pd.DataFrame:
    """Compute reservation price and bid/ask spread for a price series."""

    mid = pd.Series(mid_price).astype(float)
    sigma = pd.Series(volatility, index=mid.index).astype(float)
    variance = np.square(sigma / 100.0)
    reservation_price = mid - inventory * gamma * variance * horizon * mid
    half_spread = ((gamma * variance * horizon) / 2.0 + np.log(1 + gamma / kappa) / gamma) * mid / 100.0
    return pd.DataFrame(
        {
            "mid_price": mid,
            "reservation_price": reservation_price,
            "bid": reservation_price - half_spread,
            "ask": reservation_price + half_spread,
            "half_spread": half_spread,
            "full_spread": half_spread * 2,
            "full_spread_pct": (half_spread * 2) / mid * 100,
        }
    )
