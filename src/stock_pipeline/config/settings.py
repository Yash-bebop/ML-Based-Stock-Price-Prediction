"""Central configuration for the stock prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime settings shared across data, model, quant, and UI layers."""

    ticker: str = "AAPL"
    start_date: str = "2015-01-01"
    end_date: str = field(default_factory=lambda: datetime.today().strftime("%Y-%m-%d"))
    test_size: float = 0.20
    look_back: int = 60
    forecast_days: int = 1
    random_state: int = 42
    n_folds: int = 5
    multistep_days: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    initial_capital: float = 10_000.0
    transaction_cost: float = 0.001
    indian_tickers: List[str] = field(
        default_factory=lambda: [
            "RELIANCE.NS",
            "TCS.NS",
            "INFY.NS",
            "HDFCBANK.NS",
            "WIPRO.NS",
        ]
    )
    ensemble_weights: Dict[str, float] = field(
        default_factory=lambda: {"xgb": 0.4, "lgb": 0.4, "rf": 0.2}
    )
    cv_purge_days: int = 63
    cv_embargo_days: int = 21
    as_gamma: float = 0.1
    as_kappa: float = 1.5
    sentiment_tilt_cap: float = 0.0015


def default_config(**overrides) -> PipelineConfig:
    """Return the default notebook configuration with optional overrides."""

    values = PipelineConfig().__dict__.copy()
    values.update(overrides)
    return PipelineConfig(**values)
