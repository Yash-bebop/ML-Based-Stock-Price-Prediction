"""Modular stock prediction pipeline extracted from the research notebook.

The notebook remains the exploratory source of truth. This package groups the
same workflow by responsibility so data access, feature engineering, modeling,
quant diagnostics, agents, dashboard, and artifact export can evolve separately.
"""

__all__ = [
    "config",
    "data",
    "features",
    "sentiment",
    "models",
    "quant",
    "agents",
    "dashboard",
    "artifacts",
]
