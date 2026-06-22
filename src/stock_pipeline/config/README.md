# Configuration

This folder contains the settings that were previously defined in the notebook's configuration cell.

## Files

| File | Purpose |
|---|---|
| `settings.py` | Defines `PipelineConfig`, the shared dataclass used by the rest of the modular pipeline. |

## Notes

Keep experiment-level defaults here instead of scattering constants across modeling, dashboard, and export code. Runtime callers can use `default_config(ticker="MSFT")` or pass a fully constructed `PipelineConfig` into downstream functions.
