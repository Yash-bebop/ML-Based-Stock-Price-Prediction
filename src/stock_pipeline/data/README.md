# Data

This folder owns all market-data retrieval and quote normalization.

## Files

| File | Purpose |
|---|---|
| `market_data.py` | Downloads OHLCV history, joins market context symbols, and reads live/delayed quote data. |

## Notebook Mapping

This folder replaces the notebook cells that used `yfinance`, live price polling, and market-context ETFs such as SPY, QQQ, XLK, and VIX.

## Design Notes

- Keep raw download behavior separate from feature engineering.
- Return normalized pandas frames with predictable OHLCV columns.
- Represent live prices with the `LivePrice` dataclass so UI and prediction code do not depend on provider-specific response shapes.
