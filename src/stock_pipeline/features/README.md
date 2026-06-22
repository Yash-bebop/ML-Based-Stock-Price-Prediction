# Features

This folder owns the transformation from raw OHLCV/context data into model-ready features.

## Files

| File | Purpose |
|---|---|
| `technical_features.py` | Builds technical indicators, lagged returns, rolling volatility, volume features, calendar fields, and the next-day log-return target. |

## Notebook Mapping

This corresponds to the notebook's feature engineering step, where indicators such as SMA, EMA, MACD, RSI, Bollinger Bands, ATR, OBV, lagged returns, and market-context fields are assembled.

## Usage

```python
from stock_pipeline.features.technical_features import engineer_features, feature_target_split

features = engineer_features(raw_frame)
X, y, feature_cols = feature_target_split(features)
```
