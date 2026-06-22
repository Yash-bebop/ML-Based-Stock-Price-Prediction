# Sentiment

This folder contains the news and sentiment overlay logic.

## Files

| File | Purpose |
|---|---|
| `news_sentiment.py` | Reads free/news API sources and produces a transparent sentiment score payload. |

## Notebook Mapping

This replaces the notebook cells that fetched Yahoo, Google RSS, Alpha Vantage, and Finnhub headlines, then produced the `sentiment_report` used by the live forecast and agent workflow.

## Design Notes

The notebook intentionally keeps sentiment out of historical model training because historical sentiment values were sparse. The modular code follows the same design: sentiment is a live inference overlay and an agent input, not a silent training feature.
