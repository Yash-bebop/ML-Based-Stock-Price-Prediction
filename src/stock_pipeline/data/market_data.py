"""Market data access helpers.

This module collects the notebook's historical download, market context, and
live quote routines behind small functions that are easy to test or replace.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class LivePrice:
    ticker: str
    price: float
    currency: str
    change: float = 0.0
    change_pct: float = 0.0
    market_state: str = "UNKNOWN"
    source: str = "yfinance"
    error: Optional[str] = None


def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data and normalize the columns expected by the pipeline."""

    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No market data returned for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna().copy()


def add_market_context(
    frame: pd.DataFrame,
    ticker: str,
    start: str,
    end: str,
    symbols: Iterable[str] = ("SPY", "QQQ", "XLK", "^VIX"),
) -> pd.DataFrame:
    """Join broad-market return context used by the notebook's feature layer."""

    enriched = frame.copy()
    for symbol in symbols:
        if symbol == ticker:
            continue
        try:
            ctx = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
            if ctx.empty:
                continue
            if isinstance(ctx.columns, pd.MultiIndex):
                ctx.columns = ctx.columns.get_level_values(0)
            safe_name = symbol.replace("^", "").replace(".", "_")
            enriched[f"{safe_name}_Return"] = ctx["Close"].pct_change().reindex(enriched.index)
        except Exception:
            enriched[f"{symbol}_Return"] = np.nan
    return enriched.ffill().bfill()


def get_live_price(ticker: str) -> LivePrice:
    """Return the freshest free live/delayed quote available from yfinance."""

    try:
        quote = yf.Ticker(ticker)
        fast = getattr(quote, "fast_info", {}) or {}
        price = float(fast.get("last_price") or fast.get("lastPrice") or 0.0)
        previous_close = float(fast.get("previous_close") or fast.get("previousClose") or 0.0)
        currency = str(fast.get("currency") or ("INR" if ticker.endswith((".NS", ".BO")) else "USD"))

        if price <= 0:
            hist = quote.history(period="2d", interval="1d", auto_adjust=True)
            if hist.empty:
                raise ValueError("No live or recent close data returned")
            price = float(hist["Close"].iloc[-1])
            previous_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price

        change = price - previous_close if previous_close else 0.0
        change_pct = (change / previous_close * 100) if previous_close else 0.0
        return LivePrice(
            ticker=ticker,
            price=price,
            currency=currency,
            change=change,
            change_pct=change_pct,
            market_state="REGULAR",
        )
    except Exception as exc:
        return LivePrice(
            ticker=ticker,
            price=0.0,
            currency="INR" if ticker.endswith((".NS", ".BO")) else "USD",
            error=str(exc),
        )


def recent_window(ticker: str, days: int = 600) -> pd.DataFrame:
    """Download the warm-up window used by live prediction."""

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    data = download_stock_data(ticker, start, end)
    return add_market_context(data, ticker, start=start, end=end)
