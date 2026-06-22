"""News retrieval and lightweight sentiment scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import feedparser
import numpy as np
import requests
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class SentimentReport:
    ticker: str
    yahoo_score: float
    alternative_score: float
    combined_score: float
    yahoo_count: int
    alternative_count: int


def scrape_yahoo_news(ticker: str, limit: int = 20) -> List[str]:
    """Scrape Yahoo Finance article headlines for a ticker."""

    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    titles = [node.get_text(" ", strip=True) for node in soup.find_all(["h3", "a"])]
    return [title for title in titles if title][:limit]


def fetch_google_news_rss(ticker: str, limit: int = 50) -> List[str]:
    """Read Google News RSS headlines as a free fallback source."""

    feed = feedparser.parse(f"https://news.google.com/rss/search?q={ticker}+stock")
    return [entry.title for entry in feed.entries[:limit]]


def fetch_alphavantage_news(ticker: str, api_key: Optional[str] = None, limit: int = 50) -> List[str]:
    """Fetch Alpha Vantage news headlines when an API key is available."""

    if not api_key:
        return []
    url = "https://www.alphavantage.co/query"
    params = {"function": "NEWS_SENTIMENT", "tickers": ticker, "apikey": api_key, "limit": limit}
    data = requests.get(url, params=params, timeout=15).json()
    return [item.get("title", "") for item in data.get("feed", []) if item.get("title")]


def fetch_finnhub_news(ticker: str, api_key: Optional[str] = None, limit: int = 50) -> List[str]:
    """Fetch Finnhub company headlines when an API key is available."""

    if not api_key:
        return []
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": ticker, "from": "2024-01-01", "to": "2099-12-31", "token": api_key}
    data = requests.get(url, params=params, timeout=15).json()
    return [item.get("headline", "") for item in data[:limit] if item.get("headline")]


def score_sentiment(texts: Iterable[str]) -> float:
    """Return a simple lexicon score in [-1, 1] for headline collections."""

    positive = {"beat", "growth", "surge", "gain", "upgrade", "bull", "strong", "record", "profit"}
    negative = {"miss", "fall", "drop", "downgrade", "bear", "weak", "loss", "lawsuit", "risk"}
    scores = []
    for text in texts:
        words = {word.strip(".,:;!?()[]{}\"'").lower() for word in text.split()}
        pos = len(words & positive)
        neg = len(words & negative)
        if pos or neg:
            scores.append((pos - neg) / max(pos + neg, 1))
    return float(np.mean(scores)) if scores else 0.0


def build_sentiment_report(ticker: str, keys: Optional[Dict[str, str]] = None) -> SentimentReport:
    """Build the combined sentiment payload used by prediction and agents."""

    keys = keys or {}
    try:
        yahoo = scrape_yahoo_news(ticker)
    except Exception:
        yahoo = []
    alternative = (
        fetch_alphavantage_news(ticker, keys.get("alphavantage"))
        or fetch_finnhub_news(ticker, keys.get("finnhub"))
        or fetch_google_news_rss(ticker)
    )
    yahoo_score = score_sentiment(yahoo)
    alternative_score = score_sentiment(alternative)
    non_zero = [score for score in [yahoo_score, alternative_score] if score != 0.0]
    combined = float(np.mean(non_zero)) if non_zero else 0.0
    return SentimentReport(ticker, yahoo_score, alternative_score, combined, len(yahoo), len(alternative))
