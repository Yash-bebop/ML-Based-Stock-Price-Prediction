"""Technical feature engineering for stock-return prediction."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the notebook's technical and return-based modeling features."""

    df = frame.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    df["Daily_Return"] = close.pct_change()
    df["Log_Return"] = np.log(close / close.shift(1))
    df["Target"] = df["Log_Return"].shift(-1)

    df["SMA_20"] = SMAIndicator(close, window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close, window=50).sma_indicator()
    df["SMA_200"] = SMAIndicator(close, window=200).sma_indicator()
    df["EMA_12"] = EMAIndicator(close, window=12).ema_indicator()
    df["EMA_26"] = EMAIndicator(close, window=26).ema_indicator()
    df["Price_vs_SMA20"] = (close - df["SMA_20"]) / df["SMA_20"]
    df["Price_vs_SMA50"] = (close - df["SMA_50"]) / df["SMA_50"]

    macd = MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Diff"] = macd.macd_diff()
    df["RSI"] = RSIIndicator(close, window=14).rsi()

    stoch = StochasticOscillator(high=high, low=low, close=close)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    bb = BollingerBands(close, window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / close
    df["ATR"] = AverageTrueRange(high=high, low=low, close=close).average_true_range()
    df["OBV"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

    df["Rolling_Mean_20"] = df["Log_Return"].rolling(20).mean()
    df["Rolling_Std_20"] = df["Log_Return"].rolling(20).std()
    df["Rolling_Std_63"] = df["Log_Return"].rolling(63).std()
    df["Momentum_20"] = close.pct_change(20)
    df["Momentum_63"] = close.pct_change(63)
    df["Volume_Change"] = volume.pct_change()
    df["Volume_ZScore"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    df["Intraday_Return"] = (close - df["Open"].astype(float)) / df["Open"].astype(float)

    for lag in [1, 2, 3, 5, 10, 21]:
        df[f"Return_Lag_{lag}"] = df["Log_Return"].shift(lag)

    df["DayOfWeek"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["Quarter"] = df.index.quarter

    return df.replace([np.inf, -np.inf], np.nan).dropna().copy()


def feature_target_split(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Return X, y, and feature column names after removing non-feature columns."""

    excluded = {"Target", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
    feature_cols = [col for col in frame.columns if col not in excluded]
    return frame[feature_cols].copy(), frame["Target"].copy(), feature_cols
