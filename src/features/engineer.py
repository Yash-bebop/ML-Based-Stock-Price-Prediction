def engineer_features(df):
    data = df.copy()

    # Check if there is enough data for feature engineering
    # The largest window is 200 for SMA_200.
    min_rows_needed = 200
    if len(data) < min_rows_needed:
        raise ValueError(
            f"Insufficient data for feature engineering. Expected at least "
            f"{min_rows_needed} rows, but got {len(data)}. "
            f"Please ensure 'download_stock_data' fetches enough historical data."
        )

    close = data['Close'].squeeze(); high=data['High'].squeeze()
    low=data['Low'].squeeze();       vol=data['Volume'].squeeze()
    data['SMA_20']  = SMAIndicator(close,20).sma_indicator()
    data['SMA_50']  = SMAIndicator(close,50).sma_indicator()
    data['SMA_200'] = SMAIndicator(close,200).sma_indicator()
    data['EMA_12']  = EMAIndicator(close,12).ema_indicator()
    data['EMA_26']  = EMAIndicator(close,26).ema_indicator()
    macd=MACD(close); data['MACD']=macd.macd(); data['MACD_Signal']=macd.macd_signal(); data['MACD_Hist']=macd.macd_diff()
    data['RSI_14'] = RSIIndicator(close,14).rsi()
    stoch=StochasticOscillator(high,low,close); data['Stoch_K']=stoch.stoch(); data['Stoch_D']=stoch.stoch_signal()
    bb=BollingerBands(close,20,2)
    data['BB_Upper']=bb.bollinger_hband(); data['BB_Middle']=bb.bollinger_mavg()
    data['BB_Lower']=bb.bollinger_lband(); data['BB_Width']=bb.bollinger_wband(); data['BB_Pct']=bb.bollinger_pband()
    data['ATR_14']         = AverageTrueRange(high,low,close,14).average_true_range()
    data['Daily_Return']   = close.pct_change()
    data['Log_Return']     = np.log(close/close.shift(1))
    data['Rolling_Std_20'] = close.rolling(20).std()
    data['OBV']            = OnBalanceVolumeIndicator(close,vol).on_balance_volume()
    data['Volume_SMA_20']  = vol.rolling(20).mean()
    data['Volume_Ratio']   = vol/data['Volume_SMA_20']
    for lag in [1,2,3,5,10]: data[f'Return_Lag_{lag}']=np.log(close/close.shift(lag))
    data['Rolling_Return_5']  = data['Log_Return'].rolling(5).mean()
    data['Rolling_Return_10'] = data['Log_Return'].rolling(10).mean()
    data['Rolling_Return_20'] = data['Log_Return'].rolling(20).mean()  # FIX: was rolling(10), an exact duplicate of Rolling_Return_10
    data['Price_vs_SMA20'] = (close-data['SMA_20'])/data['SMA_20']
    data['Price_vs_SMA50'] = (close-data['SMA_50'])/data['SMA_50']
    data['Day_of_Week']=data.index.dayofweek; data['Month']=data.index.month; data['Quarter']=data.index.quarter
    # Target = log-return N days ahead (avoids price scale issues)
    data['Target'] = np.log(close.shift(-CONFIG['forecast_days'])/close)
    data['Intraday_Return'] = (data['Close'] - data['Open']) / data['Open']

    # ── QUANT RESEARCH FEATURES ──────────────────────────────────────────
    # Jegadeesh & Titman (1993) 'Returns to Buying Winners and Selling
    # Losers', Journal of Finance 48(1): past-return momentum predicts
    # future returns over 3-12 month horizons. We use shorter (1W-3M)
    # windows appropriate for a daily-bar model.
    for win in [5, 10, 21, 63]:
        data[f'Mom_{win}d'] = close.pct_change(win)

    # Garman & Klass (1980) 'On the Estimation of Security Price
    # Volatilities from Historical Data', Journal of Business 53(1):
    # an OHLC-based volatility estimator that is ~7-8x more efficient
    # than close-to-close std for the same sample size.
    # sigma^2_GK = 0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2
    # NOTE: this expression can occasionally go slightly negative on noisy
    # bars (it is a difference of two non-negative terms, not a sum) —
    # clip at 0 before sqrt() or it silently produces NaNs that cascade
    # into every downstream row via rolling/lag features.
    ln_hl = np.log(high / low)
    ln_co = np.log(close / data['Open'])
    gk_var = 0.5 * ln_hl**2 - (2*np.log(2) - 1) * ln_co**2
    data['GK_Vol'] = np.sqrt(gk_var.clip(lower=0))

    # Parkinson (1980) 'The Extreme Value Method for Estimating the
    # Variance of the Rate of Return', Journal of Business 53(1):
    # range-based estimator using only the high-low range.
    data['Parkinson_Vol'] = np.sqrt(ln_hl**2 / (4 * np.log(2)))

    # Amihud (2002) 'Illiquidity and Stock Returns: Cross-Section and
    # Time-Series Effects', Journal of Financial Markets 5(1): price
    # impact per unit of dollar volume — a standard illiquidity proxy.
    data['Amihud_Illiq'] = (
        np.abs(data['Log_Return']) / (close * vol + 1e-8)
    ).rolling(21).mean()

    # Realized variance (Andersen & Bollerslev 1998 framework): sum of
    # squared daily log-returns over a rolling window, a model-free
    # estimate of integrated variance.
    data['RealizedVar_21d'] = (data['Log_Return']**2).rolling(21).sum()

    # Ulcer Index (Martin & McCann 1987, 'The Investor's Guide to
    # Fidelity Funds'): downside-only drawdown-depth risk measure, used
    # as the denominator of the Ulcer Performance Index.
    roll_max = close.rolling(14).max()
    drawdown_pct = 100 * (close - roll_max) / roll_max
    data['Ulcer_14d'] = np.sqrt((drawdown_pct**2).rolling(14).mean())

    # Standardised volume shock (z-score vs its own rolling distribution).
    data['Volume_ZScore'] = (
        (vol - vol.rolling(21).mean()) / (vol.rolling(21).std() + 1e-8)
    )

    # Lag-1 return autocorrelation (Lo & MacKinlay 1988, 'Stock Market
    # Prices Do Not Follow Random Walks', Review of Financial Studies
    # 1(1)): positive -> trending regime, negative -> mean-reverting.
    data['AutoCorr_5d'] = (
        data['Log_Return'].rolling(20).apply(
            lambda x: x.autocorr(lag=1) if x.notna().sum() > 5 else 0.0, raw=False
        )
    )

    # Kaufman (1995) Efficiency Ratio, 'Smarter Trading': trend strength
    # on a 0->1 scale = |net price change| / sum(|bar-to-bar changes|).
    def _efficiency_ratio(s, n=10):
        net   = (s - s.shift(n)).abs()
        total = s.diff().abs().rolling(n).sum()
        return net / (total + 1e-8)
    data['Efficiency_Ratio'] = _efficiency_ratio(close, 10)

    data.dropna(inplace=True)
    print(f'✅ Features: {data.shape[1]} cols, {len(data)} rows')
    return data

