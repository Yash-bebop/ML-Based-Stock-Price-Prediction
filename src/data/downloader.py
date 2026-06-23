def download_stock_data(ticker, start, end):
    print(f'📡 Downloading {ticker}...')
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty: raise ValueError(f'No data for {ticker}')
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.strip() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True); df.dropna(how='all', inplace=True)
    print(f'  ✅ {len(df)} days | Close: {df["Close"].min():.2f} – {df["Close"].max():.2f}')
    return df

def add_market_context(df, ticker, tickers=('SPY','QQQ','XLK','^VIX'), start=None, end=None):  # FIX L2: accept window params
    """
    Adds US market ETFs and VIX as additional features.
    For Indian stocks (ticker ends with .NS or .BO), also adds NIFTY50 (^NSEI) and SENSEX (^BSESN).

    FIX: Indian-ticker detection now checks the actual `ticker` string instead of
    df.columns / df.index.name — neither of those ever contained '.NS'/'.BO', so the
    NIFTY/SENSEX branch could never fire before, despite the docstring claiming it did.
    """
    enriched = df.copy()
    is_indian = ticker.endswith('.NS') or ticker.endswith('.BO')
    extra = ('^NSEI', '^BSESN') if is_indian else ()
    all_tickers = list(tickers) + list(extra)
    for t in all_tickers:
        try:
            col = t.replace('^','')
            mkt = yf.download(t, start=start or CONFIG['start_date'], end=end or CONFIG['end_date'],  # FIX L2: use passed window
                              auto_adjust=True, progress=False)['Close']
            if isinstance(mkt, pd.DataFrame): mkt = mkt.squeeze()
            mkt.index = pd.to_datetime(mkt.index)
            enriched[f'{col}_Close']  = mkt.reindex(enriched.index, method='ffill')
            enriched[f'{col}_Return'] = enriched[f'{col}_Close'].pct_change()
            enriched[f'{col}_Lag1']   = enriched[f'{col}_Return'].shift(1)
            print(f'  ✅ {t}: {enriched[f"{col}_Return"].notna().sum()} rows')
        except Exception as e:
            print(f'  ⚠️ {t}: {e}')
    return enriched

