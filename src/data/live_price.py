def get_live_price(ticker: str) -> dict:
    """
    Agentic price fetcher — runs all relevant sources in parallel threads,
    returns the first successful non-zero result in priority order.

    Priority:
      Indian tickers (.NS/.BO) → NSE_API first, yfinance_intraday second, daily last
      US tickers               → Finnhub first, yfinance_intraday second, daily last

    This is faster than sequential fallback because all sources start at the same time.
    Whichever finishes first with a valid price wins.
    """
    is_indian     = ticker.endswith('.NS') or ticker.endswith('.BO')
    symbol_clean  = ticker.replace('.NS','').replace('.BO','')

    if is_indian:
        sources = [
            ('NSE_API',           lambda: _fetch_nse(symbol_clean)),
            ('yfinance_intraday', lambda: _fetch_yfinance_intraday(ticker)),
            ('yfinance_daily',    lambda: _fetch_yfinance_daily(ticker)),
        ]
        priority = ['NSE_API', 'yfinance_intraday', 'yfinance_daily']
    else:
        sources = [
            ('Finnhub',           lambda: _fetch_finnhub(ticker)),
            ('yfinance_intraday', lambda: _fetch_yfinance_intraday(ticker)),
            ('yfinance_daily',    lambda: _fetch_yfinance_daily(ticker)),
        ]
        priority = ['Finnhub', 'yfinance_intraday', 'yfinance_daily']

    results = {}
    errors  = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(fn): name for name, fn in sources}
        try:
            for future in concurrent.futures.as_completed(future_map, timeout=15):
                name = future_map[future]
                try:
                    res = future.result()
                    if res and res.get('price', 0) > 0:
                        results[name] = res
                except Exception as e:
                    errors[name] = str(e)
        except concurrent.futures.TimeoutError:
            pass  # use whatever results arrived before timeout

    # Return best result in priority order
    for source in priority:
        if source in results:
            r = results[source]
            print(f"  📡 {ticker}: source={source} price={r.get('currency','')} {r['price']}")
            return r

    # Everything failed
    err_summary = ' | '.join(f'{k}: {v}' for k, v in errors.items())
    return {
        'ticker'    : ticker,
        'price'     : 0,
        'currency'  : 'INR' if is_indian else 'USD',
        'error'     : f'All sources failed — {err_summary}',
        'timestamp' : datetime.now().strftime('%H:%M:%S'),
    }

def _fetch_finnhub(ticker: str) -> dict:
    """
    Finnhub /quote endpoint — real-time US stock prices.
    Free tier: 60 calls/minute, no daily limit.
    """
    key = KEYS.get('finnhub', '')
    if not key:
        raise ValueError('No FINNHUB_API_KEY in secrets')
    r = requests.get(
        f'https://finnhub.io/api/v1/quote?symbol={ticker}&token={key}',
        timeout=8
    ).json()
    price = r.get('c', 0)
    prev  = r.get('pc', 0)
    if not price or price == 0:
        raise ValueError(f'Finnhub returned zero price for {ticker}')
    change  = price - prev
    chgpct  = (change / prev * 100) if prev else 0
    return {
        'ticker'      : ticker,
        'price'       : round(price, 2),
        'prev_close'  : round(prev, 2),
        'change'      : round(change, 2),
        'change_pct'  : round(chgpct, 3),
        'volume'      : int(r.get('v', 0)),
        'currency'    : 'USD',
        'market_state': 'REGULAR' if r.get('d') is not None else 'CLOSED',
        'arrow'       : '▲' if change >= 0 else '▼',
        'color_hint'  : 'green' if change >= 0 else 'red',
        'timestamp'   : datetime.now().strftime('%H:%M:%S'),
        'source'      : 'Finnhub',
        'error'       : None,
    }

def _fetch_nse(symbol_clean: str) -> dict:
    """
    Hits NSE India's own quote endpoint — same feed the NSE website uses.
    Real-time during market hours 9:15–15:30 IST, no API key needed.
    symbol_clean: base symbol without suffix e.g. 'INFY' not 'INFY.NS'
    """
    headers = {
        'User-Agent'     : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept'         : '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer'        : 'https://www.nseindia.com',
    }
    session = requests.Session()
    # NSE blocks cold API calls — must hit homepage first to get cookies
    session.get('https://www.nseindia.com', headers=headers, timeout=10)
    r = session.get(
        f'https://www.nseindia.com/api/quote-equity?symbol={symbol_clean.upper()}',
        headers=headers, timeout=10
    )
    d = r.json()
    pi     = d['priceInfo']
    price  = float(pi['lastPrice'])
    prev   = float(pi['previousClose'])
    change = float(pi['change'])
    chgpct = float(pi['pChange'])
    try:
        vol = int(d['marketDeptOrderBook']['tradeInfo']['totalTradedVolume'])
    except Exception:
        vol = 0
    try:
        mstate = 'REGULAR' if d['marketStatus']['marketStatus'] == 'Open' else 'CLOSED'
    except Exception:
        mstate = 'CLOSED'
    return {
        'ticker'      : symbol_clean,
        'price'       : round(price, 2),
        'prev_close'  : round(prev, 2),
        'change'      : round(change, 2),
        'change_pct'  : round(chgpct, 3),
        'volume'      : vol,
        'currency'    : 'INR',
        'market_state': mstate,
        'arrow'       : '▲' if change >= 0 else '▼',
        'color_hint'  : 'green' if change >= 0 else 'red',
        'timestamp'   : datetime.now().strftime('%H:%M:%S'),
        'source'      : 'NSE_API',
        'error'       : None,
    }

def _fetch_yfinance_intraday(ticker: str) -> dict:
    """
    5-minute candles over last 2 days.
    Works for US + Indian but may lag or fail — used as fallback only.
    """
    is_indian = ticker.endswith('.NS') or ticker.endswith('.BO')
    currency  = 'INR' if is_indian else 'USD'
    df = yf.download(ticker, period='2d', interval='5m',
                     auto_adjust=True, progress=False, timeout=10)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=['Close'])
    if df.empty:
        raise ValueError('yfinance intraday empty')
    price  = float(df['Close'].iloc[-1])
    volume = int(df['Volume'].iloc[-1])
    # Separate today vs yesterday bars for accurate prev_close
    today_str  = datetime.utcnow().strftime('%Y-%m-%d')
    today_bars = df[df.index.strftime('%Y-%m-%d') == today_str]
    yest_bars  = df[df.index.strftime('%Y-%m-%d') != today_str]
    if not yest_bars.empty and not today_bars.empty:
        prev_close = float(yest_bars['Close'].iloc[-1])
    else:
        prev_close = float(df['Close'].iloc[-2]) if len(df) >= 2 else price
    # Infer market state from data freshness
    age_min = (datetime.utcnow() - df.index[-1].to_pydatetime().replace(tzinfo=None)).total_seconds() / 60
    mstate  = 'REGULAR (~5m delay)' if age_min < 20 else 'CLOSED/POST'
    change  = price - prev_close
    chgpct  = (change / prev_close * 100) if prev_close else 0
    return {
        'ticker'      : ticker,
        'price'       : round(price, 2),
        'prev_close'  : round(prev_close, 2),
        'change'      : round(change, 2),
        'change_pct'  : round(chgpct, 3),
        'volume'      : volume,
        'currency'    : currency,
        'market_state': mstate,
        'arrow'       : '▲' if change >= 0 else '▼',
        'color_hint'  : 'green' if change >= 0 else 'red',
        'timestamp'   : datetime.now().strftime('%H:%M:%S'),
        'source'      : 'yfinance_intraday',
        'error'       : None,
    }

def _fetch_yfinance_daily(ticker: str) -> dict:
    """Last resort — always works but shows yesterday's close."""
    is_indian = ticker.endswith('.NS') or ticker.endswith('.BO')
    currency  = 'INR' if is_indian else 'USD'
    df = yf.download(ticker, period='5d', interval='1d',
                     auto_adjust=True, progress=False, timeout=10)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    closes    = df['Close'].dropna()
    price     = float(closes.iloc[-1])
    prev      = float(closes.iloc[-2]) if len(closes) >= 2 else price
    change    = price - prev
    chgpct    = (change / prev * 100) if prev else 0
    return {
        'ticker'      : ticker,
        'price'       : round(price, 2),
        'prev_close'  : round(prev, 2),
        'change'      : round(change, 2),
        'change_pct'  : round(chgpct, 3),
        'volume'      : int(df['Volume'].dropna().iloc[-1]),
        'currency'    : currency,
        'market_state': 'CLOSED (daily data — prev session close)',
        'arrow'       : '▲' if change >= 0 else '▼',
        'color_hint'  : 'green' if change >= 0 else 'red',
        'timestamp'   : datetime.now().strftime('%H:%M:%S'),
        'source'      : 'yfinance_daily',
        'error'       : None,
    }

def format_price_card(info: dict) -> str:
    """Formats live price dict into a human-readable string for Gradio."""
    if info.get('error'):
        return f"❌ {info['ticker']}: {info['error']}"
    sym  = '₹' if info.get('currency') == 'INR' else '$'
    sign = '+' if info.get('change', 0) >= 0 else ''
    src  = info.get('source', '')
    return (
        f"{info.get('arrow','?')} {info['ticker']}  {sym}{info['price']:,.2f}  "
        f"{sign}{info.get('change',0):.2f} ({sign}{info.get('change_pct',0):.2f}%)  "
        f"| Vol: {info.get('volume',0):,}  "
        f"| {info.get('market_state','?')}  "
        f"| src: {src}  "
        f"| {info.get('timestamp','')}"
    )

