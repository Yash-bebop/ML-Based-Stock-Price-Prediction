def run_data_agent(df, ticker):
    print(f'\n🤖 AGENT 1: {ticker}\n' + '='*55)
    enriched = df.copy()
    yahoo  = scrape_yahoo_news(ticker)
    alt    = (fetch_alphavantage_news(ticker) or fetch_finnhub_news(ticker)  # FIX M2: AV first (pre-scored avoids FinBERT), then Finnhub, Google last
               or fetch_google_news_rss(ticker))
    ns = score_sentiment(yahoo); als = score_sentiment(alt)
    non_zero = [s for s in [ns, als] if s != 0.0]
    combined = float(np.mean(non_zero)) if non_zero else 0.0
    analyst  = scrape_analyst_rating(ticker)

    # FIX (was look-ahead leakage): today's live sentiment was previously broadcast
    # backward across the last 30 HISTORICAL rows with exponential decay. That plants
    # information from today into rows representing past trading days that didn't have
    # it yet. Sentiment reflects right now, so it can only validly apply to the most
    # recent row. (News_Sentiment is also excluded from feature_cols below, so this
    # never reached the trained models — but the historical dataframe itself was wrong,
    # which is a trap for any future code that reuses df_features.)
    enriched['News_Sentiment'] = 0.0
    enriched['Analyst_Score']  = 0.0
    enriched.iloc[-1, enriched.columns.get_loc('News_Sentiment')] = combined
    enriched.iloc[-1, enriched.columns.get_loc('Analyst_Score')]  = analyst

    macro = fetch_macro_features(CONFIG['start_date'], CONFIG['end_date'])
    if not macro.empty:
        enriched = enriched.join(macro, how='left')
        # FIX (look-ahead leak): .bfill() here used to fill EARLY rows (before a
        # series' first release / before the CPI lag-shift kicks in) with a LATER,
        # not-yet-known value — textbook look-ahead bias. ffill() only carries
        # KNOWN values forward in time, which is the only causally valid direction.
        # Rows still NaN after ffill() (the very start of history, before any macro
        # print existed) are removed by engineer_features()'s dropna() a few steps
        # downstream, same as any other warm-up-period row.
        for col in macro.columns: enriched[col] = enriched[col].ffill()

    report = {
        'ticker': ticker, 'run_timestamp': datetime.now().isoformat(),
        'news_sentiment': round(ns,4), 'alt_news_sentiment': round(als,4),
        'combined_sentiment': round(combined,4), 'analyst_score': round(analyst,4),
        'yahoo_count': len(yahoo), 'alt_news_count': len(alt),
    }
    print(f'✅ Agent 1 done. New cols: {[c for c in enriched.columns if c not in df.columns]}')
    return enriched, report

def fetch_google_news_rss(ticker, max_articles=25):
    try:
        # Works for Indian stocks too — just pass 'RELIANCE NSE' or 'TCS India'
        url = f'https://news.google.com/rss/search?q={ticker}+stock&hl=en-IN&gl=IN&ceid=IN:en'
        feed = feedparser.parse(url)
        return [{'source':'GoogleNews','text':e.title,'score':1,'pre_scored':False}
                for e in feed.entries[:max_articles] if len(e.title) > 20]
    except Exception as e: print(f'  Google RSS: {e}'); return []

def fetch_alphavantage_news(ticker, max_articles=25):
    if not KEYS.get('alphavantage'): return []
    try:
        r = requests.get(
            f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT'
            f'&tickers={ticker}&limit={max_articles}&apikey={KEYS["alphavantage"]}', timeout=15)
        feed = r.json().get('feed', [])
        return [{'source':'AlphaVantage','text':a['title'],
                 'score':float(next((s for s in a.get('ticker_sentiment',[])
                                     if s['ticker']==ticker), {}).get('ticker_sentiment_score',0)),
                 'pre_scored':True}
                for a in feed]
    except Exception as e: print(f'  AlphaVantage: {e}'); return []

def score_sentiment(texts):
    if not texts: return 0.0
    pre = [t for t in texts if t.get('pre_scored')]
    if pre: return float(np.mean([t['score'] for t in pre]))
    if finbert is None: return 0.0
    label_map = {'positive':1.0,'neutral':0.0,'negative':-1.0}
    scores = []
    for item in texts:
        try:
            res = finbert(item['text'][:512])[0]
            scores.append(label_map.get(res['label'],0.0)*res['score']*(1+np.log1p(item.get('score',1))))
        except: pass
    return float(np.mean(scores)) if scores else 0.0

def scrape_yahoo_news(ticker, max_articles=25):
    try:
        url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
        feed = feedparser.parse(url)
        return [{'source':'YahooRSS','text':e.title,'score':1,'pre_scored':False}
                for e in feed.entries[:max_articles] if len(e.get('title','')) > 20]
    except Exception as e: print(f'  Yahoo RSS: {e}'); return []

def fetch_finnhub_news(ticker, max_articles=25):
    if not KEYS.get('finnhub'): return []
    try:
        end_dt   = datetime.today().strftime('%Y-%m-%d')
        start_dt = (datetime.today()-timedelta(days=7)).strftime('%Y-%m-%d')
        r = requests.get(
            f'https://finnhub.io/api/v1/company-news?symbol={ticker}'
            f'&from={start_dt}&to={end_dt}&token={KEYS["finnhub"]}', timeout=10)
        return [{'source':'Finnhub','text':a['headline'],'score':1,'pre_scored':False}
                for a in r.json()[:max_articles] if len(a.get('headline','')) > 20]
    except Exception as e: print(f'  Finnhub: {e}'); return []

def fetch_macro_features(start_date, end_date):
    """
    DFF (Fed Funds) and VIXCLS are published same-day; T10Y2Y (10Y-2Y
    Treasury spread) is also a daily series. CPIAUCSL (CPI) is different:
    the index value is dated to the *reference month*, but the BLS doesn't
    actually release that number until roughly 2-3 weeks into the
    following month. Using CPIAUCSL's own date as if it were known on
    that date is a point-in-time / look-ahead error — a backtest using it
    that way would 'know' a CPI print before it was published. We shift
    CPI by ~10 business days (a conservative approximation of the BLS
    release lag) so the feature only reflects what was actually public
    knowledge on that date. DFF/T10Y2Y/VIXCLS are left unshifted.
    """
    if not KEYS.get('fred'): return pd.DataFrame()
    try:
        from fredapi import Fred
        f = Fred(api_key=KEYS['fred'])
        data = [f.get_series(k, observation_start=start_date, observation_end=end_date).rename(v)
                for k,v in {'DFF':'Fed_Funds_Rate','T10Y2Y':'Yield_Curve',
                             'VIXCLS':'VIX_FRED','CPIAUCSL':'CPI'}.items()]
        macro = pd.concat(data, axis=1)
        macro.index = pd.to_datetime(macro.index)
        macro = macro.resample('B').last().ffill()
        if 'CPI' in macro.columns:
            macro['CPI'] = macro['CPI'].shift(10)   # publication-lag fix — avoids look-ahead bias
        return macro
    except Exception as e: print(f'  FRED: {e}'); return pd.DataFrame()

def scrape_analyst_rating(ticker):
    score_map = {'strong buy':2.0,'buy':1.5,'overweight':1.0,'hold':0.0,
                 'neutral':0.0,'underweight':-1.0,'sell':-1.5,'strong sell':-2.0}
    try:
        r = requests.get(f'https://finviz.com/quote.ashx?t={ticker}',
                          headers={'User-Agent':'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        for td in soup.find_all('td'):
            if 'Recom' in td.get_text():
                v = td.find_next_sibling('td')
                if v: return score_map.get(v.get_text(strip=True).lower(), 0.0)
    except Exception as e: print(f'  Finviz: {e}')
    return 0.0

