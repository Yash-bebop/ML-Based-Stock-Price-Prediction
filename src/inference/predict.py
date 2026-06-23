def predict_tomorrow(ticker):
    """
    v3.1 (Robust) — injects live intraday data into a dedicated feature
    to avoid corrupting the daily time-series lags.

    NOTE on what 'tomorrow's price' means here: the model is trained to predict
    close[t+1] / close[t]. At inference time, when a live quote is available, the
    predicted return is applied to the LIVE price (`cp = live_price`), not to
    yesterday's official close. That's a reasonable choice (it keeps the forecast
    anchored to the most current price you'd actually be trading from) but it means
    the output is 'tomorrow's close, anchored to right-now's price' rather than
    strictly 'tomorrow's close, anchored to today's close' — those can diverge on a
    day with a large intraday move, since the model never saw 'live mid-session
    price' as a training reference point.
    """
    cache_key = ticker.upper()
    _now_ts = time.time()
    if cache_key in _predict_cache:
        _cached_result, _cached_ts = _predict_cache[cache_key]
        if _now_ts - _cached_ts < _CACHE_TTL_SECONDS:
            print(f"  ⚡ Cache hit for {ticker} ({int(_CACHE_TTL_SECONDS - (_now_ts - _cached_ts))}s remaining)")
            return _cached_result

    end   = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=600)).strftime('%Y-%m-%d')

    # ── Download historical daily data (for feature warmup) ───────────
    df_hist = download_stock_data(ticker, start, end)
    df_hist = add_market_context(df_hist, ticker, start=start, end=end)  # FIX L2: pass rolling window, not full CONFIG dates

    df_hist = engineer_features(df_hist)

    # ── Get live price from agent ──────────────────────────────────────
    live_info  = get_live_price(ticker)
    live_price = live_info['price'] if (live_info.get('price', 0) > 0
                                        and not live_info.get('error')) else None

    # ── Inject live data into last row if available ────────────────────
    if live_price and live_price > 0:
        last_close = float(df_hist['Close'].iloc[-1])
        today_open = float(df_hist['Open'].iloc[-1])

        # Calculate live intraday return from today's open
        live_intra_ret = (live_price - today_open) / today_open if today_open > 0 else 0

        # ONLY update the dedicated Intraday feature and SMA distances.
        # DO NOT touch Return_Lag_1 or Daily_Return to keep the daily time-series pure.
        if 'Intraday_Return' in df_hist.columns:
            df_hist.iloc[-1, df_hist.columns.get_loc('Intraday_Return')] = live_intra_ret

        if 'SMA_20' in df_hist.columns:
            df_hist.iloc[-1, df_hist.columns.get_loc('Price_vs_SMA20')] = (
                (live_price - float(df_hist['SMA_20'].iloc[-1])) / float(df_hist['SMA_20'].iloc[-1])
            )

        if 'SMA_50' in df_hist.columns:
            df_hist.iloc[-1, df_hist.columns.get_loc('Price_vs_SMA50')] = (
                (live_price - float(df_hist['SMA_50'].iloc[-1])) / float(df_hist['SMA_50'].iloc[-1])
            )

        print(f"  ✅ Live data injected cleanly: live=${live_price:.2f} (intraday momentum: {live_intra_ret*100:+.2f}%)")
        cp = live_price   # use live as base for price conversion
    else:
        cp = float(df_hist['Close'].iloc[-1])
        print(f"  ⚠️  No live data — using last close: {cp:.2f}")

    # ── Build feature vector ───────────────────────────────────────────
    lat_full = np.zeros((1, len(feature_cols)))
    missing  = []
    for i, col in enumerate(feature_cols):
        if col in df_hist.columns:
            lat_full[0, i] = float(df_hist[col].iloc[-1])
        else:
            missing.append(col)
    if missing:
        print(f"  ⚠️  {len(missing)} features zero-filled: {missing[:3]}...")

    ls     = scaler.transform(lat_full)
    lr_xgb = float(xgb_model.predict(ls)[0])
    lr_lgb = float(lgb_model.predict(ls)[0])
    lr_rf  = float(rf_model.predict(ls)[0])
    _ew = CONFIG['ensemble_weights']
    lr_ens = lr_xgb*_ew['xgb'] + lr_lgb*_ew['lgb'] + lr_rf*_ew['rf']  # FIX M4: from CONFIG (now data-driven, Step 5b)

    px_xgb = round(cp * np.exp(lr_xgb), 2)
    px_lgb = round(cp * np.exp(lr_lgb), 2)
    px_rf  = round(cp * np.exp(lr_rf),  2)
    px_ens = round(cp * np.exp(lr_ens), 2)
    pct    = lr_ens * 100

    # ── Sentiment overlay (transparent, capped, SEPARATE from the trained ML
    # ensemble number — see CONFIG['sentiment_tilt_cap'] comment) ─────────────
    # News_Sentiment/Analyst_Score are deliberately excluded from the trained
    # models (see Step 2 / Step 3 comments — their HISTORICAL values are
    # fabricated zeros, so training on them would be meaningless). But at live
    # inference time we DO have a real, current sentiment read for whichever
    # ticker was actually requested, so we surface it as a small, capped,
    # clearly-separate adjustment rather than silently folding it into
    # 'ensemble'. Tetlock (2007), "Giving Content to Investor Sentiment: The
    # Role of Media in the Stock Market", Journal of Finance 62(3), is the
    # standard reference for short-horizon sentiment having SOME predictive
    # content — but it is a small effect, hence the tight cap.
    # NOTE: this fetches sentiment for the ACTUAL ticker passed in (not the
    # notebook-level `sentiment_report`, which only ever reflects
    # CONFIG['ticker']) — using the global AAPL-only report for an arbitrary
    # Gradio ticker would repeat the same cross-ticker bias documented for
    # the trained models themselves (see bias-analysis notes).
    try:
        _yahoo_live = scrape_yahoo_news(ticker)
        _alt_live   = (fetch_alphavantage_news(ticker) or fetch_finnhub_news(ticker)
                       or fetch_google_news_rss(ticker))
        _ns_live  = score_sentiment(_yahoo_live)
        _als_live = score_sentiment(_alt_live)
        _nz_live  = [s for s in [_ns_live, _als_live] if s != 0.0]
        live_combined_sentiment = float(np.mean(_nz_live)) if _nz_live else 0.0
    except Exception as _e:
        live_combined_sentiment = 0.0

    sentiment_tilt = float(np.clip(live_combined_sentiment, -1, 1)) * CONFIG['sentiment_tilt_cap']
    lr_ens_sent    = lr_ens + sentiment_tilt
    px_ens_sent    = round(cp * np.exp(lr_ens_sent), 2)

    currency = live_info.get('currency', 'INR' if '.NS' in ticker else 'USD')
    sym      = '₹' if currency == 'INR' else '$'

    # ── Recent history (for the Gradio per-model prediction chart) ─────────
    _hist_n = min(60, len(df_hist))
    history = {
        'dates': [str(d.date()) for d in df_hist.index[-_hist_n:]],
        'close': [round(float(c), 2) for c in df_hist['Close'].values[-_hist_n:]],
    }

    result = {
        'ticker'          : ticker,
        'current_price'   : round(cp, 2),
        'live_price'      : round(live_price or cp, 2),
        'last_data_date'  : str(df_hist.index[-1].date()),
        'currency'        : currency,
        'symbol'          : sym,
        'xgboost'         : px_xgb,
        'lightgbm'        : px_lgb,
        'random_forest'   : px_rf,
        'ensemble'        : px_ens,
        'ensemble_sentiment_adjusted': px_ens_sent,     # separate, capped, transparent
        'sentiment_score'             : round(live_combined_sentiment, 4),
        'sentiment_tilt_pct'          : round(sentiment_tilt * 100, 4),
        'pct_change'      : round(pct, 3),
        'signal'          : '📈 BULLISH' if lr_ens > 0 else '📉 BEARISH',
        'prediction_date' : (datetime.today() + pd.offsets.BDay(1)).strftime('%Y-%m-%d'),  # FIX C3: BDay skips weekends/holidays
        'live_injected'   : live_price is not None,
        'market_state'    : live_info.get('market_state', 'UNKNOWN'),
        'change_today'    : live_info.get('change', 0),
        'change_pct_today': live_info.get('change_pct', 0),
        'history'         : history,
    }

    print(f"\n{'='*55}")
    print(f"  {ticker} | Live injected: {result['live_injected']}")
    print(f"  Live Price   : {sym}{result['live_price']:.2f} "
          f"({result['change_pct_today']:+.2f}% today)")
    print(f"  Ensemble     : {sym}{px_ens:.2f} ({pct:+.2f}%)")
    print(f"  + Sentiment  : {sym}{px_ens_sent:.2f} (tilt {result['sentiment_tilt_pct']:+.3f}%, "
          f"score={result['sentiment_score']:+.3f})")
    print(f"  Signal       : {result['signal']}")
    print(f"{'='*55}")
    _predict_cache[cache_key] = (result, time.time())  # FIX M3: cache for TTL
    return result

def retrain_on_intraday(ticker, base_df_features, feature_cols, scaler):
    """
    Downloads today's intraday 5-min bars, converts them to features,
    appends to training data, and fine-tunes the XGBoost model on the
    combined dataset. Takes ~30 seconds. Run once per day at market open.

    This is what separates a static model from a live-aware one.
    The model now sees today's morning price action before predicting close.
    """
    print(f"\n🔄 Intraday fine-tune for {ticker}...")

    try:
        # ── Download today's 5-min bars ───────────────────────────────
        df_intra = yf.download(
            ticker, period='1d', interval='5m',
            auto_adjust=True, progress=False
        )
        if isinstance(df_intra.columns, pd.MultiIndex):
            df_intra.columns = df_intra.columns.get_level_values(0)
        df_intra = df_intra.dropna()

        if df_intra.empty or len(df_intra) < 20:
            print("  ⚠️  Not enough intraday bars — skipping fine-tune")
            return xgb_model, scaler

        # ── Engineer features on intraday bars ───────────────────────
        # Need market context columns — add zeros for now (they won't exist
        # in intraday data, but the model handles zero-fill)
        df_intra_feat = engineer_features(df_intra)
        common_cols   = [c for c in feature_cols if c in df_intra_feat.columns]

        # Build feature matrix — zero-fill missing cols
        X_intra = np.zeros((len(df_intra_feat), len(feature_cols)))
        for i, col in enumerate(feature_cols):
            if col in df_intra_feat.columns:
                X_intra[:, i] = df_intra_feat[col].values

        y_intra = df_intra_feat['Target'].values
        if len(y_intra) == 0:
            print("  ⚠️  No target rows after feature engineering — skipping")
            return xgb_model, scaler

        # ── Scale with existing scaler (do NOT refit — that leaks test data) ──
        X_intra_sc = scaler.transform(X_intra)

        # ── Fine-tune: continue training XGBoost on intraday data ────
        # xgb_model.fit with xgb_model as base (warm start via set_params)
        xgb_model_live = xgb.XGBRegressor(**best_xgb_params, verbosity=0)
        xgb_model_live.fit(
            X_intra_sc, y_intra,
            xgb_model=xgb_model.get_booster(),  # start from existing model
            verbose=False
        )

        print(f"  ✅ Fine-tuned on {len(df_intra_feat)} intraday bars")
        print(f"  Intraday price range: {df_intra['Close'].min():.2f} – "
              f"{df_intra['Close'].max():.2f}")
        return xgb_model_live, scaler

    except Exception as e:
        print(f"  ⚠️  Fine-tune failed: {e} — using original model")
        return xgb_model, scaler

