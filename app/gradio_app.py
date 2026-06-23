def gradio_predict(ticker_input: str):
    ticker = (ticker_input or "").strip().upper() or CONFIG["ticker"]
    _current_ticker["value"] = ticker

    try:
        r   = predict_tomorrow(ticker)
        sym = r["symbol"]
        is_bull      = "BULL" in r["signal"]
        signal_color = "#4CAF50" if is_bull else "#F44336"
        pct_sign     = "+" if r["pct_change"] >= 0 else ""
        signal_label = "BULLISH" if is_bull else "BEARISH"

        model_rows = ""
        for label, key, color in [
            ("XGBoost",       "xgboost",       "#FF9800"),
            ("LightGBM",      "lightgbm",      "#64B5F6"),
            ("Random Forest", "random_forest", "#81C784"),
        ]:
            diff     = r[key] - r["live_price"]
            diff_pct = diff / max(r["live_price"], 0.01) * 100
            d_sign   = "+" if diff >= 0 else ""
            d_color  = "#4CAF50" if diff >= 0 else "#F44336"
            model_rows += f"""
  <tr style='border-bottom:1px solid #1a1a2e;'>
    <td style='padding:7px 10px;color:#aaaacc;font-size:12px;'>{label}</td>
    <td style='padding:7px 10px;color:{color};font-weight:600;'>{sym}{r[key]:.2f}</td>
    <td style='padding:7px 10px;color:{d_color};font-size:11px;'>{d_sign}{diff_pct:.2f}% vs live</td>
  </tr>"""

        sentiment_row = ""
        if "ensemble_sentiment_adjusted" in r:
            sent_score = r.get("sentiment_score", 0.0)
            sent_tilt  = r.get("sentiment_tilt_pct", 0.0)
            sent_color = "#4CAF50" if sent_score >= 0 else "#F44336"
            sentiment_row = f"""
  <tr style='border-bottom:1px solid #1a1a2e;'>
    <td style='padding:7px 10px;color:#aaaacc;font-size:12px;'>+ Sentiment</td>
    <td style='padding:7px 10px;color:#CE93D8;font-weight:600;'>{sym}{r['ensemble_sentiment_adjusted']:.2f}</td>
    <td style='padding:7px 10px;color:{sent_color};font-size:11px;'>score {sent_score:+.2f}, tilt {sent_tilt:+.3f}%</td>
  </tr>"""

        live_diff     = r["ensemble"] - r["live_price"]
        live_diff_pct = live_diff / max(r["live_price"], 0.01) * 100
        ld_sign       = "+" if live_diff >= 0 else ""
        ld_color      = "#4CAF50" if live_diff >= 0 else "#F44336"

        html = f"""<div class='pred-card'>
  <div style='display:flex;align-items:baseline;gap:10px;margin-bottom:16px;flex-wrap:wrap;'>
    <span style='font-size:13px;font-weight:700;color:{signal_color};letter-spacing:0.1em;'>{signal_label}</span>
    <span style='font-size:13px;color:#888;'>&#8594;</span>
    <span style='font-size:22px;font-weight:700;color:white;'>{sym}{r['ensemble']:.2f}</span>
    <span style='font-size:14px;color:{signal_color};'>({pct_sign}{r['pct_change']:.2f}%)</span>
    <span style='font-size:11px;color:#777;margin-left:auto;'>{r['ticker']}</span>
  </div>
  <table style='width:100%;border-collapse:collapse;'>
    <tr style='border-bottom:1px solid #1a1a2e;'>
      <td style='padding:7px 10px;color:#aaaacc;font-size:12px;'>Live price</td>
      <td style='padding:7px 10px;color:white;font-weight:600;'>{sym}{r['live_price']:.2f}</td>
      <td style='padding:7px 10px;color:#F44336;font-size:11px;'>{r.get('change_pct_today',0):+.2f}% today</td>
    </tr>
    <tr style='border-bottom:1px solid #1a1a2e;'>
      <td style='padding:7px 10px;color:#aaaacc;font-size:12px;'>Last close</td>
      <td style='padding:7px 10px;color:#cccccc;'>{sym}{r['current_price']:.2f}</td>
      <td style='padding:7px 10px;color:#777;font-size:11px;'>model base</td>
    </tr>
    {model_rows}
    <tr style='background:#0a0a1a;'>
      <td style='padding:9px 10px;color:#ffffff;font-size:13px;font-weight:600;'>Ensemble</td>
      <td style='padding:9px 10px;color:{signal_color};font-size:16px;font-weight:700;'>{sym}{r['ensemble']:.2f}</td>
      <td style='padding:9px 10px;color:{ld_color};font-size:12px;'>{ld_sign}{live_diff_pct:.2f}% vs live</td>
    </tr>
    {sentiment_row}
  </table>
  <div style='margin-top:12px;font-size:10px;color:#666;border-top:1px solid #1a1a2e;padding-top:8px;'>
    Data: {r['last_data_date']} &#124; Market: {r['market_state']}
    &#124; Live injected: {r['live_injected']}
    &#124; For: {r['prediction_date']}
    &#124; Educational only
  </div>
</div>"""
        chart = build_prediction_chart(r)
        return html, chart

    except Exception as e:
        import traceback
        err_html = f"""<div class='pred-card' style='border-color:#3a1a1a;'>
  <div style='color:#F44336;'>Error: {e}</div>
  <pre style='color:#888;font-size:10px;margin-top:8px;white-space:pre-wrap;'>{traceback.format_exc()}</pre>
</div>"""
        return err_html, _empty_fig("Prediction failed")

def run_agents(ticker_input: str):
    ticker = (ticker_input or "").strip().upper() or CONFIG["ticker"]
    _EMPTY = "<span style='color:#555;'>—</span>"

    if not KEYS.get("groq"):
        msg = _agent_section("NO GROQ KEY", "#FFB300",
                              "Add GROQ_API_KEY to Colab Secrets to enable agent reasoning.")
        return msg, msg, msg, msg, msg, msg

    try:
        ml_result = predict_tomorrow(ticker)
        final = trading_graph.invoke({
            "ticker":           ticker,
            "ml_signal":        ml_result,
            "shap_features":    list(zip(feature_cols, shap_values[-1].tolist())),
            "news_sentiment":   float(ml_result.get("sentiment_score", 0.0)),  # FIX: was frozen global sentiment_report (always CONFIG['ticker']); now uses per-ticker live sentiment already fetched by predict_tomorrow(ticker) above
            "technical_report": "",
            "sentiment_report": "",
            "bull_case":        "",
            "bear_case":        "",
            "final_decision":   "",
        })

        dec = final["final_decision"].upper()
        dec_color = "#4CAF50" if "BUY" in dec else ("#F44336" if "SELL" in dec else "#FFB300")
        verdict   = "BUY" if "BUY" in dec else ("SELL" if "SELL" in dec else "HOLD")

        decision_html = f"""<div class='agent-section' style='border-color:{dec_color};'>
  <div style='font-size:11px;font-weight:700;letter-spacing:0.12em;color:{dec_color};margin-bottom:10px;'>
    PORTFOLIO MANAGER — FINAL DECISION
  </div>
  <div style='font-size:28px;font-weight:700;color:{dec_color};margin-bottom:10px;'>{verdict}</div>
  <div style='color:#cccccc;font-size:13px;line-height:1.8;'>{final['final_decision']}</div>
</div>"""

        technical_html = _agent_section("&#128200; TECHNICAL ANALYST",  "#64B5F6", final.get("technical_report", _EMPTY))
        sentiment_html = _agent_section("&#128248; SENTIMENT ANALYST",  "#FFB300", final.get("sentiment_report", _EMPTY))
        bull_html      = _agent_section("&#9650; BULL RESEARCHER",      "#4CAF50", final.get("bull_case",        _EMPTY))
        bear_html      = _agent_section("&#9660; BEAR RESEARCHER",      "#F44336", final.get("bear_case",        _EMPTY))
        portfolio_html = _agent_section("&#127959; PORTFOLIO MANAGER",  "#CE93D8", final.get("final_decision",   _EMPTY))

        return decision_html, technical_html, sentiment_html, bull_html, bear_html, portfolio_html

    except Exception as e:
        import traceback
        err = _agent_section("ERROR", "#F44336",
            f"{e}<br><pre style='font-size:10px;color:#888;'>{traceback.format_exc()}</pre>")
        return err, err, err, err, err, err

def refresh_indian_watchlist(_=None) -> str:
    tickers = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
        "WIPRO.NS", "ADANIENT.NS", "BAJFINANCE.NS", "ICICIBANK.NS",
    ]
    rows = ""
    for t in tickers:
        try:
            info = get_live_price(t)
        except Exception as e:
            info = {"ticker": t, "error": f"fetch failed — {e}"}
        if info.get("error"):
            rows += f"<tr><td colspan='5' style='padding:6px 8px;color:#F44336;font-size:11px;'>{t}: {info['error']}</td></tr>"
            continue
        sign  = "+" if info["change"] >= 0 else ""
        clr   = "#4CAF50" if info["change"] >= 0 else "#F44336"
        name  = t.replace(".NS","").replace(".BO","")
        price = info["price"]
        sym   = "&#8377;"
        price_flag = " &#9888;" if (price < 50 and info["currency"] == "INR") else ""
        rows += f"""<tr style='border-bottom:1px solid #0f0f1e;'>
  <td style='padding:6px 8px;color:#cccccc;font-size:12px;'>{name}</td>
  <td style='padding:6px 8px;color:white;font-weight:600;font-size:12px;'>{sym}{price:,.2f}{price_flag}</td>
  <td style='padding:6px 8px;color:{clr};font-size:12px;'>{sign}{info['change']:.2f}</td>
  <td style='padding:6px 8px;color:{clr};font-size:11px;'>({sign}{info['change_pct']:.2f}%)</td>
  <td style='padding:6px 8px;color:#888;font-size:10px;'>{info.get("market_state","")}</td>
</tr>"""

    now = datetime.now().strftime("%H:%M:%S")
    return f"""<div class='watch-card'>
  <div style='font-size:11px;color:#8a8aaa;letter-spacing:0.1em;margin-bottom:12px;'>
    NSE LIVE &nbsp;&#124;&nbsp; auto-refresh 30s &nbsp;&#124;&nbsp; {now}
  </div>
  <table style='width:100%;border-collapse:collapse;'>
    <tr style='border-bottom:1px solid #1a1a2e;'>
      <th style='padding:5px 8px;color:#aaaacc;font-size:10px;text-align:left;letter-spacing:0.08em;'>STOCK</th>
      <th style='padding:5px 8px;color:#aaaacc;font-size:10px;text-align:left;letter-spacing:0.08em;'>PRICE</th>
      <th style='padding:5px 8px;color:#aaaacc;font-size:10px;text-align:left;letter-spacing:0.08em;'>CHG</th>
      <th style='padding:5px 8px;color:#aaaacc;font-size:10px;text-align:left;letter-spacing:0.08em;'>%</th>
      <th style='padding:5px 8px;color:#aaaacc;font-size:10px;text-align:left;letter-spacing:0.08em;'>STATUS</th>
    </tr>
    {rows}
  </table>
  <div style='font-size:10px;color:#555;margin-top:8px;'>
    Delayed ~15 min. NSE closes 15:30 IST. &#9888; = possible USD data leak.
  </div>
</div>"""

def refresh_live_ticker(ticker_input: str) -> str:
    ticker = (ticker_input or "").strip().upper() or CONFIG["ticker"]
    _current_ticker["value"] = ticker
    info = get_live_price(ticker)

    if info.get("error"):
        return f"""<div class='price-card' style='border-color:#3a1a1a;'>
  <div style='color:#888;font-size:11px;font-family:monospace;'>ERROR</div>
  <div style='color:#F44336;font-size:14px;margin-top:6px;'>{ticker}: {info["error"]}</div>
</div>"""

    sym       = "&#8377;" if info["currency"] == "INR" else "$"
    is_up     = info["change"] >= 0
    clr       = "#4CAF50" if is_up else "#F44336"
    arrow     = "&#9650;" if is_up else "&#9660;"
    sign      = "+" if is_up else ""
    card_cls  = "up" if is_up else "down"
    ms        = info.get("market_state", "")
    dot_color = "#4CAF50" if ms == "REGULAR" else ("#FFB300" if ("PRE" in ms or "POST" in ms) else "#666")
    vol       = f"{info['volume']:,}" if info.get('volume', 0) > 0 else "N/A"
    src       = info.get("source", "")
    ts        = info.get("timestamp", "")

    currency_warning = ""
    if ticker.endswith(".NS") and info["currency"] == "INR" and info["price"] < 10 and info.get('source') != 'NSE_API':
        currency_warning = "<div style='color:#FFB300;font-size:11px;margin-top:6px;'>&#9888; Possible USD data leak</div>"

    return f"""<div class='price-card {card_cls}'>
  <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>
    <span style='width:8px;height:8px;border-radius:50%;background:{dot_color};display:inline-block;'></span>
    <span style='color:#aaaacc;font-size:11px;letter-spacing:0.08em;'>{ms.upper()}</span>
    <span style='color:#555;font-size:11px;'>&#124;</span>
    <span style='color:#aaa;font-size:11px;'>{ts}</span>
    <span style='color:#555;font-size:11px;'>&#124;</span>
    <span style='color:#aaa;font-size:11px;'>{src}</span>
  </div>
  <div style='font-size:15px;font-weight:600;color:#ffffff;letter-spacing:0.1em;margin-bottom:4px;'>{ticker}</div>
  <div style='font-size:38px;font-weight:700;color:white;line-height:1.1;margin-bottom:6px;'>
    {sym}{info['price']:,.2f}
  </div>
  <div style='font-size:16px;color:{clr};margin-bottom:10px;'>
    {arrow} {sign}{info['change']:.2f} &nbsp; <span style='font-size:14px;'>({sign}{info['change_pct']:.2f}%)</span>
  </div>
  <div style='display:flex;gap:20px;font-size:11px;color:#888;border-top:1px solid #1a1a2e;padding-top:8px;'>
    <span>Prev close: {sym}{info['prev_close']:,.2f}</span>
    <span>Vol: {vol}</span>
    <span>CCY: {info['currency']}</span>
  </div>
  <div style='font-size:10px;color:#555;margin-top:6px;'>Prices delayed ~15 min (yfinance free tier)</div>
  {currency_warning}
</div>"""

def _agent_section(title: str, color: str, content: str) -> str:
    return f"""<div class='agent-section'>
  <div class='agent-section-title' style='color:{color};'>{title}</div>
  <div style='color:#cccccc;'>{content if content else '<span style="color:#555;">No output.</span>'}</div>
</div>"""

