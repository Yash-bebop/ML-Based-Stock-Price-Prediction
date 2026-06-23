class TradingState(TypedDict):
    ticker:           Annotated[str,   keep_latest]
    ml_signal:        Annotated[dict,  keep_latest]
    shap_features:    Annotated[list,  keep_latest]
    news_sentiment:   Annotated[float, keep_latest]
    technical_report: Annotated[str,   keep_latest]
    sentiment_report: Annotated[str,   keep_latest]
    bull_case:        Annotated[str,   keep_latest]
    bear_case:        Annotated[str,   keep_latest]
    final_decision:   Annotated[str,   keep_latest]

def technical_analyst(state: TradingState) -> dict:
    ml = state['ml_signal']
    sym        = ml.get('symbol', '$')
    ensemble   = ml.get('ensemble', 0.0)
    pct_change = ml.get('pct_change', 0.0)
    signal     = ml.get('signal', 'UNKNOWN')
    live_price = ml.get('live_price', 0.0)
    top_feats  = state.get('shap_features', [])[:5]

    prompt = f"""You are a quantitative technical analyst.
The ML pipeline produced these results for {state['ticker']}:
- Ensemble price forecast : {sym}{ensemble:.2f} ({pct_change:+.2f}%)
- Signal                  : {signal}
- Live price              : {sym}{live_price:.2f}
- Top 5 SHAP features     : {top_feats}

Write exactly 3 bullet points of technical analysis.
Be specific about what the numbers imply."""

    if llm is None:
        return {'technical_report': '[Groq key not set]'}
    report = llm.invoke(prompt).content
    return {'technical_report': report}

def sentiment_analyst(state: TradingState) -> dict:
    prompt = f"""You are a market sentiment analyst for {state['ticker']}.
News sentiment score: {state['news_sentiment']:.4f}  (scale: -1 very negative → +1 very positive)

Write exactly 2 bullet points interpreting this sentiment score
and what it implies for short-term price action."""

    if llm is None:
        return {'sentiment_report': '[Groq key not set]'}
    report = llm.invoke(prompt).content
    return {'sentiment_report': report}

def bull_researcher(state: TradingState) -> dict:
    prompt = f"""You are a BULLISH equity researcher for {state['ticker']}.
You have access to these reports:

TECHNICAL ANALYSIS:
{state['technical_report']}

SENTIMENT ANALYSIS:
{state['sentiment_report']}

Make the strongest possible bull case in exactly 3 points.
Focus on upside catalysts and why the stock should rise tomorrow."""

    if llm is None:
        return {'bull_case': '[Groq key not set]'}
    case = llm.invoke(prompt).content
    return {'bull_case': case}

def bear_researcher(state: TradingState) -> dict:
    prompt = f"""You are a BEARISH equity researcher for {state['ticker']}.
You have access to these reports:

TECHNICAL ANALYSIS:
{state['technical_report']}

SENTIMENT ANALYSIS:
{state['sentiment_report']}

Make the strongest possible bear case in exactly 3 points.
Focus on downside risks and why the stock might fall or underperform tomorrow."""

    if llm is None:
        return {'bear_case': '[Groq key not set]'}
    case = llm.invoke(prompt).content
    return {'bear_case': case}

def portfolio_manager(state: TradingState) -> dict:
    ml   = state['ml_signal']
    sym  = ml.get('symbol', '$')
    sig  = ml.get('signal', 'UNKNOWN')
    pct  = ml.get('pct_change', 0.0)
    ens  = ml.get('ensemble', 0.0)

    prompt = f"""You are the Portfolio Manager making the final trading decision for {state['ticker']}.

ML ENSEMBLE FORECAST : {sym}{ens:.2f} ({pct:+.2f}%) — {sig}

BULL RESEARCHER SAYS:
{state['bull_case']}

BEAR RESEARCHER SAYS:
{state['bear_case']}

Synthesize all of the above and give:
1. Decision    : BUY / SELL / HOLD
2. Position    : FULL / HALF / QUARTER
3. Rationale   : One clear paragraph explaining the decision,
                 acknowledging the strongest point from each side."""

    if llm is None:
        return {'final_decision': '[Groq key not set]'}
    decision = llm.invoke(prompt).content
    return {'final_decision': decision}

def run_groq_agent(ticker,metrics_df,sentiment_report,tomorrow_pred,wf_summary):
    combined = sentiment_report.get('combined_sentiment', sentiment_report.get('alt_news_sentiment',0))
    s={
        'ticker':ticker,'best_model':metrics_df['RMSE'].idxmin(),
        'best_rmse':float(metrics_df['RMSE'].min()),
        'best_da':float(metrics_df['Directional Acc %'].max()),
        'sig':metrics_df[metrics_df['Sig (p<0.05)']=='✅ Yes'].index.tolist(),
        'wf_mu':float(wf_summary['RMSE'].mean()),
        'wf_sd':float(wf_summary['RMSE'].std()),
        'tm':tomorrow_pred,'sent':sentiment_report,
    }
    prompt=(
        f"Senior quant analyst reviewing ML stock prediction pipeline.\n"
        f"Stock:{s['ticker']} | Best:{s['best_model']}(RMSE:{s['best_rmse']:.6f}) | DirAcc:{s['best_da']:.1f}%\n"
        f"Significant models (p<0.05 vs naive): {s['sig']}\n"
        f"Walk-Forward RMSE: {s['wf_mu']:.6f} ± {s['wf_sd']:.6f}\n"
        f"Sentiment: news={s['sent']['news_sentiment']:.4f} combined={combined:.4f} analyst={s['sent']['analyst_score']:.4f}\n"
        f"Tomorrow: live={s['tm']['symbol']}{s['tm']['live_price']:.2f} "        f"ensemble={s['tm']['symbol']}{s['tm']['ensemble']:.2f} ({s['tm']['pct_change']:+.2f}%) "        f"signal={s['tm']['signal']}\n\n"
        f"Write 5-point analysis: model performance, sentiment signals, "        f"prediction confidence, key risks, one-line recommendation. <300 words, bullets."
    )
    if not KEYS.get('groq'):
        return (f"LOCAL REPORT — {ticker}\n"
                f"Best: {s['best_model']} RMSE={s['best_rmse']:.6f} DirAcc={s['best_da']:.1f}%\n"
                f"WF RMSE: {s['wf_mu']:.6f} ± {s['wf_sd']:.6f}\n"
                f"Signal: {s['tm']['signal']} | Ensemble: {s['tm']['symbol']}{s['tm']['ensemble']:.2f} ({s['tm']['pct_change']:+.2f}%)\n"
                f"Live: {s['tm']['symbol']}{s['tm']['live_price']:.2f} | Market: {s['tm']['market_state']}\n"
                f"Note: Add GROQ_API_KEY to Colab Secrets for Llama 3 narrative report.")
    try:
        from groq import Groq
        resp=Groq(api_key=KEYS['groq']).chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role':'user','content':prompt}],
            max_tokens=500,temperature=0.3)
        return resp.choices[0].message.content
    except Exception as e: return f'Groq error: {e}'

