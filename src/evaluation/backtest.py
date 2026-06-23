def backtest_strategy(prices_true, opens_true, y_pred, dates, model_name,
                      cap=10000.0, tc=0.001, sl=0.05, tp=0.10):
    """
    Fixed backtest (v2 — execution timing actually matches the claim now):
    - prices_true : actual CLOSING prices (used to mark equity / check SL-TP)
    - opens_true  : actual OPENING prices (used as the trade EXECUTION price)
    - y_pred      : predicted log-returns (signal: >0 = LONG, <=0 = FLAT)
    - No lookahead: signal at day i is formed from information available at
      close of day i (preds[i]); the order can only realistically fill at the
      *next* tradable price, which is the open of day i+1 — opens_true[i+1].
      The previous version computed the signal this way but then executed at
      prices[i] (day i's own CLOSE), i.e. the same price the signal was
      derived from — not tradable in practice. That mismatch is fixed here.
    - Stop-loss 5%, take-profit 10%, transaction cost 0.1%
    - Caveat: still daily-bar granularity, so SL/TP are checked against the
      day's close, not real intrabar prices — a true intrabar SL/TP would
      need higher-frequency data.
    """
    capital, pos, entry = cap, 0.0, 0.0
    eq, bh, trades = [], [], []
    prices = np.array(prices_true, dtype=float)
    opens  = np.array(opens_true,  dtype=float)
    preds  = np.array(y_pred,      dtype=float)

    # Validate inputs — catch the log-return / price confusion early
    if prices.mean() < 1.0:
        raise ValueError(
            f'prices_true looks like log-returns (mean={prices.mean():.4f}). '
            f'Pass actual closing prices, not y_test.')

    bhs = cap / opens[1]   # buy-and-hold benchmark also enters at a tradable open

    for i in range(len(prices) - 1):
        # Decision uses info available at close of day i (preds[i]); earliest
        # realistic fill is the OPEN of day i+1 — not the close the signal came from.
        exec_price = opens[i + 1]
        mark_price = prices[i + 1]   # day i+1 close — used to mark equity / SL-TP
        sig = 1 if preds[i] > 0 else -1

        # Check stop-loss / take-profit before considering a new signal
        if pos > 0 and entry > 0:
            r = (mark_price - entry) / entry
            if r <= -sl or r >= tp:
                capital = pos * mark_price * (1 - tc)
                trades.append({'type': 'STP/TP', 'ret': r})
                pos, entry = 0.0, 0.0

        # Open / close position — at next day's OPEN, not today's close
        if sig == 1 and pos == 0:
            pos    = (capital * (1 - tc)) / exec_price
            entry  = exec_price
            capital = 0.0
            trades.append({'type': 'BUY', 'ret': None})
        elif sig == -1 and pos > 0:
            r       = (exec_price - entry) / entry
            capital = pos * exec_price * (1 - tc)
            trades.append({'type': 'SELL', 'ret': r})
            pos, entry = 0.0, 0.0

        eq.append(capital + (pos * mark_price if pos > 0 else 0))
        bh.append(bhs * mark_price)

    # Close any open position at end
    if pos > 0:
        eq[-1] = pos * prices[-1] * (1 - tc)

    equity = np.array(eq); bha = np.array(bh)
    tr_ret = (equity[-1] - cap) / cap * 100
    bh_ret = (bha[-1]    - cap) / cap * 100

    dr     = np.diff(equity) / np.maximum(equity[:-1], 1e-10)
    sharpe = (dr.mean() / (dr.std() + 1e-10)) * np.sqrt(252)
    mdd    = ((equity - np.maximum.accumulate(equity)) /
               np.maximum.accumulate(equity)).min() * 100
    trets  = [t['ret'] for t in trades if t.get('ret') is not None]
    wr     = sum(r > 0 for r in trets) / max(len(trets), 1) * 100

    print(f'\n💰 {model_name}: Return={tr_ret:+.1f}% | B&H={bh_ret:+.1f}% | '
          f'Sharpe={sharpe:.2f} | MDD={mdd:.1f}% | '
          f'WinRate={wr:.1f}% | Trades={len(trets)}')

    fig = go.Figure()
    plot_dates = dates[1:len(equity)+1]  # equity[i] now marks day i+1 (post execution-timing fix)
    fig.add_trace(go.Scatter(x=list(plot_dates), y=equity,
                              name='Strategy', line=dict(color='#4CAF50', width=2)))
    fig.add_trace(go.Scatter(x=list(plot_dates),    y=bha,
                              name='Buy & Hold',
                              line=dict(color='#2196F3', width=2, dash='dash')))
    fig.add_hline(y=cap, line_dash='dot', line_color='white', opacity=0.5)
    fig.update_layout(
        template='plotly_dark',
        title=f'{model_name} vs Buy & Hold | Sharpe={sharpe:.2f}',
        height=420)
    fname = f"backtest_{model_name.replace(' ', '_')}.html"
    fig.write_html(fname); fig.show()

    return pd.DataFrame({
        'Date'    : plot_dates,
        'Strategy': equity,
        'BuyHold' : bha
    })

def portfolio_risk_report(daily_returns, label='Strategy'):
    r = np.array(daily_returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 10:
        print(f"  {label}: insufficient data"); return {}

    ann_ret  = (1 + r.mean())**252 - 1
    ann_vol  = r.std() * np.sqrt(252)
    sharpe   = ann_ret / (ann_vol + 1e-10)

    downside = r[r < 0]
    down_dev = np.sqrt((downside**2).mean()) * np.sqrt(252) if len(downside) > 0 else 1e-10
    sortino  = ann_ret / (down_dev + 1e-10)

    var_95  = np.percentile(r, 5)
    var_99  = np.percentile(r, 1)
    cvar_95 = r[r <= var_95].mean()
    cvar_99 = r[r <= var_99].mean()

    cum_ret   = np.cumprod(1 + r)
    peak      = np.maximum.accumulate(cum_ret)
    dd_series = (cum_ret - peak) / peak
    max_dd    = dd_series.min()
    calmar    = ann_ret / (abs(max_dd) + 1e-10)

    gains  = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    omega  = gains / (losses + 1e-10)
    profit_factor = gains / (losses + 1e-10)

    skew = float(_stats.skew(r))
    kurt = float(_stats.kurtosis(r))   # excess kurtosis; > 0 = fatter-than-normal tails
    tail_ratio = abs(np.percentile(r, 95)) / (abs(np.percentile(r, 5)) + 1e-10)

    metrics = {
        'label': label, 'Ann Return %': round(ann_ret*100,2), 'Ann Vol %': round(ann_vol*100,2),
        'Sharpe': round(sharpe,3), 'Sortino': round(sortino,3), 'Calmar': round(calmar,3),
        'Omega': round(omega,3), 'Profit Factor': round(profit_factor,3),
        'VaR 95% (1d)': round(var_95*100,3), 'CVaR 95% (1d)': round(cvar_95*100,3),
        'VaR 99% (1d)': round(var_99*100,3), 'CVaR 99% (1d)': round(cvar_99*100,3),
        'Max DD %': round(max_dd*100,2), 'Skewness': round(skew,3),
        'Exc Kurtosis': round(kurt,3), 'Tail Ratio': round(tail_ratio,3),
    }
    print(f"\n  ── {label} ──")
    for k, v in metrics.items():
        if k == 'label': continue
        flag = ''
        if k == 'Sharpe':        flag = ' ✅' if v > 1   else (' ⚠️' if v > 0.5 else ' ❌')
        if k == 'Sortino':       flag = ' ✅' if v > 1.5 else (' ⚠️' if v > 0.8 else ' ❌')
        if k == 'Calmar':        flag = ' ✅' if v > 1   else (' ⚠️' if v > 0.3 else ' ❌')
        if k == 'CVaR 95% (1d)': flag = ' ✅' if v > -3  else ' ❌'
        if k == 'Exc Kurtosis':  flag = ' ⚠️ fat tails' if v > 3 else ''
        print(f"    {k:<20}: {v}{flag}")
    return metrics

