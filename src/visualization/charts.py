def build_prediction_chart(r: dict):
    hist       = r.get("history", {})
    hist_dates = pd.to_datetime(hist.get("dates", []))
    hist_close = hist.get("close", [])

    fig = go.Figure()
    if len(hist_dates) > 0:
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_close, name="Actual Close",
            mode="lines", line=dict(color="#9aa0b0", width=2),
            hovertemplate="%{x|%b %d}<br>Close: %{y:.2f}<extra></extra>"))
        last_date, last_close = hist_dates[-1], hist_close[-1]
    else:
        last_date, last_close = pd.Timestamp.today(), r.get("current_price", 0)

    next_date = pd.to_datetime(r["prediction_date"])
    for name, key, color in [
        ("XGBoost",       "xgboost",       "#FF9800"),
        ("LightGBM",      "lightgbm",      "#64B5F6"),
        ("Random Forest", "random_forest", "#81C784"),
        ("Ensemble",      "ensemble",      "#FFFFFF"),
    ]:
        px = r.get(key)
        if px is None:
            continue
        fig.add_trace(go.Scatter(
            x=[last_date, next_date], y=[last_close, px],
            mode="lines+markers", name=name,
            line=dict(color=color, width=1.8, dash="dot"),
            marker=dict(size=[0, 10], color=color),
            hovertemplate=f"{name}<br>%{{x|%b %d}}: %{{y:.2f}}<extra></extra>"))

    if "ensemble_sentiment_adjusted" in r:
        fig.add_trace(go.Scatter(
            x=[last_date, next_date],
            y=[last_close, r["ensemble_sentiment_adjusted"]],
            mode="lines+markers", name="Ensemble + Sentiment",
            line=dict(color="#CE93D8", width=1.8, dash="dot"),
            marker=dict(size=[0, 10], color="#CE93D8"),
            hovertemplate="Ensemble+Sentiment<br>%{x|%b %d}: %{y:.2f}<extra></extra>"))

    return _apply_dark(fig,
        title=f"{r.get('ticker','')} — Per-Model Forecast ({r.get('prediction_date','')})",
        height=460)

def _chart_lstm_training(ticker):
    """LSTM training vs validation loss curves."""
    if 'history' not in globals() or history is None:
        return _empty_fig("Run Step 7 (LSTM) first")
    from plotly.subplots import make_subplots
    ep = list(range(1, len(history.history["loss"])+1))
    best = int(np.argmin(history.history["val_loss"])) + 1
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Train vs Validation Loss (Huber)", "Train vs Validation MAE"],
        horizontal_spacing=0.12)

    fig.add_trace(go.Scatter(x=ep, y=history.history["loss"], name="Train Loss",
        line=dict(color="#2196F3", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep, y=history.history["val_loss"], name="Val Loss",
        line=dict(color="#F44336", width=2, dash="dash")), row=1, col=1)
    fig.add_vline(x=best, line=dict(color="#FFB300", dash="dot", width=1.5), row=1, col=1)
    fig.add_annotation(x=best, yref="y", y=max(history.history["val_loss"]),
        text=f"Best ({best})", font=dict(color="#FFB300", size=10), showarrow=False)

    fig.add_trace(go.Scatter(x=ep, y=history.history["mae"], name="Train MAE",
        line=dict(color="#4CAF50", width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=ep, y=history.history["val_mae"], name="Val MAE",
        line=dict(color="#FF9800", width=2, dash="dash")), row=1, col=2)
    fig.add_vline(x=best, line=dict(color="#FFB300", dash="dot", width=1.5), row=1, col=2)

    gap = np.array(history.history["val_loss"]) - np.array(history.history["loss"])
    overfit_flag = "  ⚠️ overfit signal" if gap[-1] > gap[0] * 2 else "  ✅ ok"
    fig.update_layout(
        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
        font=dict(color="#cccccc", family="JetBrains Mono, monospace"),
        margin=dict(l=60, r=20, t=80, b=90), height=480,
        legend=dict(orientation="h", y=-0.2, x=0, font=dict(color="#cccccc"), bgcolor="rgba(0,0,0,0)"),
        title=dict(text=f"{ticker} — LSTM Training Diagnostics{overfit_flag}",
                   font=dict(color="#ffffff", size=13), x=0),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#aaaacc"; ann.font.size = 11
    for ax in [fig.layout.xaxis, fig.layout.xaxis2]:
        ax.update(gridcolor="#1a1a2e", tickfont=dict(color="#aaaacc"))
    for ax in [fig.layout.yaxis, fig.layout.yaxis2]:
        ax.update(gridcolor="#1a1a2e", tickfont=dict(color="#aaaacc"))
    return fig

def _chart_hmm_garch(ticker):
    """HMM regime overlay + GARCH volatility."""
    if 'df_features' not in globals() or df_features is None:
        return _empty_fig("Run Step 6b (HMM/GARCH) first")
    from plotly.subplots import make_subplots
    df = df_features.copy()
    fig = make_subplots(rows=3, cols=1,
        subplot_titles=["Price with HMM Regime Overlay", "HMM Regime State", "GARCH(1,1) Conditional Volatility"],
        vertical_spacing=0.10, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25])

    # Price
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].squeeze(),
        name="Close", line=dict(color="#9aa0b0", width=1.5)), row=1, col=1)

    # Regime shading
    if "HMM_Regime" in df.columns:
        regimes = df["HMM_Regime"].values
        bull_color = "rgba(76,175,80,0.12)"; bear_color = "rgba(244,67,54,0.12)"
        start_i = 0
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i-1] or i == len(regimes)-1:
                clr = bull_color if regimes[start_i] == 0 else bear_color
                fig.add_vrect(x0=df.index[start_i], x1=df.index[min(i, len(df)-1)],
                    fillcolor=clr, line_width=0, row=1, col=1)
                start_i = i
        fig.add_trace(go.Scatter(x=df.index, y=regimes.astype(float),
            name="Regime (0=Bull, 1=Bear)", mode="lines",
            line=dict(color="#FFB300", width=1.5, shape="hv")), row=2, col=1)

    # GARCH vol
    if "GARCH_Vol" in df.columns:
        gv = df["GARCH_Vol"].squeeze()
        fig.add_trace(go.Scatter(x=df.index, y=gv, name="GARCH Vol",
            line=dict(color="#CE93D8", width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=list(df.index)+list(df.index[::-1]),
            y=list(gv)+[0]*len(gv),
            fill="toself", fillcolor="rgba(206,147,216,0.10)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False), row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
        font=dict(color="#cccccc", family="JetBrains Mono, monospace"),
        margin=dict(l=60, r=20, t=80, b=80), height=640,
        legend=dict(orientation="h", y=-0.14, x=0, font=dict(color="#cccccc"), bgcolor="rgba(0,0,0,0)"),
        title=dict(text=f"{ticker} — HMM Regime Detection + GARCH(1,1) Volatility",
                   font=dict(color="#ffffff", size=13), x=0),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#aaaacc"; ann.font.size = 11
    for ax in [fig.layout.xaxis, fig.layout.xaxis2, fig.layout.xaxis3]:
        ax.update(gridcolor="#1a1a2e", tickfont=dict(color="#aaaacc"))
    for ax in [fig.layout.yaxis, fig.layout.yaxis2, fig.layout.yaxis3]:
        ax.update(gridcolor="#1a1a2e", tickfont=dict(color="#aaaacc"))
    return fig

def _build_ticker_chart_data(ticker: str) -> dict:
    """
    Downloads 600d of history for `ticker`, engineers features, scores the
    existing trained models (xgb_model / lgb_model / rf_model) on the new
    ticker's test split, and computes SHAP — all without re-training.
    Results are cached in _CHART_CACHE so repeated chart switches are instant.
    """
    ticker = ticker.strip().upper()
    if ticker in _CHART_CACHE:
        return _CHART_CACHE[ticker]

    from datetime import datetime, timedelta
    import shap as _shap_mod

    end   = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=600)).strftime('%Y-%m-%d')

    # Download + feature-engineer for the new ticker
    df_t = download_stock_data(ticker, start, end)
    df_t = add_market_context(df_t, ticker, start=start, end=end)
    df_t = engineer_features(df_t)

    # Split exactly as the main pipeline does
    X_tr_t, X_te_t, y_tr_t, y_te_t, sc_t, td_t = split_data(
        df_t, feature_cols, CONFIG['test_size']
    )

    # Score existing trained models — NO retraining
    yp_xgb_t = xgb_model.predict(X_te_t)
    yp_lgb_t = lgb_model.predict(X_te_t)
    yp_rf_t  = rf_model.predict(X_te_t)
    _ew       = CONFIG['ensemble_weights']
    yp_ens_t  = (yp_xgb_t * _ew['xgb'] +
                 yp_lgb_t * _ew['lgb'] +
                 yp_rf_t  * _ew['rf'])

    # Metrics for all four models
    _m_list = []
    for _name, _pred in [("XGBoost",       yp_xgb_t),
                          ("LightGBM",      yp_lgb_t),
                          ("Random Forest", yp_rf_t),
                          ("Ensemble",      yp_ens_t)]:
        _m_list.append(evaluate_model(y_te_t, _pred, _name))
    mdf_t = pd.DataFrame(_m_list).set_index('Model')

    # SHAP on the new ticker's test set
    try:
        _exp      = _shap_mod.TreeExplainer(xgb_model)
        _shap_t   = _exp.shap_values(X_te_t)
    except Exception:
        _shap_t   = None

    result = {
        "df_features":     df_t,
        "metrics_df":      mdf_t,
        "y_test":          y_te_t,
        "y_pred_ensemble": yp_ens_t,
        "test_dates":      td_t,
        "shap_values":     _shap_t,
    }
    _CHART_CACHE[ticker] = result
    print(f"  📊 Chart data built + cached for {ticker}")
    return result

def _chart_accuracy_dashboard(ticker):
    """4-metric accuracy dashboard for all models."""
    _res = _build_ticker_chart_data(ticker)
    from plotly.subplots import make_subplots
    _df  = _res["metrics_df"].copy()
    models = list(_df.index)
    palette = ["#2196F3","#9C27B0","#FF9800","#4CAF50","#F44336","#00BCD4","#CE93D8","#FF5722"]
    colors  = [palette[i % len(palette)] for i in range(len(models))]

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["RMSE (↓ better)", "Directional Accuracy % (↑ better)",
                        "R² Score (↑ better)", "IC Spearman (↑ better | 0.05 threshold)"],
        vertical_spacing=0.22, horizontal_spacing=0.12)

    def _bar(row, col, metric, ascending, ref=None):
        if metric not in _df.columns:
            return
        vals = _df[metric].fillna(0)
        best = vals.idxmin() if ascending else vals.idxmax()
        bar_colors = ["#FFD700" if m == best else colors[i] for i, m in enumerate(models)]
        fig.add_trace(go.Bar(x=models, y=list(vals), marker_color=bar_colors,
            hovertemplate=f"{metric}: %{{y:.4f}}<extra></extra>", showlegend=False), row=row, col=col)
        if ref is not None:
            fig.add_shape(type="line", x0=-0.5, x1=len(models)-0.5, y0=ref, y1=ref,
                line=dict(color="#FFB300", width=1.2, dash="dot"), row=row, col=col)

    _bar(1, 1, "RMSE",             ascending=True)
    _bar(1, 2, "Directional Acc %", ascending=False, ref=50)
    _bar(2, 1, "R2",                ascending=False)
    _bar(2, 2, "IC (Spearman)",     ascending=False, ref=0.05)

    fig.update_layout(
        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
        font=dict(color="#cccccc", family="JetBrains Mono, monospace"),
        margin=dict(l=50, r=20, t=80, b=60),
        height=600,
        title=dict(text=f"{ticker} — Accuracy Dashboard (gold = best per metric)",
                   font=dict(color="#ffffff", size=13), x=0.0),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#aaaacc"
        ann.font.size  = 11
    for ax in [fig.layout.xaxis, fig.layout.xaxis2, fig.layout.xaxis3, fig.layout.xaxis4]:
        ax.update(gridcolor="#1a1a2e", tickfont=dict(color="#aaaacc"))
    for ax in [fig.layout.yaxis, fig.layout.yaxis2, fig.layout.yaxis3, fig.layout.yaxis4]:
        ax.update(gridcolor="#1a1a2e", tickfont=dict(color="#aaaacc"))
    return fig

def _chart_multistep(ticker):
    """Multi-step forecast price curves."""
    if 'multistep_df' not in globals() or multistep_df is None:
        return _empty_fig("Run Step 11 (Multi-Step Forecasting) first")
    df = multistep_df.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Actual_Close"], name="Actual Close",
        line=dict(color="#9aa0b0", width=2),
        hovertemplate="%{x|%b %d}<br>Close: %{y:.2f}<extra></extra>"))
    colors = ["#2196F3","#4CAF50","#FF9800","#F44336"]
    for h, color in zip(CONFIG["multistep_days"], colors):
        col = f"Pred_T{h}_price"
        if col not in df.columns:
            continue
        s = df[col].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=f"T+{h}d forecast",
            mode="lines", line=dict(color=color, dash="dash", width=1.6),
            hovertemplate=f"T+{h}d: %{{y:.2f}}<extra></extra>"))
    return _apply_dark(fig, title=f"{ticker} — Multi-Step Price Forecast (T+1/3/5/10 days)", height=500)

def _chart_quant_risk(ticker):
    """Drawdown, ensemble weights, return distribution with VaR/CVaR."""
    _res           = _build_ticker_chart_data(ticker)
    _has_backtest  = (ticker.upper() == CONFIG["ticker"].upper()) and (backtest_df is not None)  # backtest only exists for startup ticker
    _has_raw_preds = (_res["y_pred_ensemble"] is not None and _res["test_dates"] is not None
                      and _res["df_features"] is not None)
    from plotly.subplots import make_subplots

    # Prefer backtest_df returns (include SL/TP/costs); else reconstruct from raw predictions
    if _has_backtest:
        _strat_ret = backtest_df["Strategy"].pct_change().dropna().values
        _bh_ret    = backtest_df["BuyHold"].pct_change().dropna().values
        # x-axis: use Date column if present, else numeric index → converted to string
        _dd_x = (list(backtest_df["Date"].iloc[1:].values)
                 if "Date" in backtest_df.columns
                 else list(backtest_df.index[1:]))
    else:
        _prices = _res["df_features"].loc[_res["test_dates"], "Close"].squeeze().values
        _y_ens  = np.array(_res["y_pred_ensemble"])
        _min_n  = min(len(_prices)-1, len(_y_ens))
        _sig    = np.where(_y_ens[:_min_n] > 0, 1, -1)
        _close  = _prices[1:_min_n+1]; _prev = _prices[:_min_n]
        _strat_ret = (_close - _prev) / (_prev + 1e-10) * _sig
        _bh_ret    = (_close - _prev) / (_prev + 1e-10)
        _dd_x = list(_res["test_dates"][1:_min_n+1])

    fig = make_subplots(rows=3, cols=1,
        subplot_titles=["Drawdown — Strategy vs Buy & Hold",
                        "Data-Driven Ensemble Weights",
                        "Return Distribution with VaR/CVaR"],
        vertical_spacing=0.14, row_heights=[0.38, 0.24, 0.38])

    # Drawdown
    def _dd(r):
        cum = np.cumprod(1+r); peak = np.maximum.accumulate(cum)
        return (cum-peak)/(peak+1e-10)*100
    dd_s = _dd(_strat_ret); dd_b = _dd(_bh_ret)
    dd_x = _dd_x  # set above based on whether backtest_df is available
    fig.add_trace(go.Scatter(x=dd_x, y=list(dd_s), name="Strategy DD",
        fill="tozeroy", fillcolor="rgba(76,175,80,0.18)", line=dict(color="#4CAF50", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd_x, y=list(dd_b), name="Buy & Hold DD",
        fill="tozeroy", fillcolor="rgba(33,150,243,0.12)", line=dict(color="#2196F3", width=1, dash="dash")), row=1, col=1)

    # Weights
    _ew = CONFIG["ensemble_weights"]
    _mnames = list(_ew.keys()); _wvals = list(_ew.values())
    _wcolors = ["#FF9800","#64B5F6","#81C784"][:len(_mnames)]
    fig.add_trace(go.Bar(x=_mnames, y=_wvals, name="Weights",
        marker_color=_wcolors, showlegend=False,
        text=[f"{w:.4f}" for w in _wvals], textposition="outside",
        textfont=dict(color="#cccccc", size=11)), row=2, col=1)
    fig.add_shape(type="line", x0=-0.5, x1=len(_mnames)-0.5, y0=1/3, y1=1/3,
        line=dict(color="#FFB300", dash="dot", width=1.2), row=2, col=1)

    # Return distribution
    _bins = np.histogram(_strat_ret, bins=60)
    fig.add_trace(go.Bar(x=list(0.5*(_bins[1][:-1]+_bins[1][1:])), y=list(_bins[0]),
        name="Returns", marker_color="#2196F3", opacity=0.55, showlegend=False), row=3, col=1)
    for pct, clr, lbl in [(5,"#FFB300","VaR 95%"),(1,"#F44336","VaR 99%")]:
        var = np.percentile(_strat_ret, pct)
        cvar = _strat_ret[_strat_ret<=var].mean()
        fig.add_vline(x=var,  line=dict(color=clr, dash="dash", width=1.5), row=3, col=1)
        fig.add_vline(x=cvar, line=dict(color=clr, dash="dot",  width=1.2), row=3, col=1)
        fig.add_annotation(x=var, y=0, yref="y3", text=f"{lbl}", showarrow=False,
            font=dict(color=clr, size=9), yanchor="bottom")

    fig.update_layout(
        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
        font=dict(color="#cccccc", family="JetBrains Mono, monospace"),
        margin=dict(l=60, r=20, t=80, b=90), height=720,
        legend=dict(orientation="h", y=-0.12, x=0, font=dict(color="#cccccc"), bgcolor="rgba(0,0,0,0)"),
        title=dict(text=f"{ticker} — Quant Risk: Drawdown · Weights · VaR/CVaR",
                   font=dict(color="#ffffff", size=13), x=0),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#aaaacc"; ann.font.size = 11
    for ax in [fig.layout.xaxis, fig.layout.xaxis2, fig.layout.xaxis3]:
        ax.update(gridcolor="#1a1a2e", tickfont=dict(color="#aaaacc"))
    for ax in [fig.layout.yaxis, fig.layout.yaxis2, fig.layout.yaxis3]:
        ax.update(gridcolor="#1a1a2e", tickfont=dict(color="#aaaacc"))
    return fig

def _chart_shap(ticker):
    """SHAP feature importance as a horizontal bar chart."""
    _res = _build_ticker_chart_data(ticker)
    if _res["shap_values"] is None:
        return _empty_fig("SHAP computation failed for this ticker")
    mean_abs = np.abs(_res["shap_values"]).mean(axis=0)
    idx = np.argsort(mean_abs)[-30:][::-1]
    feats = [feature_cols[i] for i in idx]
    vals  = mean_abs[idx]
    palette = ["#CE93D8" if v == vals[0] else "#2196F3" for v in vals]
    fig = go.Figure(go.Bar(x=list(vals), y=feats, orientation="h",
        marker_color=palette, opacity=0.85,
        hovertemplate="%{y}<br>Mean |SHAP|: %{x:.5f}<extra></extra>"))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _apply_dark(fig, title=f"{ticker} — Top 30 SHAP Feature Importances (XGBoost)", height=680)

def _chart_residuals(ticker):
    """Residual distribution + QQ plot for ensemble model."""
    _res  = _build_ticker_chart_data(ticker)
    from plotly.subplots import make_subplots
    from scipy import stats as _scs
    resid = np.array(_res["y_test"]) - np.array(_res["y_pred_ensemble"])
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Residual Distribution (Ensemble)", "Q-Q Plot vs Normal"],
        horizontal_spacing=0.12)

    # Histogram
    fig.add_trace(go.Histogram(x=resid, nbinsx=60, name="Residuals",
        marker_color="#2196F3", opacity=0.75,
        hovertemplate="Residual: %{x:.5f}<br>Count: %{y}<extra></extra>"), row=1, col=1)
    _xr = np.linspace(resid.min(), resid.max(), 200)
    _kde = _scs.norm.pdf(_xr, resid.mean(), resid.std()) * len(resid) * (resid.max()-resid.min())/60
    fig.add_trace(go.Scatter(x=_xr, y=_kde, name="Normal fit",
        line=dict(color="#FFB300", width=2, dash="dash")), row=1, col=1)

    # QQ
    qq = _scs.probplot(resid, dist="norm")
    theoretical = qq[0][0]; sample = qq[0][1]
    fig.add_trace(go.Scatter(x=theoretical, y=sample, mode="markers", name="QQ",
        marker=dict(color="#CE93D8", size=4, opacity=0.7),
        hovertemplate="Theoretical: %{x:.3f}<br>Sample: %{y:.5f}<extra></extra>"), row=1, col=2)
    _qmin, _qmax = theoretical.min(), theoretical.max()
    fig.add_trace(go.Scatter(x=[_qmin, _qmax], y=[_qmin*qq[1][0]+qq[1][1], _qmax*qq[1][0]+qq[1][1]],
        mode="lines", name="Perfect normal", line=dict(color="#FFB300", dash="dash", width=1.5)), row=1, col=2)

    skew = float(_scs.skew(resid)); kurt = float(_scs.kurtosis(resid))
    fig.add_annotation(text=f"mean={resid.mean():.5f} | std={resid.std():.5f}<br>skew={skew:.3f} | excess-kurt={kurt:.3f}",
        xref="paper", yref="paper", x=0, y=-0.18, showarrow=False, font=dict(color="#888", size=10))

    fig.update_layout(
        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
        font=dict(color="#cccccc", family="JetBrains Mono, monospace"),
        margin=dict(l=60, r=20, t=80, b=100), height=500,
        legend=dict(orientation="h", y=-0.28, x=0, font=dict(color="#cccccc"), bgcolor="rgba(0,0,0,0)"),
        title=dict(text=f"{ticker} — Ensemble Residual Distribution & Q-Q Plot",
                   font=dict(color="#ffffff", size=13), x=0),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#aaaacc"; ann.font.size = 11
    for ax in [fig.layout.xaxis, fig.layout.xaxis2]:
        ax.update(gridcolor="#1a1a2e", tickfont=dict(color="#aaaacc"))
    for ax in [fig.layout.yaxis, fig.layout.yaxis2]:
        ax.update(gridcolor="#1a1a2e", tickfont=dict(color="#aaaacc"))
    return fig

def _chart_walkforward(ticker):
    """Walk-forward RMSE + directional accuracy drift."""
    if 'wf_results' not in globals() or wf_results is None:
        return _empty_fig("Run Step 8 (Walk-Forward Validation) first")
    from plotly.subplots import make_subplots
    wf = wf_results.copy()
    x  = list(range(1, len(wf) + 1))
    fig = make_subplots(rows=2, cols=1,
        subplot_titles=["RMSE per Walk-Forward Window", "Directional Accuracy % per Window"],
        vertical_spacing=0.18, shared_xaxes=True)

    rmse_v = wf["RMSE"].values
    rmse_m = rmse_v.mean(); rmse_s = rmse_v.std()
    fig.add_trace(go.Bar(x=x, y=list(rmse_v), name="Window RMSE",
        marker_color="#2196F3", opacity=0.8,
        hovertemplate="Window %{x}<br>RMSE: %{y:.6f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[rmse_m]*len(x), name=f"Mean RMSE ({rmse_m:.5f})",
        line=dict(color="#FFB300", dash="dash", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x+x[::-1],
        y=list(rmse_v+rmse_s)+list((rmse_v-rmse_s)[::-1]),
        fill="toself", fillcolor="rgba(33,150,243,0.12)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"), row=1, col=1)

    if "Directional Acc %" in wf.columns:
        da_v = wf["Directional Acc %"].values
        fig.add_trace(go.Scatter(x=x, y=list(da_v), name="Dir Acc %",
            mode="lines+markers", line=dict(color="#4CAF50", width=2),
            marker=dict(size=6, color="#4CAF50"),
            hovertemplate="Window %{x}<br>Dir Acc: %{y:.1f}%<extra></extra>"), row=2, col=1)
        fig.add_shape(type="line", x0=0.5, x1=len(x)+0.5, y0=50, y1=50,
            line=dict(color="#F44336", dash="dot", width=1.2), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
        font=dict(color="#cccccc", family="JetBrains Mono, monospace"),
        margin=dict(l=60, r=20, t=80, b=80), height=580,
        legend=dict(orientation="h", y=-0.15, x=0, font=dict(color="#cccccc"), bgcolor="rgba(0,0,0,0)"),
        title=dict(text=f"{ticker} — Walk-Forward Performance Drift", font=dict(color="#ffffff", size=13), x=0),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#aaaacc"; ann.font.size = 11
    for ax in [fig.layout.xaxis, fig.layout.xaxis2]:
        ax.update(gridcolor="#1a1a2e", tickfont=dict(color="#aaaacc"))
    for ax in [fig.layout.yaxis, fig.layout.yaxis2]:
        ax.update(gridcolor="#1a1a2e", tickfont=dict(color="#aaaacc"))
    return fig

def build_quant_chart(chart_name: str, ticker_input: str) -> go.Figure:
    """Route a chart name to its builder. All builders safe-fail with _empty_fig."""
    ticker = (ticker_input or "").strip().upper() or CONFIG["ticker"]
    try:
        if chart_name == "Prediction Forecast":
            r = predict_tomorrow(ticker)
            return build_prediction_chart(r)
        elif chart_name == "Model Comparison (RMSE · R² · Dir Acc)":
            return _chart_model_comparison(ticker)
        elif chart_name == "Accuracy Dashboard (4-metric)":
            return _chart_accuracy_dashboard(ticker)
        elif chart_name == "Walk-Forward Drift":
            return _chart_walkforward(ticker)
        elif chart_name == "HMM Regime + GARCH Volatility":
            return _chart_hmm_garch(ticker)
        elif chart_name == "Residual Distribution & Q-Q Plot":
            return _chart_residuals(ticker)
        elif chart_name == "Multi-Step Price Forecast":
            return _chart_multistep(ticker)
        elif chart_name == "Quant Risk (Drawdown · VaR · Weights)":
            return _chart_quant_risk(ticker)
        elif chart_name == "SHAP Feature Importance":
            return _chart_shap(ticker)
        elif chart_name == "LSTM Training Curves":
            return _chart_lstm_training(ticker)
        else:
            return _empty_fig("Select a chart from the menu")
    except Exception as e:
        import traceback
        return _empty_fig(f"{e}\n{traceback.format_exc()[:300]}")

def _chart_model_comparison(ticker):
    """RMSE / R² / Directional Accuracy bar chart for all models."""
    _res = _build_ticker_chart_data(ticker)
    _df  = _res["metrics_df"].copy()
    metrics_to_plot = [
        ("RMSE", True,  "RMSE (lower is better)",             "#2196F3"),
        ("R2",   False, "R² Score (higher is better)",        "#4CAF50"),
        ("Directional Acc %", False, "Directional Accuracy %", "#FF9800"),
    ]
    fig = go.Figure()
    palette = ["#2196F3","#9C27B0","#FF9800","#4CAF50","#F44336","#00BCD4","#CE93D8","#FF5722"]
    # Show all three as grouped bars
    for col, asc, label, _ in metrics_to_plot:
        if col not in _df.columns:
            continue
        vals = _df[col].fillna(0)
        fig.add_trace(go.Bar(
            x=list(_df.index), y=list(vals), name=col,
            hovertemplate=f"{col}: %{{y:.4f}}<extra></extra>",
        ))
    if "Directional Acc %" in _df.columns:
        fig.add_shape(type="line", x0=-0.5, x1=len(_df)-0.5, y0=50, y1=50,
                      line=dict(color="#FFB300", width=1.5, dash="dot"))
        fig.add_annotation(x=len(_df)-0.5, y=50, text="50% random", showarrow=False,
                           font=dict(color="#FFB300", size=10), xanchor="right", yanchor="bottom")
    fig.update_layout(barmode="group")
    return _apply_dark(fig, title=f"{ticker} — Model Comparison (RMSE · R² · Dir Acc)", height=500)

def _apply_dark(fig, title="", height=500):
    layout = dict(**_DARK, height=height)
    layout["title"] = dict(text=title, font=dict(color="#ffffff", size=13), x=0.0, xanchor="left")
    fig.update_layout(**layout)
    return fig

