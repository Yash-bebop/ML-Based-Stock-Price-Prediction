def evaluate_model(y_true, y_pred, name='Model'):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Remove NaN pairs (last row of each WF window has NaN Target)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {'Model': name, 'RMSE': np.nan, 'MAE': np.nan,
                'MAPE (%)': np.nan, 'R2': np.nan,
                'Directional Acc %': np.nan, 'P-Value': np.nan,
                'Sig (p<0.05)': '❌', 'RMSE_CI_95': '[nan,nan]'}

    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)

    # MAPE — guard against near-zero true values
    nonzero = np.abs(y_true) > 1e-8
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero])
                           / y_true[nonzero])) * 100 if nonzero.any() else np.nan

    # DirAcc — use boolean > 0, NOT np.sign()
    # np.sign(0.0) = 0.0, which equals neither +1 nor -1
    # causing 0% DirAcc whenever XGBoost predicts near-zero in calm markets
    dir_acc = np.mean((y_pred > 0) == (y_true > 0)) * 100

    # Bootstrap RMSE CI
    rng = np.random.default_rng(42)
    boot = [np.sqrt(mean_squared_error(
                y_true[idx := rng.integers(0, len(y_true), len(y_true))],
                y_pred[idx])) for _ in range(1000)]
    ci = f'[{np.percentile(boot,2.5):.6f},{np.percentile(boot,97.5):.6f}]'

    # Wilcoxon test vs naive baseline (predict zero return)
    naive  = np.zeros_like(y_true)
    errors_model = np.abs(y_true - y_pred)
    errors_naive = np.abs(y_true - naive)
    try:
        from scipy.stats import wilcoxon
        _, pval = wilcoxon(errors_model, errors_naive)
    except Exception:
        pval = 1.0

    # ── QUANT+ additions ────────────────────────────────────────────────
    # Diebold & Mariano (1995) 'Comparing Predictive Accuracy', Journal of
    # Business & Economic Statistics 13(3): tests whether the difference
    # in forecast-loss between two models is statistically significant,
    # rather than just comparing point RMSE. One-sided: H1 = model beats
    # naive. Wilcoxon (above) tests the same idea non-parametrically on
    # absolute errors; DM is the standard parametric companion quoted
    # alongside it in forecast-evaluation papers.
    from scipy.stats import norm as _norm
    loss_diff = errors_naive - errors_model          # positive = model better than naive
    dm_mean   = loss_diff.mean()
    dm_se     = loss_diff.std(ddof=1) / np.sqrt(max(len(loss_diff), 1)) + 1e-12
    dm_stat   = dm_mean / dm_se
    dm_pval   = float(1 - _norm.cdf(dm_stat))         # one-sided

    # Wilson score interval (Wilson 1927; see Brown, Cai & DasGupta 2001,
    # 'Interval Estimation for a Binomial Proportion', Statistical Science
    # 16(2), for why this beats the naive normal-approximation CI for a
    # proportion — directional accuracy IS a binomial proportion). Tighter
    # and better-calibrated than +-1.96*sqrt(p(1-p)/n) for n in the
    # hundreds, which is the regime a single test split usually falls in.
    n_dir  = len(y_true)
    p_hat  = dir_acc / 100
    z95    = 1.96
    denom  = 1 + z95**2 / n_dir
    centre = (p_hat + z95**2 / (2*n_dir)) / denom
    margin = z95 * np.sqrt(p_hat*(1-p_hat)/n_dir + z95**2/(4*n_dir**2)) / denom
    da_ci  = f'[{round(max(0,centre-margin)*100,1)},{round(min(1,centre+margin)*100,1)}]'

    # Information Coefficient: Spearman rank correlation of predicted vs
    # realised returns. Standard quant-PM diagnostic (Grinold & Kahn,
    # 'Active Portfolio Management', 2nd ed., 1999) — IC > 0.05 is
    # considered a meaningful signal at the individual-forecast level;
    # > 0.10 is strong for daily equity forecasts.
    from scipy.stats import spearmanr as _spearmanr
    _ic, _ = _spearmanr(y_pred, y_true)
    ic_val = round(float(_ic), 4) if not np.isnan(_ic) else np.nan

    return {
        'Model'             : name,
        'RMSE'              : round(rmse,  6),
        'RMSE_CI_95'        : ci,
        'MAE'               : round(mae,   6),
        'MAPE (%)'          : round(mape,  6) if not np.isnan(mape) else np.nan,
        'R2'                : round(r2,    4),
        'Directional Acc %' : round(dir_acc, 2),
        'DirAcc CI 95%'     : da_ci,                         # Wilson score interval
        'IC (Spearman)'     : ic_val,                        # Information coefficient
        'P-Value'           : round(pval,  4),               # Wilcoxon vs naive
        'DM-stat'           : round(float(dm_stat), 3),      # Diebold-Mariano
        'DM p-val'          : round(dm_pval, 4),              # one-sided, model<naive
        'Sig (p<0.05)'      : '✅ Yes' if pval < 0.05 else '❌ No',
    }

def plot_confusion_matrix(y_true, y_pred, model_name):
    # FIX: direction is now defined the SAME way as evaluate_model()'s Directional Acc %
    # (sign of the value itself), not np.diff() of the series. Those are different
    # questions — the old version made this table disagree with the Step 12 summary.
    true_dir=(np.asarray(y_true) > 0).astype(int)
    pred_dir=(np.asarray(y_pred) > 0).astype(int)
    cm=confusion_matrix(true_dir,pred_dir)
    fig,ax=plt.subplots(figsize=(5,4))
    # FIX: cmap (not colormap) is the correct kwarg for ConfusionMatrixDisplay.plot
    ConfusionMatrixDisplay(cm,display_labels=['DOWN','UP']).plot(ax=ax,cmap='Blues',values_format='d')
    ax.set_title(f'{model_name} — Direction Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"confusion_{model_name.replace(' ','_')}.png",dpi=120,bbox_inches='tight')
    plt.show()
    tn,fp,fn,tp=cm.ravel()
    print(f'  Precision: {tp/(tp+fp+1e-8):.2%} | Recall: {tp/(tp+fn+1e-8):.2%}')

