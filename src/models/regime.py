import numpy as np

def fit_hmm_regimes(df_features, test_size, random_state, y_pred_ensemble, y_test):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  ⚠️  hmmlearn not installed — run: !pip install hmmlearn -q")
        df_features['HMM_Regime'], df_features['HMM_Bull_Prob'] = 0, 0.5
        return df_features, False, None

    try:
        _hmm_split = int(len(df_features) * (1 - test_size))
        train_returns = df_features['Log_Return'].values[:_hmm_split]

        hmm_model = GaussianHMM(n_components=2, covariance_type='full',
                                 n_iter=200, random_state=random_state)
        hmm_model.fit(train_returns.reshape(-1, 1))

        state_means = hmm_model.means_.flatten()
        bull_state  = int(np.argmax(state_means))   # higher mean return = bull
        bear_state  = 1 - bull_state

        all_returns  = df_features['Log_Return'].values
        hmm_regimes  = hmm_model.predict(all_returns.reshape(-1, 1))
        bull_prob    = hmm_model.predict_proba(all_returns.reshape(-1, 1))[:, bull_state]
        df_features['HMM_Regime']    = hmm_regimes
        df_features['HMM_Bull_Prob'] = bull_prob

        test_regimes = hmm_regimes[_hmm_split:]
        bull_frac    = (test_regimes == bull_state).mean()
        print(f"  State means      : bull={state_means[bull_state]:+.5f}  bear={state_means[bear_state]:+.5f}")
        print(f"  Test-set regimes : {bull_frac:.1%} bull  |  {1-bull_frac:.1%} bear")

        # Regime-conditional directional accuracy of the ensemble (diagnostic only)
        if y_pred_ensemble is not None and y_test is not None:
            _bull_mask = (test_regimes == bull_state)
            _ens_dir   = (y_pred_ensemble > 0) == (y_test > 0)
            if _bull_mask.sum() > 0:
                print(f"  Ensemble DirAcc in BULL regime : {_ens_dir[_bull_mask].mean()*100:.1f}%  (n={_bull_mask.sum()})")
            if (~_bull_mask).sum() > 0:
                print(f"  Ensemble DirAcc in BEAR regime : {_ens_dir[~_bull_mask].mean()*100:.1f}%  (n={(~_bull_mask).sum()})")

        print("  ✅ HMM fitted — regime labels stored in df_features['HMM_Regime' / 'HMM_Bull_Prob']")
        return df_features, True, hmm_model

    except Exception as e:
        print(f"  ⚠️  HMM failed: {e}")
        df_features['HMM_Regime'], df_features['HMM_Bull_Prob'] = 0, 0.5
        return df_features, False, None

def fit_garch_volatility(df_features, test_size):
    try:
        from arch import arch_model
    except ImportError:
        print("  ⚠️  arch not installed — skipping GARCH volatility")
        df_features['GARCH_Vol'] = df_features['Rolling_Std_20']
        return df_features, False, None

    try:
        ret_pct    = df_features['Log_Return'].values * 100   # arch expects %-scale returns
        _gsplit    = int(len(ret_pct) * (1 - test_size))

        am        = arch_model(ret_pct[:_gsplit], mean='Zero', vol='Garch', p=1, q=1, dist='t')
        garch_res = am.fit(disp='off', show_warning=False)

        omega = garch_res.params['omega']
        alpha = garch_res.params['alpha[1]']
        beta  = garch_res.params['beta[1]']
        print(f"  GARCH(1,1) params : ω={omega:.6f}  α={alpha:.4f}  β={beta:.4f}")
        print(f"  Persistence (α+β) : {alpha+beta:.4f}  (closer to 1.0 = longer volatility memory)")

        sigma2 = np.full(len(ret_pct), np.nan)
        sigma2[0] = np.var(ret_pct[:_gsplit])
        for t in range(1, len(ret_pct)):
            sigma2[t] = omega + alpha * (ret_pct[t-1]**2) + beta * sigma2[t-1]

        df_features['GARCH_Vol'] = np.sqrt(sigma2) / 100.0  # convert back to fraction
        print("  ✅ GARCH fitted — conditional volatility stored in df_features['GARCH_Vol']")
        return df_features, True, garch_res
    except Exception as e:
        print(f"  ⚠️  GARCH failed: {e}")
        df_features['GARCH_Vol'] = df_features['Rolling_Std_20']
        return df_features, False, None

def calculate_avellaneda_stoikov(df_features, y_test, as_gamma, as_kappa):
    print("\n📈 STEP 6c: Avellaneda-Stoikov Market-Making Model")
    print("=" * 55)

    n_test = len(y_test)
    test_prices_as   = df_features['Close'].values[-n_test:]
    test_garch_pct   = df_features['GARCH_Vol'].values[-n_test:]
    test_bull_prob   = df_features['HMM_Bull_Prob'].values[-n_test:]

    q_inventory = 2 * test_bull_prob - 1
    sigma_price = (test_garch_pct / 100) * test_prices_as

    reservation_price = test_prices_as - q_inventory * as_gamma * (sigma_price**2) * 1.0
    half_spread = (as_gamma * (sigma_price**2) * 1.0 + (2.0 / as_gamma) * np.log(1.0 + as_gamma / as_kappa))
    optimal_bid  = reservation_price - half_spread
    optimal_ask  = reservation_price + half_spread
    full_spread  = 2 * half_spread
    
    mid_px = test_prices_as.mean()
    print(f"  Avellaneda-Stoikov summary (test period):")
    print(f"  Mean full spread  : ${full_spread.mean():.4f} ({full_spread.mean()/mid_px*10000:.1f} bps)")
    print(f"  Max full spread   : ${full_spread.max():.4f}")
    return reservation_price, optimal_bid, optimal_ask, full_spread, sigma_price
