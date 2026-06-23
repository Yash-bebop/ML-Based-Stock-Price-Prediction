def walk_forward_validation(df, feature_cols, train_months=24, test_months=3):
    df = df.copy(); df['YM'] = df.index.to_period('M')
    periods = df['YM'].unique(); results = []

    for i in range(train_months, len(periods) - test_months, test_months):
        tr = df[df['YM'].isin(periods[i - train_months:i])]
        te = df[df['YM'].isin(periods[i:i + test_months])]

        # Drop NaN targets — last row of each window has NaN from shift(-1)
        tr = tr.dropna(subset=['Target'])
        te = te.dropna(subset=['Target'])

        if len(te) < 5: continue

        sc  = StandardScaler()
        Xtr = sc.fit_transform(tr[feature_cols])
        Xte = sc.transform(te[feature_cols])

        m   = xgb.XGBRegressor(**best_xgb_params, verbosity=0).fit(Xtr, tr['Target'])
        row = evaluate_model(te['Target'].values, m.predict(Xte), f'W{i}')
        row['test_start'] = str(periods[i]); results.append(row)

        print(f"  W{i}: {periods[i]} RMSE={row['RMSE']:.6f} "
              f"DirAcc={row['Directional Acc %']:.1f}%")

    wf = pd.DataFrame(results)
    print(f"\nWF: {len(wf)} windows | "
          f"Mean RMSE={wf['RMSE'].mean():.6f} ± {wf['RMSE'].std():.6f} | "
          f"Mean DirAcc={wf['Directional Acc %'].mean():.1f}% "
          f"± {wf['Directional Acc %'].std():.1f}%")
    return wf

