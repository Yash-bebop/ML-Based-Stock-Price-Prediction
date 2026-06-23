def _oof_rmse(model_fn, X, y, splitter):
    """Out-of-fold RMSE via the SAME purged/embargoed CV used for HPO above —
    never touches X_test/y_test."""
    oof_pred = np.full(len(y), np.nan)
    n_folds_used = 0
    for tr, va in splitter.split(X):
        m = model_fn()
        m.fit(X[tr], y[tr])
        oof_pred[va] = m.predict(X[va])
        n_folds_used += 1
    mask = ~np.isnan(oof_pred)
    if n_folds_used == 0 or mask.sum() < 10:
        return None
    return float(np.sqrt(mean_squared_error(y[mask], oof_pred[mask])))

def xgb_objective(trial):
    p={'n_estimators':trial.suggest_int('n_estimators',200,800),
       'max_depth':trial.suggest_int('max_depth',3,9),
       'learning_rate':trial.suggest_float('learning_rate',0.01,0.2,log=True),
       'subsample':trial.suggest_float('subsample',0.6,1.0),
       'colsample_bytree':trial.suggest_float('colsample_bytree',0.6,1.0),
       'reg_alpha':trial.suggest_float('reg_alpha',1e-4,5.0,log=True),
       'reg_lambda':trial.suggest_float('reg_lambda',1e-4,5.0,log=True),
       'random_state':42,'n_jobs':-1}
    return np.mean([np.sqrt(mean_squared_error(y_train[te],
        xgb.XGBRegressor(**p,verbosity=0).fit(X_train[tr],y_train[tr]).predict(X_train[te])))
        for tr,te in tscv.split(X_train)])

def lgb_objective(trial):
    p={'n_estimators':trial.suggest_int('n_estimators',200,800),
       'max_depth':trial.suggest_int('max_depth',3,9),
       'learning_rate':trial.suggest_float('learning_rate',0.01,0.2,log=True),
       'num_leaves':trial.suggest_int('num_leaves',20,100),
       'subsample':trial.suggest_float('subsample',0.6,1.0),
       'colsample_bytree':trial.suggest_float('colsample_bytree',0.6,1.0),
       'random_state':42,'n_jobs':-1,'verbose':-1}
    return np.mean([np.sqrt(mean_squared_error(y_train[te],
        lgb.LGBMRegressor(**p).fit(X_train[tr],y_train[tr]).predict(X_train[te])))
        for tr,te in tscv.split(X_train)])

