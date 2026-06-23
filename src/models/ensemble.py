import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import optuna

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

def optimize_models(X_train, y_train, tscv, CONFIG):
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

    print('🔍 Optuna: XGBoost (40 trials)...')
    xs=optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=CONFIG['random_state']))
    xs.optimize(xgb_objective,n_trials=40,show_progress_bar=True)
    best_xgb_params={**xs.best_params,'random_state':42,'n_jobs':-1}
    print(f'✅ Best XGBoost RMSE: {xs.best_value:.6f}')
    print(f'   Params: {best_xgb_params}')

    print('\n🔍 Optuna: LightGBM (40 trials)...')
    ls=optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=CONFIG['random_state']))
    ls.optimize(lgb_objective,n_trials=40,show_progress_bar=True)
    best_lgb_params={**ls.best_params,'random_state':42,'n_jobs':-1,'verbose':-1}
    print(f'✅ Best LightGBM RMSE: {ls.best_value:.6f}')

    return best_xgb_params, best_lgb_params

def compute_ensemble_weights(X_train, y_train, _weight_cv, best_xgb_params, best_lgb_params, CONFIG):
    print("\n⚖️  STEP 5b: Data-Driven Ensemble Weights (OOF inverse-RMSE)")
    print("=" * 55)

    _rf_fixed_params = dict(n_estimators=300, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt', random_state=CONFIG['random_state'], n_jobs=-1)

    try:
        _oof_rmse_xgb = _oof_rmse(lambda: xgb.XGBRegressor(**best_xgb_params, verbosity=0),
                                   X_train, y_train, _weight_cv)
        _oof_rmse_lgb = _oof_rmse(lambda: lgb.LGBMRegressor(**best_lgb_params),
                                   X_train, y_train, _weight_cv)
        _oof_rmse_rf  = _oof_rmse(lambda: RandomForestRegressor(**_rf_fixed_params),
                                   X_train, y_train, _weight_cv)
        _oof = {'xgb': _oof_rmse_xgb, 'lgb': _oof_rmse_lgb, 'rf': _oof_rmse_rf}

        if all(v is not None and v > 0 for v in _oof.values()):
            _inv = {k: 1.0 / v for k, v in _oof.items()}
            _total = sum(_inv.values())
            data_driven_weights = {k: round(v / _total, 4) for k, v in _inv.items()}
            CONFIG['ensemble_weights'] = data_driven_weights
            print(f"  OOF RMSE   : xgb={_oof['xgb']:.6f}  lgb={_oof['lgb']:.6f}  rf={_oof['rf']:.6f}")
            print(f"  ✅ Data-driven weights : {CONFIG['ensemble_weights']}  (lower OOF error -> higher weight)")
        else:
            print("  ⚠️  OOF computation incomplete for one or more models — keeping CONFIG default weights "
                  f"{CONFIG['ensemble_weights']}")
    except Exception as e:
        print(f"  ⚠️  OOF weight fitting failed ({e}) — keeping CONFIG default weights {CONFIG['ensemble_weights']}")

    print(f"\n  Weights used downstream (Step 6 ensemble, Step 13 predict_tomorrow): {CONFIG['ensemble_weights']}")
    return CONFIG
