import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib

def evaluate_model(y_true, y_pred, name):
    # Dummy implementation so it runs if needed, should be imported from evaluation.metrics
    return {'RMSE': 0, 'Directional Acc %': 0, 'P-Value': 0, 'Sig (p<0.05)': ''}

def train_eval(name, model, Xtr, Xte, ytr, yte, predictions, metrics_list):
    model.fit(Xtr, ytr); pred=model.predict(Xte)
    predictions[name]=pred
    m=evaluate_model(yte,pred,name); metrics_list.append(m)
    print(f"  {name}: RMSE={m['RMSE']} DirAcc={m['Directional Acc %']}% p={m['P-Value']} {m['Sig (p<0.05)']}")
    return pred

def split_data(df, feature_cols, test_size):
    X=df[feature_cols].values; y=df['Target'].values
    idx=int(len(X)*(1-test_size))
    X_train,X_test=X[:idx],X[idx:]
    y_train,y_test=y[:idx],y[idx:]
    train_dates=df.index[:idx]; test_dates=df.index[idx:]
    scaler=StandardScaler()
    X_train_sc=scaler.fit_transform(X_train)
    X_test_sc=scaler.transform(X_test)
    print(f'Train: {len(X_train):,} | {train_dates[0].date()} -> {train_dates[-1].date()}')
    print(f'Test : {len(X_test):,}  | {test_dates[0].date()} -> {test_dates[-1].date()}')
    return X_train_sc, X_test_sc, y_train, y_test, scaler, test_dates

class PurgedTimeSeriesSplit:
    def __init__(self, n_splits=5, purge=0, embargo=0):
        self.n_splits, self.purge, self.embargo = n_splits, purge, embargo
    def split(self, X):
        n = len(X)
        fold_sizes = np.full(self.n_splits + 1, n // (self.n_splits + 1))
        fold_sizes[: n % (self.n_splits + 1)] += 1
        boundaries = np.cumsum(fold_sizes)
        for i in range(self.n_splits):
            val_start, val_end = boundaries[i], boundaries[i + 1]
            train_end = max(0, val_start - self.purge)
            train_idx = np.arange(0, train_end)
            val_idx   = np.arange(val_start, val_end)
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            yield train_idx, val_idx

def train_tree_models(X_train, X_test, y_train, y_test, best_xgb_params, best_lgb_params, CONFIG, predictions, metrics_list):
    print('🔵 Linear Regression...'); y_pred_lr    = train_eval('Linear Regression',LinearRegression(),X_train,X_test,y_train,y_test, predictions, metrics_list)
    print('🔵 Ridge Regression...');  y_pred_ridge = train_eval('Ridge Regression',Ridge(alpha=1.0),X_train,X_test,y_train,y_test, predictions, metrics_list)

    print('🟠 XGBoost...')
    xgb_model=xgb.XGBRegressor(**best_xgb_params,verbosity=0)
    xgb_model.fit(X_train,y_train,eval_set=[(X_test,y_test)],verbose=False)
    y_pred_xgb=xgb_model.predict(X_test)
    predictions['XGBoost']=y_pred_xgb
    m=evaluate_model(y_test,y_pred_xgb,'XGBoost'); metrics_list.append(m)
    print(f"  XGBoost: RMSE={m['RMSE']} DirAcc={m['Directional Acc %']}% {m['Sig (p<0.05)']}")
    xgb_model.save_model('xgboost_stock_model.json')

    print('🟡 LightGBM...')
    lgb_model=lgb.LGBMRegressor(**best_lgb_params)
    lgb_model.fit(X_train,y_train)
    y_pred_lgb=lgb_model.predict(X_test)
    predictions['LightGBM']=y_pred_lgb
    m=evaluate_model(y_test,y_pred_lgb,'LightGBM'); metrics_list.append(m)
    print(f"  LightGBM: RMSE={m['RMSE']} DirAcc={m['Directional Acc %']}% {m['Sig (p<0.05)']}")
    joblib.dump(lgb_model,'lightgbm_stock_model.pkl')

    print('🟢 Random Forest...')
    rf_model=RandomForestRegressor(n_estimators=300,max_depth=15,min_samples_split=5,
        min_samples_leaf=2,max_features='sqrt',random_state=CONFIG['random_state'],n_jobs=-1)
    y_pred_rf=train_eval('Random Forest',rf_model,X_train,X_test,y_train,y_test, predictions, metrics_list)
    joblib.dump(rf_model,'random_forest_stock_model.pkl')

    # Weighted ensemble
    _ew = CONFIG['ensemble_weights']
    y_pred_ensemble=y_pred_xgb*_ew['xgb']+y_pred_lgb*_ew['lgb']+y_pred_rf*_ew['rf']  # FIX M4: from CONFIG
    predictions['Ensemble']=y_pred_ensemble
    m=evaluate_model(y_test,y_pred_ensemble,'Ensemble'); metrics_list.append(m)
    print(f"  Ensemble: RMSE={m['RMSE']} DirAcc={m['Directional Acc %']}% {m['Sig (p<0.05)']}")
    print('✅ All tree models trained.')

    return xgb_model, lgb_model, rf_model, y_pred_ensemble
