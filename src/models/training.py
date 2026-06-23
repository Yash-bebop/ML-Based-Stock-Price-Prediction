def train_eval(name, model, Xtr, Xte, ytr, yte):
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
    print(f'Train: {len(X_train):,} | {train_dates[0].date()} → {train_dates[-1].date()}')
    print(f'Test : {len(X_test):,}  | {test_dates[0].date()} → {test_dates[-1].date()}')
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

