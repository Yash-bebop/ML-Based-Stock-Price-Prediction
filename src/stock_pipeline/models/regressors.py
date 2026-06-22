"""Traditional model training and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb


@dataclass
class TrainedRegressors:
    scaler: StandardScaler
    models: Dict[str, object]
    predictions: Dict[str, np.ndarray]
    ensemble_weights: Dict[str, float]


def time_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Chronological train/test split for financial time series."""

    split = int(len(X) * (1 - test_size))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def fit_regressors(X_train, y_train, X_test, random_state: int = 42) -> TrainedRegressors:
    """Fit the notebook's linear, tree, and boosted regressors."""

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "xgb": xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            objective="reg:squarederror",
        ),
        "lgb": lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
        ),
        "rf": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    predictions = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions[name] = np.asarray(model.predict(X_test_scaled))

    weights = inverse_rmse_weights({k: v for k, v in predictions.items() if k in {"xgb", "lgb", "rf"}}, y_test=None)
    return TrainedRegressors(scaler=scaler, models=models, predictions=predictions, ensemble_weights=weights)


def inverse_rmse_weights(predictions: Mapping[str, np.ndarray], y_test=None) -> Dict[str, float]:
    """Compute inverse-RMSE ensemble weights; fall back to notebook defaults without y."""

    if y_test is None:
        return {"xgb": 0.4, "lgb": 0.4, "rf": 0.2}
    rmses = {name: np.sqrt(mean_squared_error(y_test, pred)) for name, pred in predictions.items()}
    inv = {name: 1.0 / max(value, 1e-12) for name, value in rmses.items()}
    total = sum(inv.values()) or 1.0
    return {name: value / total for name, value in inv.items()}


def weighted_ensemble(predictions: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    """Combine XGBoost, LightGBM, and random forest predictions."""

    return sum(np.asarray(predictions[name]) * weights.get(name, 0.0) for name in weights)


def evaluate_predictions(y_true, predictions: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Build the notebook-style model benchmark table."""

    rows = []
    y_true = np.asarray(y_true)
    actual_direction = y_true > 0
    for name, pred in predictions.items():
        pred = np.asarray(pred)
        rows.append(
            {
                "Model": name,
                "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
                "MAE": float(mean_absolute_error(y_true, pred)),
                "R2": float(r2_score(y_true, pred)),
                "Directional Acc %": float(((pred > 0) == actual_direction).mean() * 100),
                "IC": float(pd.Series(pred).corr(pd.Series(y_true), method="spearman")),
            }
        )
    return pd.DataFrame(rows).set_index("Model")
