# Models

This folder contains the forecasting models that were previously trained in notebook cells.

## Files

| File | Purpose |
|---|---|
| `regressors.py` | Chronological splitting, scaling, linear/ridge/XGBoost/LightGBM/random-forest training, ensemble weighting, and benchmark metrics. |
| `neural.py` | LSTM sequence model and neural meta-learner/stacker builders. |

## Notebook Mapping

This folder maps to the notebook steps that trained the classical benchmark models, weighted ensemble, LSTM, and neural stacker.

## Design Notes

- Training remains chronological; no random row shuffling.
- The ensemble helper keeps the notebook's inverse-RMSE weighting idea separate from model fitting.
- Neural builders are isolated so TensorFlow-specific imports do not leak into unrelated modules.
