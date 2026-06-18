<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=Stock%20Price%20Prediction&fontSize=48&fontColor=ffffff&fontAlignY=38&desc=Full%20ML%20Pipeline%20%E2%80%A2%20v1%20%E2%86%92%20v2%20Iteration%20Story&descAlignY=60&descSize=16" width="100%"/>

<br/>

<!-- Badges row 1 — tech stack -->
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Optuna-2980B9?style=for-the-badge)
![FinBERT](https://img.shields.io/badge/FinBERT-NLP%20Sentiment-8E44AD?style=for-the-badge&logo=huggingface&logoColor=white)

<!-- Badges row 2 — capabilities -->
![SHAP](https://img.shields.io/badge/SHAP-Explainability-E74C3C?style=for-the-badge)
![Optuna](https://img.shields.io/badge/Optuna-AutoML-27AE60?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLM%20Agent-F39C12?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-Live%20UI-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Google Colab](https://img.shields.io/badge/Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

<!-- Badges row 3 — project meta -->
![Status](https://img.shields.io/badge/Status-Active-00C851?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![Cost](https://img.shields.io/badge/APIs-100%25%20Free-brightgreen?style=for-the-badge)
![Semester](https://img.shields.io/badge/Built%20in-2nd%20Sem%20B.Tech-red?style=for-the-badge)

<br/>

> **An end-to-end stock price prediction system** that grew from a clean ML baseline (v1) into a production-grade pipeline (v2) with agentic data collection, NLP sentiment scoring, automated hyperparameter search, SHAP explainability, walk-forward validation, an LLM narrative analyst, and a live Gradio app — all built on 100% free APIs.

<br/>

[![Open v1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/Stock_Price_Prediction_ML.ipynb)
&nbsp;&nbsp;
[![Open v2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/stock_prediction_pipeline_v2.ipynb)

</div>

---

## 📖 Table of Contents

- [The Iteration Story](#-the-iteration-story--v1--v2)
- [Architecture Overview](#-architecture-overview)
- [V1 — The Foundation](#-v1--the-foundation)
- [V2 — The Upgrade](#-v2--the-upgrade-21-step-pipeline)
- [Feature Engineering](#-feature-engineering-25-indicators)
- [Models & Results](#-models--results)
- [Walk-Forward Validation](#-walk-forward-validation)
- [SHAP Explainability](#-shap-explainability)
- [Backtest Engine](#-backtest-engine)
- [Groq LLM Agent](#-groq-llm-agent)
- [Live Gradio App](#-live-gradio-app)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Key Takeaways for Quants & Researchers](#-key-takeaways)

---

## 🚀 The Iteration Story — V1 → V2

> This project wasn't built in one shot. It started as a solid ML baseline and was systematically upgraded into a research-grade pipeline. Here's exactly what changed and why.

```
V1  ──────────────────────────────────────────────────────────  V2
│                                                                │
│  Yahoo Finance only          ──▶  Yahoo + SPY/QQQ/XLK/VIX    │
│  No sentiment                ──▶  FinBERT NLP (news+Reddit)   │
│  No macro data               ──▶  FRED API (VIX/CPI/Yield)   │
│  5 models, manual params     ──▶  7 models + Optuna 40-trial  │
│  Single 80/20 split + CV     ──▶  + Walk-forward validation   │
│  Feature importance only     ──▶  Full SHAP (waterfall+swarm) │
│  1-day forecast              ──▶  Multi-step 1/3/5/10 days    │
│  RMSE/MAE/R² only            ──▶  + p-values + 95% CI (boot)  │
│  Simple long/flat backtest   ──▶  Stop-loss + take-profit     │
│  No deployment               ──▶  Gradio app + Groq LLM agent │
│                                                                │
```

### Why these additions matter

| What was added | Why it matters |
|---|---|
| **Market context (SPY/QQQ/VIX)** | Individual stocks don't move in a vacuum. Sector and macro correlation are real alpha signals used by quant funds |
| **FinBERT sentiment** | NLP on financial text is a first-class signal in modern quant research. Raw price alone is never the full picture |
| **FRED macro features** | CPI, yield curve inversion, Fed funds rate — regime-aware models outperform regime-blind ones |
| **Optuna hyperparameter search** | Manual grid search is a ceiling. 40-trial Bayesian optimization finds regions no human would explore |
| **Walk-forward validation** | The only statistically honest way to validate a time-series model. Single-split validation leaks future regime information |
| **SHAP explainability** | Required in any regulated or institutional setting. Knowing *what* drives predictions is as important as the prediction itself |
| **Wilcoxon p-values** | Tests whether the model is statistically better than a naive baseline. Without this, good metrics could just be luck |
| **Bootstrap 95% CI on RMSE** | A single RMSE number is a point estimate. Confidence intervals communicate reliability — critical for research credibility |
| **Stop-loss / take-profit backtest** | Naive backtests overstate returns. Adding risk management makes the simulation realistic |
| **Groq LLM agent** | Bridges quant output and human narrative. Synthesizes model metrics + sentiment into an institutional-grade report |

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                            │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Yahoo Finance│  │  Reddit NLP  │  │  FRED Macro  │  │  Finviz    │ │
│  │ OHLCV + News │  │  PRAW + NLP  │  │  VIX/CPI/FFR │  │  Ratings   │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │
│         └─────────────────┴─────────────────┴────────────────┘        │
│                                    │                                    │
│                          ┌─────────▼──────────┐                        │
│                          │   AGENT 1 (Scraper) │                        │
│                          │  FinBERT Scorer      │                        │
│                          └─────────┬──────────┘                        │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│                       FEATURE ENGINEERING LAYER                         │
│                                                                         │
│   Trend: SMA20/50/200, EMA12/26       Momentum: RSI14, Stoch K/D       │
│   Volatility: BB, ATR14, Std20        Volume: OBV, Volume Ratio         │
│   MACD + Signal + Histogram           Lag Features: t-1,2,3,5,10       │
│   Calendar: DoW, Month, Quarter       Macro: VIX, CPI, Yield Curve     │
│   Sentiment: News Score, Reddit Score, Analyst Rating                  │
│                                                                         │
│   ◎  35+ total features after market context enrichment                 │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│                          MODEL TRAINING LAYER                           │
│                                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │Linear/   │ │ XGBoost  │ │LightGBM  │ │  Random  │ │    LSTM      │ │
│  │Ridge Reg │ │ Optuna   │ │  Optuna  │ │  Forest  │ │ 3-layer deep │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘ │
│       └────────────┴────────────┴──────────────┴─────────────┘         │
│                                    │                                    │
│                    ┌───────────────▼────────────────┐                  │
│                    │  ENSEMBLE (XGB×0.4 + LGB×0.4   │                  │
│                    │           + RF×0.2)             │                  │
│                    └───────────────┬────────────────┘                  │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│                         EVALUATION LAYER                                │
│                                                                         │
│   Walk-Forward Validation    SHAP TreeExplainer     Confusion Matrix    │
│   Wilcoxon p-values          Bootstrap 95% CI       Multi-step Forecast │
│   Sharpe Ratio               Max Drawdown           Win Rate            │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│                           OUTPUT LAYER                                  │
│                                                                         │
│   Groq LLM Narrative Report     Gradio Live Inference UI                │
│   Interactive Plotly Dashboard  Zipped Artifact Bundle                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📓 V1 — The Foundation

**File:** `Stock_Price_Prediction_ML.ipynb`

V1 was deliberately built to be clean, correct, and well-documented. The goal was to establish a solid baseline with no shortcuts on the fundamentals before adding complexity.

### What V1 does

```python
CONFIG = {
    "ticker"        : "AAPL",
    "start_date"    : "2018-01-01",
    "test_size"     : 0.2,          # Strict temporal holdout
    "look_back"     : 60,           # LSTM sequence length (days)
    "forecast_days" : 1,
    "random_state"  : 42,
    "n_folds"       : 5,            # TimeSeriesSplit CV folds
}
```

**Pipeline steps:**

| Step | What it does |
|------|-------------|
| `download_stock_data()` | Typed, documented Yahoo Finance downloader with error handling |
| `engineer_features()` | 25+ indicators with inline comments per indicator group |
| `split_data()` | Time-aware 80/20 split — NO shuffling, prevents look-ahead bias |
| `evaluate_model()` | RMSE, MAE, MAPE, R², directional accuracy |
| Training | Linear Reg → Ridge → XGBoost (5-fold CV) → Random Forest → LSTM |
| `backtest_strategy()` | Long/flat signal vs Buy & Hold with Sharpe + max drawdown |
| Visualizations | Plotly: predicted vs actual, candlestick + BB, RSI, MACD, equity curve |

**V1 saved artifacts:**
```
predictions.csv
xgboost_stock_model.json
random_forest_stock_model.pkl
lstm_stock_model.h5
predictions_chart.html
technical_analysis.html
feature_importance.html
equity_curve.html
correlation_heatmap.html
```

### V1 design decisions worth noting

- **Full docstrings with type hints** on every function — `def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame`
- **Scaler fit only on training data** — `scaler.fit_transform(X_train)`, `scaler.transform(X_test)`. A common beginner mistake is scaling the full dataset before splitting, which leaks test statistics into training.
- **`TimeSeriesSplit`** for cross-validation, not `KFold`. Standard KFold on time-series shuffles the temporal order and produces optimistically biased CV scores.
- **LSTM architecture:** 3-layer stacked (128 → 64 → 32) with `BatchNormalization` and `Dropout`, trained with `EarlyStopping` and `ReduceLROnPlateau`.

---

## 🔥 V2 — The Upgrade (21-Step Pipeline)

**File:** `stock_prediction_pipeline_v2.ipynb`

V2 was built by asking: *"What would make this usable in a real institutional or research context?"* Each addition was motivated by a specific gap in V1.

### The 21 steps

```
Step  1 │ Raw OHLCV download (Yahoo Finance, auto_adjust)
Step  2 │ Multi-ticker market context — SPY · QQQ · XLK · VIX
Step  3 │ Agent 1: Live data scraper
        │   ├── Yahoo Finance news (BeautifulSoup)
        │   ├── Reddit posts r/stocks · r/wallstreetbets · r/investing (PRAW)
        │   ├── FRED macro series — DFF · T10Y2Y · VIXCLS · CPIAUCSL
        │   └── Finviz analyst consensus rating
Step  4 │ Feature engineering — 35+ features (25 TA + market context + sentiment + macro)
Step  5 │ Time-aware train/test split (StandardScaler fit on train only)
Step  6 │ Evaluation framework — RMSE + CI + MAE + MAPE + R² + DirAcc + p-value
Step  7 │ Optuna hyperparameter tuning — XGBoost + LightGBM (40 trials each, minimize CV RMSE)
Step  8 │ Train all models — LR · Ridge · XGB · LGB · RF · Ensemble blend
Step  9 │ LSTM deep learning — 3-layer stacked, Huber loss, EarlyStopping
Step 10 │ Walk-forward validation — rolling 24-month train / 3-month test windows
Step 11 │ SHAP explainability — summary bar, beeswarm, waterfall for last prediction
Step 12 │ Directional confusion matrices — UP/DOWN precision & recall per model
Step 13 │ Multi-step forecasting — XGBoost trained separately for T+1/3/5/10
Step 14 │ Groq LLM agent — Llama3 generates quant narrative report (free tier)
Step 15 │ Model performance leaderboard with statistical significance flags
Step 16 │ Live inference — predict_tomorrow() with ensemble consensus
Step 17 │ Backtest with stop-loss (5%), take-profit (10%), transaction costs (0.1%)
Step 18 │ Full interactive Plotly dashboard — predictions + residuals + RSI
Step 19 │ Groq LLM report generation and display
Step 20 │ Gradio UI — live ticker input → ensemble prediction
Step 21 │ Save all artifacts to ZIP and download
```

---

## 📊 Feature Engineering — 25+ Indicators

All indicators computed using the `ta` library on raw OHLCV data. Features are grouped by signal type:

<details>
<summary><b>📈 Trend Indicators (5 features)</b></summary>

| Feature | Window | What it captures |
|---------|--------|-----------------|
| `SMA_20` | 20 days | Short-term price trend |
| `SMA_50` | 50 days | Medium-term trend |
| `SMA_200` | 200 days | Long-term institutional trend |
| `EMA_12` | 12 days | Faster exponential smoothing |
| `EMA_26` | 26 days | Slower EMA (MACD component) |

</details>

<details>
<summary><b>📉 MACD (3 features)</b></summary>

| Feature | Description |
|---------|-------------|
| `MACD` | 12-EMA minus 26-EMA |
| `MACD_Signal` | 9-day EMA of MACD |
| `MACD_Hist` | MACD minus Signal (momentum bar) |

</details>

<details>
<summary><b>💫 Momentum Oscillators (3 features)</b></summary>

| Feature | Range | Signal |
|---------|-------|--------|
| `RSI_14` | 0–100 | >70 overbought, <30 oversold |
| `Stoch_K` | 0–100 | Fast stochastic |
| `Stoch_D` | 0–100 | Smoothed stochastic |

</details>

<details>
<summary><b>📦 Bollinger Bands (5 features)</b></summary>

| Feature | Description |
|---------|-------------|
| `BB_Upper` | μ + 2σ (20-day) |
| `BB_Middle` | 20-day SMA |
| `BB_Lower` | μ - 2σ (20-day) |
| `BB_Width` | Band width — measures volatility expansion/compression |
| `BB_Pct` | %B — price position within the band (0–1) |

</details>

<details>
<summary><b>🌊 Volatility & Volume (6 features)</b></summary>

| Feature | Description |
|---------|-------------|
| `ATR_14` | Average True Range — absolute volatility |
| `Daily_Return` | Percentage change day-over-day |
| `Log_Return` | Log return — preferred for statistical modelling |
| `Rolling_Std_20` | 20-day realized volatility |
| `OBV` | On-Balance Volume — volume-price divergence |
| `Volume_Ratio` | Day volume / 20-day average volume |

</details>

<details>
<summary><b>🕐 Lag & Rolling Features (9 features)</b></summary>

| Feature | Description |
|---------|-------------|
| `Close_Lag_1/2/3/5/10` | Lagged closing prices — explicit autoregressive signal |
| `Rolling_Mean_5` | 5-day price average |
| `Rolling_Mean_10` | 10-day price average |
| `Price_vs_SMA20` | (Close − SMA20) / SMA20 — deviation from trend |
| `Price_vs_SMA50` | (Close − SMA50) / SMA50 |

</details>

<details>
<summary><b>🗓 Calendar & V2-Only Features</b></summary>

| Feature | Type | Source |
|---------|------|--------|
| `Day_of_Week` | Calendar | Day-of-week seasonality |
| `Month` | Calendar | Monthly seasonality |
| `Quarter` | Calendar | Earnings season effects |
| `SPY/QQQ/XLK_Return` | Market context | Broad market co-movement |
| `VIX_Return` | Market context | Fear index momentum |
| `News_Sentiment` | NLP (FinBERT) | Yahoo Finance news score |
| `Reddit_Sentiment` | NLP (FinBERT) | Social sentiment score |
| `Analyst_Score` | Scrape (Finviz) | Professional consensus rating |
| `Fed_Funds_Rate` | FRED macro | Monetary policy regime |
| `Yield_Curve` | FRED macro | T10Y2Y — recession indicator |
| `VIX_FRED` | FRED macro | Macro fear index |
| `CPI` | FRED macro | Inflation regime |

</details>

---

## 🤖 Models & Results

### Model Zoo

| Model | Type | Key config | V1 | V2 |
|-------|------|-----------|----|----|
| Linear Regression | Baseline | StandardScaler | ✅ | ✅ |
| Ridge Regression | Regularized linear | α=1.0 | ✅ | ✅ |
| XGBoost | Gradient boosting | **Optuna-tuned** in V2 | ✅ | ✅ |
| LightGBM | Gradient boosting | **Optuna-tuned**, new in V2 | ❌ | ✅ |
| Random Forest | Bagging ensemble | 300 trees, max_depth=15 | ✅ | ✅ |
| **Ensemble** | Weighted blend | XGB×0.4 + LGB×0.4 + RF×0.2 | ❌ | ✅ |
| LSTM | Deep learning | 3-layer stacked, Huber loss | ✅ | ✅ |

### Evaluation metrics — V2 framework

```python
def evaluate_model(y_true, y_pred, model_name):
    rmse    = np.sqrt(mean_squared_error(y_true, y_pred))
    mae     = mean_absolute_error(y_true, y_pred)
    mape    = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2      = r2_score(y_true, y_pred)
    dir_acc = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100

    # Statistical significance — is this better than a naive baseline?
    _, p_value = stats.wilcoxon(
        np.abs(y_true - y_pred),
        np.abs(np.diff(y_true, prepend=y_true[0])),
        alternative='less'
    )

    # Bootstrap 95% CI on RMSE
    rng = np.random.default_rng(42)
    boot_rmses = [
        np.sqrt(mean_squared_error(
            y_true[i := rng.integers(0, len(y_true), len(y_true))],
            y_pred[i]
        )) for _ in range(500)
    ]
    ci = np.percentile(boot_rmses, [2.5, 97.5])

    return {
        'RMSE': rmse, 'RMSE_CI_95': ci,
        'MAE': mae, 'MAPE (%)': mape, 'R2': r2,
        'Directional Acc %': dir_acc,
        'P-Value': p_value, 'Sig (p<0.05)': p_value < 0.05
    }
```

### Optuna hyperparameter search

```python
def xgb_objective(trial):
    params = {
        'n_estimators'     : trial.suggest_int('n_estimators', 200, 800),
        'max_depth'        : trial.suggest_int('max_depth', 3, 9),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
    }
    # Minimize TimeSeriesSplit CV RMSE across 5 folds
    return np.mean([
        np.sqrt(mean_squared_error(
            y_train[te],
            xgb.XGBRegressor(**params).fit(X_train[tr], y_train[tr]).predict(X_train[te])
        ))
        for tr, te in TimeSeriesSplit(n_splits=5).split(X_train)
    ])

study = optuna.create_study(direction='minimize')
study.optimize(xgb_objective, n_trials=40)  # Same for LightGBM
```

---

## 🔁 Walk-Forward Validation

> The most rigorous validation strategy for time-series models. Standard cross-validation assumes i.i.d. data — financial time-series is not i.i.d.

```
Timeline ──────────────────────────────────────────────────────▶

Window 1:  [████████████████████████ TRAIN ][ TEST ]
Window 2:       [████████████████████████ TRAIN ][ TEST ]
Window 3:            [████████████████████████ TRAIN ][ TEST ]
Window 4:                 [████████████████████████ TRAIN ][ TEST ]
...

Each window: 24-month rolling train / 3-month test
```

```python
def walk_forward_validation(df, feature_cols, train_months=24, test_months=3):
    for each rolling window:
        # Re-fit scaler on this window's training data only
        scaler = StandardScaler().fit(X_train_window)
        # Re-train XGBoost with best Optuna params on this window
        model = xgb.XGBRegressor(**best_xgb_params).fit(X_train_scaled, y_train)
        # Evaluate on the out-of-sample test window
        metrics = evaluate_model(y_test, model.predict(X_test_scaled))
```

**Why this matters:** A model might show great metrics on a single holdout split that happens to cover a bull market. Walk-forward shows how the model would have actually performed if re-trained and deployed every quarter — much closer to real trading operation.

---

## 🔍 SHAP Explainability

```python
explainer    = shap.TreeExplainer(xgb_model)
shap_values  = explainer.shap_values(X_test)

# Three views generated:
shap.summary_plot(shap_values, X_test, plot_type='bar')       # Global importance
shap.summary_plot(shap_values, X_test)                         # Beeswarm (impact direction)
shap.waterfall_plot(shap.Explanation(..., data=X_test[-1]))    # Last prediction breakdown
```

**What SHAP reveals that feature importance doesn't:**
- **Direction of impact** — does high RSI push the prediction up or down?
- **Interaction effects** — how does the model weight RSI differently when BB_Width is also high?
- **Per-prediction audit** — the waterfall plot shows exactly which features moved the last prediction and by how much
- **Regulatory compliance** — in any real institutional or algorithmic trading context, explainability is required

---

## 📈 Backtest Engine

### V1 backtest — simple long/flat

```python
# Signal: buy if predicted price > current price
portfolio['Signal'] = (portfolio['Predicted'] > portfolio['Actual'].shift(1)).astype(int)
portfolio['Strategy_Return'] = portfolio['Signal'].shift(1) * portfolio['Market_Return']
```

Metrics: Sharpe ratio, max drawdown, vs Buy & Hold benchmark.

### V2 backtest — realistic risk management

```python
def backtest_strategy(y_true, y_pred, dates,
                      cap=10_000.0,
                      tc=0.001,   # 0.1% transaction cost per trade
                      sl=0.05,    # 5% stop-loss
                      tp=0.10):   # 10% take-profit

    for each day:
        # Check if existing position hits stop-loss or take-profit
        if position_open and (return <= -sl or return >= tp):
            close_position(transaction_cost)

        # Enter long on bullish signal, exit on bearish
        if signal == LONG  and no_position: open_long(transaction_cost)
        if signal == SHORT and in_position: close_position(transaction_cost)
```

**What changed:** V1 assumed zero transaction cost and no position sizing rules. V2 models realistic friction. A strategy that looks profitable ignoring costs may not survive in production.

**Risk metrics computed:**
- Total return vs Buy & Hold
- Sharpe ratio (annualized, √252 scaling)
- Maximum drawdown
- Win rate across all closed trades

---

## 🧠 Groq LLM Agent

> V2 integrates a Groq-hosted Llama3 model as a narrative analyst that synthesizes all quantitative outputs into a structured investment report.

```python
def run_groq_agent(ticker, metrics_df, sentiment_report, tomorrow_pred, wf_summary):
    prompt = f"""
    Senior quant analyst reviewing ML stock prediction.

    Stock: {ticker}
    Best model: {best_model} (RMSE: {best_rmse:.4f}, DirAcc: {best_da:.1f}%)
    Statistically significant models: {sig_models}
    Walk-forward RMSE: {wf_mu:.4f} ± {wf_sd:.4f}
    Sentiment: news={news_score}, reddit={reddit_score}, analyst={analyst_score}
    Tomorrow signal: {signal} (ensemble Δ {pct_change:+.2f}%)

    Write a 5-point analysis covering: performance, sentiment, confidence,
    risks, and recommendation. Max 300 words, bullet format.
    """
    response = Groq(api_key=KEYS['groq']).chat.completions.create(
        model='llama3-8b-8192', messages=[{'role': 'user', 'content': prompt}]
    )
```

**The agent synthesizes:**
1. Which models are statistically reliable (p < 0.05)
2. Walk-forward stability (mean ± std RMSE)
3. Sentiment signal alignment with price prediction
4. Ensemble consensus signal + magnitude
5. Risk flags (disagreement between models, high VIX, negative sentiment)

**API cost: $0.00** — Groq's free tier handles this workload entirely.

---

## 🖥 Live Gradio App

```python
with gr.Blocks(theme=gr.themes.Soft(), title='Stock Predictor v2') as demo:
    gr.Markdown('# Stock Price Prediction — ML Pipeline v2.0')

    with gr.Row():
        ticker_input = gr.Textbox(label='Ticker', placeholder='AAPL, MSFT, TSLA...')
        predict_btn  = gr.Button('Predict', variant='primary')

    output_box = gr.Textbox(label='Ensemble Report', lines=12)
    predict_btn.click(fn=gradio_predict, inputs=ticker_input, outputs=output_box)

demo.launch(share=True)
```

**What the app does at inference time:**
1. Downloads last 400 days of OHLCV for the entered ticker
2. Adds market context (SPY/QQQ/XLK/VIX)
3. Runs full feature engineering pipeline
4. Gets predictions from XGBoost + LightGBM + Random Forest
5. Outputs ensemble prediction, % change, and BULLISH/BEARISH signal

---

## 🛠 Tech Stack

### Core ML & Data

| Library | Version | Role |
|---------|---------|------|
| `yfinance` | latest | OHLCV + news download |
| `ta` | latest | Technical indicator computation |
| `scikit-learn` | latest | Linear models, RF, preprocessing, CV |
| `xgboost` | latest | Gradient boosted trees |
| `lightgbm` | latest | Faster gradient boosting (V2) |
| `tensorflow` / `keras` | 2.x | LSTM deep learning |
| `optuna` | latest | Bayesian hyperparameter optimization (V2) |
| `shap` | latest | Model explainability (V2) |

### Data & NLP

| Library | Role |
|---------|------|
| `transformers` (HuggingFace) | FinBERT sentiment pipeline |
| `praw` | Reddit API client |
| `fredapi` | FRED macroeconomic data |
| `beautifulsoup4` | Yahoo Finance + Finviz scraping |
| `scipy` | Wilcoxon statistical test |

### Visualization & Deployment

| Library | Role |
|---------|------|
| `plotly` | Interactive charts + dashboard |
| `matplotlib` + `seaborn` | SHAP plots, confusion matrices |
| `gradio` | Live inference web UI |
| `groq` | Llama3 LLM API client |
| `joblib` | Model serialization |

---

## ⚡ Quick Start

### Prerequisites

```bash
# All free — no paid API required
GROQ_API_KEY     → https://console.groq.com          (free)
FRED_API_KEY     → https://fred.stlouisfed.org        (free)
REDDIT_CLIENT_ID → https://www.reddit.com/prefs/apps  (free)
```

### Run in Colab (recommended)

1. Click the **Open in Colab** badge at the top of this README
2. Go to `Runtime → Change runtime type → T4 GPU`
3. Add your API keys: `Tools → Secrets` (or leave blank to run without sentiment/macro)
4. `Runtime → Run all`

### Run locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

pip install yfinance ta xgboost lightgbm plotly scikit-learn tensorflow \
            seaborn transformers torch optuna shap praw fredapi \
            requests beautifulsoup4 groq gradio scipy

jupyter notebook stock_prediction_pipeline_v2.ipynb
```

### What runs without API keys

| Feature | No keys | With keys |
|---------|---------|-----------|
| Yahoo Finance OHLCV | ✅ | ✅ |
| Market context (SPY/VIX) | ✅ | ✅ |
| All ML models | ✅ | ✅ |
| Walk-forward, SHAP | ✅ | ✅ |
| Reddit sentiment | ❌ | ✅ |
| FRED macro features | ❌ | ✅ |
| Groq LLM report | ❌ (uses local fallback) | ✅ |

---

## 📁 Project Structure

```
📦 stock-price-prediction/
│
├── 📓 Stock_Price_Prediction_ML.ipynb     ← V1: clean baseline pipeline
├── 📓 stock_prediction_pipeline_v2.ipynb  ← V2: full production pipeline
│
├── 📊 outputs/ (generated at runtime)
│   ├── predictions.csv
│   ├── xgboost_stock_model.json
│   ├── lightgbm_stock_model.pkl
│   ├── random_forest_stock_model.pkl
│   ├── lstm_stock_model.h5
│   ├── predictions_chart.html
│   ├── technical_analysis.html
│   ├── full_prediction_dashboard.html
│   ├── feature_importance.html
│   ├── shap_summary.png
│   ├── shap_beeswarm.png
│   ├── walkforward_rmse.html
│   ├── multistep_forecast.html
│   ├── equity_curve.html
│   └── AAPL_ML_outputs.zip
│
└── 📄 README.md
```

---

## 💡 Key Takeaways

### For ML/AI Recruiters

- End-to-end pipeline ownership: data ingestion → feature engineering → model training → evaluation → deployment
- Demonstrates awareness of **time-series specific pitfalls**: look-ahead bias, temporal leakage, regime shifts
- NLP integration (FinBERT) and LLM agent (Groq/Llama3) show multi-modal ML fluency
- SHAP explainability shows understanding that **black-box predictions aren't enough** in high-stakes domains

### For Quant Professionals

- **Walk-forward validation** is the correct methodology — not just a single backtest window
- **Wilcoxon p-values** test whether outperformance is statistically significant vs a naive baseline
- **Bootstrap confidence intervals** on RMSE communicate model reliability — a point estimate RMSE is not sufficient
- **Transaction costs + stop-loss/take-profit** make the backtest realistic — naive backtests overstate returns
- Multi-step forecasting (T+1/3/5/10) reflects real trading operation, not just 1-day prediction

### For AI Researchers

- The **V1 → V2 iteration story** reflects proper empirical ML methodology: baseline first, then motivated additions
- Each V2 addition is **hypothesis-driven**: market context because stocks co-move, FinBERT because NLP signals are alpha, walk-forward because i.i.d. assumption fails on financial time-series
- **SHAP waterfall plots** provide the per-instance explanation needed for scientific audit of model decisions
- The ensemble blend (XGB×0.4 + LGB×0.4 + RF×0.2) is manually weighted — an obvious extension is **stacking with a meta-learner**

---

## 🗺 Roadmap / What comes next

- [ ] Meta-learner stacking (replace fixed ensemble weights with a trained blender)
- [ ] Hugging Face Spaces deployment (live public demo URL)
- [ ] MLflow experiment tracking (log all runs, compare hyperparameter trials)
- [ ] Transformer-based sequence model (replace LSTM with a time-series Transformer)
- [ ] Monte Carlo prediction intervals on the price forecast
- [ ] Options implied volatility as an additional feature

---

## ⚠️ Disclaimer

> This project is built for **educational and research purposes only**. Nothing here constitutes financial advice. Stock markets are inherently unpredictable and past model performance does not guarantee future returns. Do not trade real money based on this system.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer" width="100%"/>

**Built with curiosity in 2nd semester B.Tech CSE**

*If this helped you or impressed you — drop a ⭐ on the repo*

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=YOUR_USERNAME.stock-price-prediction)

</div>
