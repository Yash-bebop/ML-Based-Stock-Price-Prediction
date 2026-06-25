<div align="center">

<img src="assets/hero_v4.svg" alt="Agentic Quant ML Pipeline v4 hero" width="100%">

<br>

![Python](https://img.shields.io/badge/Python-Quant%20ML-00d4ff?style=for-the-badge&labelColor=071018)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20RMSE-b388ff?style=for-the-badge&labelColor=071018)
![Neural Stacker](https://img.shields.io/badge/Neural%20Stacker-Best%20Direction-61e294?style=for-the-badge&labelColor=071018)
![GARCH](https://img.shields.io/badge/GARCH-Volatility-ffcc66?style=for-the-badge&labelColor=071018)
![Gradio](https://img.shields.io/badge/Gradio-%20Quant%20Dashboard-ff6b6b?style=for-the-badge&labelColor=071018)

</div>

---

## Abstract

This repository presents an end-to-end machine learning research pipeline for equity return forecasting, applied to AAPL (Apple Inc.) and extensible to NASDAQ, NSE, and BSE-listed securities. The platform integrates seven distinct forecasting models, a trainable neural signal stacker, Hidden Markov Model (HMM) regime detection, GARCH(1,1) conditional volatility estimation, Avellaneda-Stoikov market-making spread modeling, rigorous walk-forward validation, residual diagnostics, portfolio risk analytics, natural language sentiment fusion, and a multi-agent reasoning layer implemented in LangGraph. An interactive Gradio dashboard surfaces all diagnostic outputs as a compact quantitative cockpit.

The central research question is not whether daily equity returns can be predicted with high precision -- the efficient market hypothesis (Fama, 1970) provides a strong prior that they cannot. The more tractable question is whether a well-validated ML pipeline can produce a statistically significant directional signal, maintain robust out-of-sample behavior across changing market regimes, and control portfolio risk more effectively than a passive benchmark. This project attempts to answer each of those sub-questions in turn through reproducible, evidence-grounded experimentation.

> **Disclaimer:** This is an educational research project. Nothing in this repository constitutes financial advice or a recommendation to trade any security.

---

## Runtime Snapshot

The following values are from the most recent execution (2026-06-25), with live price injection from Finnhub/yfinance at inference time.

| Field | Value |
|---|---:|
| Ticker | AAPL |
| Runtime date | 2026-06-25 |
| Last data date | 2026-06-23 |
| Test observations | 537 |
| Test window | 2024-05-01 to 2026-06-23 |
| Best RMSE model | XGBoost |
| Best RMSE | 0.017625 |
| Best directional model | Neural Stacker |
| Best directional accuracy | 52.92% |
| Live price | $293.08 |
| Ensemble forecast (T+1) | $292.10 |
| Sentiment-adjusted forecast | $292.14 |
| Percent change forecast | -0.336% |
| Signal | BEARISH |
| Prediction date | 2026-06-26 |

---

## System Architecture

<img src="assets/architecture_v4.svg" alt="v4 architecture" width="100%">

The notebook is organized as a ten-layer research system, progressing from raw data ingestion to agentic decision synthesis. Each layer is independently validated before its outputs are consumed downstream.

| Layer | Components |
|---|---|
| Data and sentiment | AAPL OHLCV (yfinance, 2015-present), SPY/QQQ/XLK/VIX market context, Finnhub live quotes, FRED macro series (Fed Funds Rate, CPI), news sentiment from Yahoo Finance and alternative feeds. |
| Feature engineering | 67-column feature frame spanning trend, momentum, volatility, volume, lag structure, macro indicators, calendar effects, and a live-injected price field at inference time. |
| Cross-validation framework | Purged and embargoed time-series CV (Lopez de Prado, 2018) with a 63-day purge gap matched to the longest active rolling momentum window and a 21-day post-fold embargo. |
| Forecast bench | Linear regression, ridge regression, XGBoost, LightGBM, random forest, data-driven inverse-RMSE weighted ensemble, and LSTM. |
| Neural stacker | A three-layer MLP trained on base model output signals. It becomes the best directional model in this run while trading off absolute magnitude accuracy. |
| Quant diagnostics | ADF stationarity testing, Ljung-Box residual autocorrelation, QQ/fat-tail residual analysis, and 34-window walk-forward performance drift. |
| Regime and volatility | Two-state HMM for bull/bear regime classification (Hamilton, 1989) and GARCH(1,1) for conditional volatility modeling (Bollerslev, 1986). |
| Market-making model | Avellaneda-Stoikov (2008) reservation price and half-spread estimation as a tradability and liquidity-cost lens. |
| Risk layer | Strategy versus buy-and-hold comparison across Sharpe, Sortino, Calmar, Omega, VaR, CVaR, maximum drawdown, and tail ratio. |
| Agentic reasoning | A five-agent LangGraph workflow (technical analyst, sentiment analyst, bull researcher, bear researcher, portfolio manager) that synthesizes all upstream metrics into a structured decision narrative. |

---

## Data Sources and Feature Engineering

### Data Inputs

The primary data source is the adjusted daily OHLCV series for AAPL retrieved via yfinance, covering 2015-01-01 to the execution date. Market context is provided by four auxiliary series: SPY, QQQ, XLK, and VIX, capturing broad market, technology sector, and volatility dynamics. Macroeconomic inputs from FRED include the effective Federal Funds Rate and CPI. Live price at inference time is injected from Finnhub (primary) or yfinance intraday (fallback), ensuring the T+1 prediction reflects the most recent observable market state rather than the prior close.

For Indian markets, NSE-listed tickers accept the `.NS` suffix and BSE-listed tickers accept the `.BO` suffix. The live price feed routes NSE tickers through the NSE India public API (no key required, real-time during 09:15-15:30 IST) before falling back to yfinance.

### Feature Construction

The 67-column feature frame is constructed from five broad categories.

**Technical indicators.** Trend features include SMA-20, SMA-50, SMA-200, EMA-12, EMA-26, MACD histogram, and normalized price deviation from the 20- and 50-day moving averages. Momentum features include RSI-14, Stochastic K/D, and multi-horizon momentum following Jegadeesh and Titman (1993) at windows of 5, 10, 21, and 63 trading days. Volatility features include Bollinger Band width, Bollinger %B, ATR-14, 20-day rolling standard deviation, and the Garman-Klass (1980) high-low-close volatility estimator, which is roughly 7-8 times more efficient than close-to-close standard deviation for a given sample size. Volume features include On-Balance Volume and the ratio of daily volume to its 20-day moving average.

**Lag structure.** Log-return lags at 1, 2, 3, 5, and 10 trading days provide the model with short-memory autocorrelation information, supported by additional rolling mean log-returns at 5, 10, and 20 trading days.

**Risk indicators.** Realized variance over 21 days and the Ulcer Index over 14 days capture drawdown intensity and downside risk concentration in the recent window.

**Calendar effects.** Day-of-week, month, and quarter are included as encoded integers to capture intraweek and seasonal seasonality, consistent with the day-of-week effect documented in French (1980).

**Target variable.** The prediction target is the one-day-ahead log return: Target = log(Close[t+1] / Close[t]). Using log returns rather than price levels removes the price scale issue, produces a more stationary series, and aligns with standard return modeling conventions in quantitative finance.

---

## Methodology

### Cross-Validation and Hyperparameter Optimization

Hyperparameter optimization (HPO) is performed using Optuna with a time-series-aware purged and embargoed cross-validation scheme following Lopez de Prado (2018, Chapter 7). A 63-day purge gap between training and validation folds is matched to the longest actively engineered rolling window (63-day momentum). A 21-day embargo follows each validation fold to prevent any overlapping return windows from leaking signal. It is explicitly noted that SMA-200 uses a longer window than the purge gap; removing 200 days of data from each CV fold would eliminate most of the usable validation history, so the 63-day purge is a deliberate documented trade-off rather than a claim of zero residual leakage. Final model fits use the full training set with the Optuna-selected hyperparameters.

### Ensemble Weighting

Fixed equal-weight or hand-tuned ensemble weights are replaced by a data-driven inverse-RMSE weighting scheme following the optimal combination principles in Bates and Granger (1969). Out-of-fold CV errors from XGBoost, LightGBM, and Random Forest are used to compute weights proportional to 1/RMSE, normalized to sum to 1. The resulting weights for this run are XGBoost 34.60%, LightGBM 33.07%, and Random Forest 32.33%, reflecting the marginal RMSE advantage of XGBoost and the near parity across tree-based models. A fixed fallback of (0.4, 0.4, 0.2) is retained as a safe default if the out-of-fold computation fails for any reason.

---

## Model Benchmark

<img src="assets/model_comparison.png" alt="Model comparison" width="100%">

Eight models are evaluated on 537 out-of-sample observations spanning 2024-05-01 to 2026-06-23. RMSE is reported on log-return scale. Directional accuracy, information coefficient (IC via Spearman rank correlation), and the Diebold-Mariano (DM) test statistic relative to a naive forecast are also reported. The 95% confidence intervals for both RMSE and directional accuracy are computed via bootstrap.

| Model | RMSE | MAE | R2 | Dir Acc % | Dir Acc CI (95%) | IC (Spearman) |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 0.018354 | 0.013236 | -0.0969 | 50.84 | [46.6, 55.0] | 0.0523 |
| Ridge Regression | 0.018333 | 0.013203 | -0.0944 | 51.02 | [46.8, 55.2] | 0.0539 |
| XGBoost | **0.017625** | 0.011883 | -0.0115 | 46.37 | [42.2, 50.6] | -0.0292 |
| LightGBM | 0.017704 | 0.012227 | -0.0206 | 49.53 | [45.3, 53.8] | 0.0645 |
| Random Forest | 0.018669 | 0.013302 | -0.1349 | 45.62 | [41.5, 49.9] | 0.0963 |
| Ensemble | 0.017795 | 0.012299 | -0.0311 | 46.55 | [42.4, 50.8] | 0.0661 |
| LSTM | 0.110633 | 0.088950 | -39.3616 | 50.66 | [46.4, 54.9] | 0.0101 |
| Neural Stacker | 0.019091 | 0.013226 | -0.2019 | **52.92** | [48.7, 57.1] | -0.0397 |

Several results are worth noting.

**1. No model achieves positive R2.** All R2 values are negative, confirming that predicting daily log-return variance is extremely difficult. This is consistent with the efficient market hypothesis and should not be interpreted as a pipeline failure; it reflects the nature of the target variable.

**2. XGBoost achieves the best RMSE (0.017625) but not the best directional accuracy (46.37%).** Gradient-boosted trees are strong magnitude forecasters but do not automatically produce the best trade signals.

**3. The Neural Stacker achieves the best directional accuracy (52.92%, CI: 48.7-57.1%).** The stacker improves direction at the cost of magnitude accuracy (RMSE 0.019091). This suggests it functions more usefully as a trade filter than as a raw price predictor. The CI lower bound of 48.7% does not exclude the 50% naive baseline, so this directional edge should be treated as an encouraging but tentative finding rather than a robust alpha signal.

**4. MAPE is reported in the artifacts but is not treated as a primary metric.** Because the prediction target is log return, which can be close to or equal to zero, MAPE values in the hundreds of percent are expected and uninformative. RMSE and directional accuracy are the primary evaluation criteria.

**5. LSTM substantially underperforms tree-based models (RMSE 0.110633, R2 -39.36).** The sequence model does not overcome the noise level of daily returns at this look-back window and data scale.

---

## Neural Stacker

The Neural Stacker is a three-layer MLP that takes the out-of-sample predictions from XGBoost, LightGBM, Random Forest, and the LSTM as input features and learns a meta-prediction. Unlike fixed-weight averaging, it can learn nonlinear combinations of base model signals and potentially recognize when certain models are more reliable in particular market conditions.

| Metric | Value |
|---|---:|
| RMSE | 0.019091 |
| MAE | 0.013226 |
| R2 | -0.2019 |
| Directional accuracy | 52.92% |
| Directional accuracy CI (95%) | [48.7, 57.1] |
| DM statistic | -5.335 |

The separation between the stacker's RMSE rank (worst among the non-LSTM models) and its directional rank (best overall) is the most interesting result in the benchmark. It implies that the meta-learned combination finds a signal in the alignment and divergence of base model forecasts that is orthogonal to raw magnitude accuracy. This is consistent with the ensemble learning literature: stacking can improve specific loss surfaces even when the underlying base learners are individually weak (Wolpert, 1992).

---

## LSTM Analysis

<img src="assets/lstm_training_curve.png" alt="LSTM training curve" width="100%">

The LSTM model uses a look-back window of 60 trading days (approximately three calendar months) and is trained with early stopping and learning rate reduction on plateau. The training curve reveals rapid overfitting.

| LSTM Diagnostic | Value |
|---|---:|
| Best validation loss epoch | 3 |
| Final training/validation loss gap | +0.020618 |
| Test RMSE | 0.110633 |
| Test directional accuracy | 50.66% |
| R2 | -39.3616 |

The test RMSE of 0.1106 is approximately 6.3 times larger than XGBoost's 0.01763. The model is included as a research baseline rather than a production component. The result is consistent with the broader finding in Gu, Kelly, and Xiu (2020) that tree-based models tend to outperform deep learning methods on cross-sectional and time-series equity prediction tasks, at least without very large datasets and extensive hyperparameter tuning. One candidate explanation is that daily equity return sequences at the single-asset level do not contain enough persistent temporal structure for an LSTM to extract meaningful sequential features before the noise dominates.

---

## Regime Detection and Volatility Modeling

<img src="assets/hmm_regime_garch.png" alt="HMM regime and GARCH volatility" width="100%">

### Hidden Markov Model

A two-state HMM (Hamilton, 1989) is fitted to the log-return series using the Baum-Welch algorithm (via hmmlearn). The two states are interpreted post-hoc as bull and bear regimes based on their estimated mean returns.

| Regime Metric | Value |
|---|---:|
| Bull state mean daily return | +0.00157 |
| Bear state mean daily return | -0.00098 |
| Test-set bull regime share | 89.6% |
| Test-set bear regime share | 10.4% |

The dominant bull-regime fraction (89.6%) over the 2024-2026 test window reflects the broadly positive AAPL price trajectory over that period. Regime labels can in principle be used as a conditioning variable for position sizing, for example, reducing or eliminating long exposure when the HMM assigns high probability to the bear state. This integration is not implemented in the current backtest but is noted as a natural extension.

### GARCH(1,1) Volatility

A GARCH(1,1) model (Bollerslev, 1986) is estimated on log returns using maximum likelihood via the arch library.

| Volatility Metric | Value |
|---|---:|
| GARCH persistence (alpha + beta) | 0.9860 |
| Average annualized conditional volatility | 27.1% |
| Conditional volatility range | 15.4% to 94.1% |

The persistence parameter near 1.0 confirms well-known stylized facts of equity volatility: shocks are long-lived and volatility clusters. The range from 15.4% to 94.1% annualized reflects the significant realized volatility spikes in the 2024-2026 sample (visible in the accompanying chart). The GARCH estimate provides a time-varying volatility input to the Avellaneda-Stoikov spread model and a contextual signal for the portfolio risk layer.

---

## Stationarity and Residual Diagnostics

<img src="assets/stationarity_diagnostics.png" alt="Stationarity diagnostics" width="100%">

<img src="assets/residual_distribution.png" alt="Residual distribution" width="100%">

Before modeling, the return series is verified to be stationary using the Augmented Dickey-Fuller (ADF) test. Residual diagnostics are applied to the ensemble model's out-of-sample prediction errors.

| Diagnostic | Result |
|---|---:|
| ADF statistic (log returns) | -16.8033 |
| ADF p-value | < 0.0001 |
| Ljung-Box p-value (lag 5) | 0.0350 |
| Ljung-Box p-value (lag 10) | 0.1206 |
| Residual mean | 0.003600 |
| Residual standard deviation | 0.017482 |
| Residual skew | 0.142 |
| Residual excess kurtosis | 8.618 |
| QQ R2 (normality fit) | 0.9026 |

The ADF result (p < 0.0001) provides strong evidence that log returns are stationary, satisfying a prerequisite for the linear model assumptions. The Ljung-Box test at lag 5 shows marginal autocorrelation in residuals (p = 0.035), which disappears at lag 10 (p = 0.121). Economically, this suggests a small but observable short-term predictability structure in the errors that the current feature set does not fully capture.

More substantially, the excess kurtosis of 8.618 and the QQ plot deviation at the tails (visible in the residual chart) confirm fat-tailed error distributions. This invalidates standard normal-error assumptions for confidence intervals and position sizing. Any risk management application downstream should use fat-tailed distributions (Student's t, Cornish-Fisher adjustment, or empirical quantiles) rather than Gaussian VaR approximations. The positive residual mean of 0.0036 indicates a slight systematic upward bias in prediction errors, consistent with the broadly bullish trend in the test window.

---

## Walk-Forward Validation

<img src="assets/walkforward_drift.png" alt="Walk-forward drift" width="100%">

Walk-forward validation asks whether model performance is stable across rolling time windows rather than relying on a single train-test split. The expanding-window scheme runs 34 validation windows from October 2017 through January 2026, each with a minimum window size of 24 months.

| Walk-Forward Metric | Value |
|---|---:|
| Number of windows | 34 |
| Mean RMSE | 0.017914 |
| RMSE standard deviation | 0.006855 |
| Mean directional accuracy | 51.1% |
| Directional accuracy standard deviation | 7.6% |
| Directional accuracy range | 35.5% to 65.6% |
| Directional accuracy trend slope | -0.14% per window |

The RMSE is relatively stable across windows (coefficient of variation: 0.38), suggesting that gradient-boosted tree models maintain consistent magnitude forecasting behavior across market regimes. The directional accuracy is much more volatile (standard deviation 7.6%), with windows ranging from 35.5% to 65.6%. This variance reflects genuine regime sensitivity: the model's directional signal strengthens in trending markets and weakens in choppy or reversal-dominated regimes. The mild negative slope (-0.14% per window) in directional accuracy does not indicate severe decay but warrants monitoring in longer deployments. Consistent with the Diebold-Mariano statistics in the benchmark table, the directional signal is statistically significant in aggregate but unstable across individual windows.

---

## Multi-Step Forecasting

Interactive chart: [`assets/multistep_forecast.html`](assets/multistep_forecast.html)

The ensemble model is evaluated at four forecast horizons by fitting separate models for each target (T+1, T+3, T+5, T+10), following the direct multi-step strategy rather than iterative single-step propagation.

| Horizon | RMSE (log-return) | RMSE (price, USD) |
|---:|---:|---:|
| T+1 | 0.017668 | $3.94 |
| T+3 | 0.033279 | $7.45 |
| T+5 | 0.043337 | $9.78 |
| T+10 | 0.060362 | $13.87 |

RMSE grows monotonically with horizon, as expected under the efficient markets framework: longer forecast horizons accumulate more uncertainty. The T+1 RMSE of $3.94 on a $293 stock (approximately 1.3%) indicates that even the best-performing horizon carries substantial absolute error. The T+10 RMSE of $13.87 (approximately 4.7%) reinforces that the platform is most useful as a short-horizon research tool rather than a long-range price oracle. The interactive HTML chart allows visual inspection of forecast confidence bands across all four horizons.

---

## Feature Attribution (SHAP)

<img src="assets/shap_summary.png" alt="SHAP global importance" width="70%">

<img src="assets/shap_beeswarm.png" alt="SHAP beeswarm" width="100%">

SHAP (Shapley Additive Explanations) TreeExplainer values are computed for the XGBoost model to produce model-agnostic attribution at both the global and per-observation level (Lundberg and Lee, 2017). Two complementary views are provided.

| SHAP View | Purpose |
|---|---|
| Bar summary (shap_summary.png) | Global feature importance ranked by mean absolute SHAP value, showing the average magnitude of each feature's contribution across all test observations. |
| Beeswarm (shap_beeswarm.png) | Per-observation SHAP values colored by feature value (high/low), showing the direction and heterogeneity of each feature's influence across the test set. |

The top features by global importance are On-Balance Volume (OBV), Volume SMA-20, EMA-12, Garman-Klass Volatility (GK_Vol), and Return_Lag_3. The OBV dominance in magnitude forecasting is consistent with the literature showing that volume-price relationship carries short-term return predictability (Granville, 1963; Lo and Wang, 2000). The beeswarm plot shows that high values of GK_Vol and Rolling_Std_20 tend to push predictions in both directions (wide spread of SHAP values), indicating that the XGBoost model uses volatility features to identify periods of amplified expected moves rather than consistent directional bets. Momentum features (Mom_21d, Mom_63d) show opposing SHAP directions depending on whether feature values are high or low, consistent with the intermediate-horizon momentum reversal patterns documented for individual large-cap stocks.

---

## Backtest and Risk Analysis

Interactive backtest: [`assets/backtest_Ensemble.html`](assets/backtest_Ensemble.html)

<img src="assets/quant_risk_charts.png" alt="Quant risk charts" width="100%">

The backtest applies the ensemble model's directional signal to a long/flat strategy with an initial capital of $10,000 and a transaction cost of 0.1% per trade. The strategy holds a long position when the ensemble predicts a positive next-day return and moves to cash otherwise. Performance is compared against a passive buy-and-hold position in AAPL over the same period.

| Metric | Ensemble Strategy | Buy and Hold |
|---|---:|---:|
| Annualized return % | 33.17 | 34.14 |
| Annualized volatility % | 25.84 | 28.02 |
| Sharpe ratio | 1.284 | 1.219 |
| Sortino ratio | 1.017 | 1.230 |
| Calmar ratio | 3.573 | 1.023 |
| Omega ratio | 2.884 | 1.221 |
| Profit factor | 2.884 | 1.221 |
| VaR 95% (1-day, %) | 0.00 | -2.672 |
| CVaR 95% (1-day, %) | -0.064 | -3.918 |
| VaR 99% (1-day, %) | -2.064 | -4.620 |
| CVaR 99% (1-day, %) | -3.314 | -6.030 |
| Maximum drawdown % | -9.28 | -33.36 |
| Return skewness | 15.945 | 0.857 |

The most notable result is the substantial improvement in downside risk metrics. The strategy reduces maximum drawdown from -33.36% to -9.28% (a reduction of 72%), and improves the Calmar ratio from 1.023 to 3.573. The 95% VaR of approximately 0% reflects the strategy's cash-holding periods, which largely sidestep the worst daily drawdowns that a buy-and-hold investor would absorb. The Sharpe ratio of 1.284 marginally exceeds buy-and-hold (1.219) despite the annualized return being slightly lower (33.17% versus 34.14%), indicating that the risk-adjusted return improvement comes primarily from volatility reduction rather than alpha generation.

The extreme skewness (15.945) and excess kurtosis (314.766) in the strategy return distribution, and the near-infinite tail ratio, reflect the heavily right-skewed daily return profile when most cash-holding days record near-zero returns while positive trend days are captured. These statistics are not economically meaningful at face value and should be interpreted with caution. The Sortino ratio of 1.017 (lower than buy-and-hold's 1.230) suggests that the strategy does not improve downside semi-deviation relative to upside, which is a more conservative reading of the risk-adjusted performance.

**Summary interpretation.** The strategy does not produce alpha over buy-and-hold on an absolute or Sortino-adjusted basis. Its primary value is as a drawdown-control mechanism: an investor who cannot tolerate a -33% drawdown might accept the marginally lower annualized return in exchange for the -9% drawdown profile.

---

## Avellaneda-Stoikov Market-Making Model

Interactive chart: [`assets/avellaneda_stoikov_spread.html`](assets/avellaneda_stoikov_spread.html)

The Avellaneda-Stoikov (2008) model is included not as a trading strategy but as a liquidity and uncertainty lens. The model estimates the reservation price (the price at which a market maker would be indifferent between bidding and offering) and the optimal bid-ask half-spread as functions of inventory risk, arrival intensity of market orders, and remaining time horizon.

| Parameter / Metric | Value |
|---|---:|
| Risk aversion parameter (gamma) | 0.1 |
| Order arrival intensity (kappa) | 1.5 |
| Mean half-spread | $3.0193 |
| Mean full spread | $6.0385 |
| Mean full spread (percent) | 2.5589% |
| Mean reservation offset from mid | -$0.6862 |
| Spread range | $3.53 to $28.32 |

The mean full spread of approximately 2.56% of the stock price provides a practical tradability threshold. If the ensemble model's T+1 predicted move is smaller than the estimated spread, the forecast may not survive transaction costs and adverse selection in practice. At the current live price of $293.08, the mean full spread of $6.04 implies a required minimum price move of approximately 2.06% for a round-trip trade to be profitable before any other costs. This is substantially larger than the typical daily absolute log return, which reinforces that the model is better suited for identifying directional regime context than for high-frequency execution.

---

## Sentiment Integration

| Sentiment Field | Value |
|---|---:|
| News sentiment (Yahoo Finance, FinBERT) | 0.1221 |
| Alternative news sentiment | 0.0724 |
| Combined sentiment score | 0.0973 |
| Analyst consensus score | 0.0000 |
| Yahoo Finance articles (7-day) | 20 |
| Alternative news articles (7-day) | 50 |

Sentiment is sourced from two pipelines: a Yahoo Finance RSS feed scored via FinBERT (a financial domain BERT variant), and an alternative financial news feed. Scores are normalized to the range [-1, 1]. The combined sentiment of +0.097 indicates mildly positive news tone, while the neutral analyst score reflects the absence of fresh consensus upgrades or downgrades. Despite the modestly positive sentiment, the ensemble forecast remains bearish (-0.336%), illustrating that the sentiment tilt is a small perturbation to the ML signal rather than a driver of it.

The sentiment adjustment is applied only at live inference time as a capped tilt of at most 0.15% (Tetlock, 2007), and is always presented as a separate number alongside the pure ML ensemble forecast rather than silently merged into it. Historical sentiment values are not used as training features, because filling historical sentiment fields with fabricated zeros would introduce a systematic bias in the model's learned signal-to-sentiment relationship.

---

## Agentic Reasoning Layer

The LangGraph workflow implements a structured multi-agent debate over the quantitative outputs, using llama-3.1-70b-versatile via the Groq inference API. Each agent operates from a defined evidence role and produces a written assessment that feeds into the portfolio manager's final synthesis.

| Agent | Evidence Domain | Role |
|---|---|---|
| Technical analyst | Price forecast, SHAP feature drivers, trend indicators | Assesses model signal strength and technical market structure. |
| Sentiment analyst | FinBERT scores, analyst consensus, macro context | Interprets sentiment divergence and news flow relevance. |
| Bull researcher | All upstream outputs | Constructs the strongest available upside case. |
| Bear researcher | All upstream outputs | Constructs the strongest available downside case. |
| Portfolio manager | Bull/bear debate output, risk metrics | Produces a final position recommendation with explicit uncertainty acknowledgment. |

The agentic layer is not a decision engine. Its primary research value is to force structured reasoning over conflicting evidence. When sentiment is mildly bullish and the ML forecast is bearish, the portfolio manager agent must reconcile that conflict rather than averaging it away. The quality of the output depends on the quality of the quantitative inputs, so the agent layer is always placed downstream of validated diagnostics rather than as a substitute for them.

The five-point analysis from the most recent run (see `analysis_report.txt`) concludes with a "hold or reduce" recommendation, citing the ensemble's bearish read, a moderate confidence level from stable walk-forward RMSE, and key tail risks from the residual fat-tail diagnostics and the model's directional accuracy uncertainty interval.

---

## Gradio Dashboard

The Gradio v7 interface exposes the full research output as an interactive, tab-based cockpit. It is designed to be runnable directly in Google Colab without additional infrastructure.

| Dashboard Tab | Asset | Content |
|---|---|---|
| Live ticker | Finnhub/yfinance | Real-time price card for any US, NSE, or BSE ticker with 30-second polling. |
| Forecast output | tomorrow_prediction.json | Per-model T+1 forecasts, ensemble value, sentiment-adjusted value, and signal badge. |
| Model comparison | model_comparison.png | RMSE, directional accuracy, and R2 comparison across all eight models. |
| Walk-forward drift | walkforward_drift.png | Rolling RMSE and directional accuracy across 34 windows. |
| HMM and GARCH | hmm_regime_garch.png | Bull/bear regime probability overlay with conditional volatility bands. |
| Residual diagnostics | residual_distribution.png | Residual histogram, QQ plot, and bias-over-time chart. |
| Multi-step forecast | multistep_forecast.html | Interactive T+1 through T+10 forecast horizon view. |
| Backtest | backtest_Ensemble.html | Interactive equity curve and trade entry/exit visualization. |
| Quant risk | quant_risk_charts.png | Drawdown comparison, VaR/CVaR zones, and ensemble model weights. |
| Avellaneda-Stoikov spread | avellaneda_stoikov_spread.html | Interactive spread and reservation price chart. |
| SHAP | shap_summary.png, shap_beeswarm.png | Bar importance and per-observation SHAP attribution. |
| LSTM training | lstm_training_curve.png | Training and validation loss curves by epoch. |
| NSE watchlist | Live via NSE India API | Multi-ticker Indian equity monitor with real-time updates. |

---

## Principal Findings

| Finding | Interpretation |
|---|---|
| XGBoost is the best magnitude forecaster (RMSE 0.017625). | Gradient-boosted trees consistently outperform both neural approaches and linear models on log-return RMSE for this data scale and horizon. |
| Neural Stacker achieves the best directional accuracy (52.92%). | Signal stacking over base model outputs is a more effective mechanism for improving directional forecasts than improving any single base model. The confidence interval [48.7, 57.1] means the result is encouraging but not conclusively above the 50% naive baseline. |
| No model achieves positive out-of-sample R2. | Consistent with efficient markets; the pipeline's value is in relative noise control and directional signal extraction, not variance explanation. |
| Walk-forward RMSE is stable (CV 0.38); directional accuracy is not (CV 0.15). | Magnitude forecasting is more regime-robust than directional forecasting. Regime-conditioned position sizing is a natural extension to address directional volatility. |
| GARCH persistence of 0.986 confirms volatility clustering. | Risk models should treat volatility as time-varying, not constant. Position sizing that ignores realized volatility will over-size during high-volatility regimes. |
| Ensemble strategy reduces max drawdown by 72% (from -33.36% to -9.28%) at the cost of marginally lower absolute return. | The primary practical value of the signal is drawdown control, not alpha generation, at the current directional accuracy level. |
| Avellaneda-Stoikov mean full spread (2.56%) exceeds typical daily absolute returns. | Most individual forecast days are not tradeable after realistic transaction costs. The pipeline is more useful as a regime signal or multi-day filter than as a daily execution trigger. |
| LSTM substantially underperforms (RMSE 0.1106) and overfits at epoch 3. | For single-asset daily return prediction at this data scale, recurrent sequence modeling does not outperform tree-based methods without architectural changes, larger datasets, or multi-asset input structures. |

---

## Limitations

The following limitations should be considered when interpreting results.

**1. Single-asset scope.** All quantitative results are specific to AAPL in the 2024-2026 test window. Performance on other tickers, sectors, or market regimes has not been evaluated systematically. Generalizing the findings beyond this context requires additional experiments.

**2. Directional accuracy uncertainty.** The Neural Stacker's 52.92% directional accuracy has a 95% confidence interval of [48.7, 57.1]. The lower bound does not exclude random performance. This signal should be treated as preliminary rather than as a reliable trading edge without further validation on independent data.

**3. MAPE unreliability.** MAPE is reported in the benchmark artifact for completeness but should not be used as a primary evaluation metric. Because the target is log return, which frequently passes through zero, MAPE values in the hundreds of percent are expected artifacts rather than informative accuracy statistics.

**4. Transaction cost simplification.** The backtest uses a fixed 0.1% transaction cost. Real execution costs, market impact, bid-ask spread, and slippage are not modeled. The Avellaneda-Stoikov spread analysis (Section 10) suggests that true round-trip costs are likely higher than the fixed fee assumption.

**5. LSTM baseline.** The LSTM is implemented as a research baseline, not an optimized deep learning model. Its underperformance should not be interpreted as evidence that deep learning cannot work for equity prediction; it is evidence that the current architecture and hyperparameter space are insufficient for this task.

**6. Look-ahead bias in SMA-200.** The 200-day moving average feature introduces a longer historical dependency than the 63-day CV purge gap. This is explicitly documented in the notebook as a deliberate trade-off to preserve usable CV data. It does not invalidate out-of-sample evaluation on the held-out test set, but users extending the CV analysis should be aware of this asymmetry.

**7. Agentic output quality.** The LangGraph agents produce fluent, contextually structured narratives, but the quality of their reasoning is bounded by the quality of their input metrics. The agents do not have access to information beyond what is passed programmatically; they should not be treated as independent validators.

**8. Interactive chart embedding.** The `.html` interactive charts render fully in local and Colab environments. GitHub's static README preview will show them as hyperlinks rather than embedded visualizations.

---

## Reproducibility

All artifacts from the most recent run are included in the repository.

| File | Contents |
|---|---|
| `predictions.csv` | 537 rows of out-of-sample actual versus predicted log returns for all five tabular models. |
| `metrics.json` | Full model benchmark: RMSE with bootstrap CI, MAE, MAPE, R2, directional accuracy with CI, Spearman IC, p-value, and DM test statistics. |
| `walkforward_results.csv` | Per-window metrics across all 34 walk-forward validation windows, including test start date for temporal alignment. |
| `tomorrow_prediction.json` | Live forecast payload: per-model T+1 predictions, sentiment score, sentiment-adjusted ensemble, percent change, and signal. |
| `sentiment_report.json` | News sentiment snapshot: FinBERT score, alternative news score, combined score, analyst consensus, and article counts. |
| `risk_metrics.json` | Strategy versus buy-and-hold risk comparison: full set of return, volatility, drawdown, and tail metrics. |
| `analysis_report.txt` | LLM-generated five-point summary of the run produced by the Groq-backed report agent. |

The notebook is self-contained and runnable in Google Colab. All required libraries are installed in the first cell. The four optional API keys (Groq, FRED, AlphaVantage, Finnhub) extend functionality but are not required for core pipeline execution. Without them, the pipeline uses yfinance for all market data, skips macro FRED features, and generates a static analysis report instead of an LLM-generated one.

---

## Research References

1. Eugene F. Fama, [Efficient Capital Markets: A Review of Theory and Empirical Work](https://www.jstor.org/stable/2325486), Journal of Finance, 1970.
2. Andrew W. Lo and A. Craig MacKinlay, [Stock Market Prices Do Not Follow Random Walks](https://academic.oup.com/rfs/article-abstract/1/1/41/1601244), Review of Financial Studies, 1988.
3. Shihao Gu, Bryan Kelly, and Dacheng Xiu, [Empirical Asset Pricing via Machine Learning](https://academic.oup.com/rfs/article/33/5/2223/5758276), Review of Financial Studies, 2020.
4. Tianqi Chen and Carlos Guestrin, [XGBoost: A Scalable Tree Boosting System](https://dl.acm.org/doi/10.1145/2939672.2939785), KDD, 2016.
5. Guolin Ke et al., [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), NeurIPS, 2017.
6. Scott Lundberg and Su-In Lee, [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874), NeurIPS, 2017.
7. James D. Hamilton, [A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle](https://www.jstor.org/stable/1912559), Econometrica, 1989.
8. Tim Bollerslev, [Generalized Autoregressive Conditional Heteroskedasticity](https://www.sciencedirect.com/science/article/abs/pii/0304407686900631), Journal of Econometrics, 1986.
9. Marco Avellaneda and Sasha Stoikov, [High-Frequency Trading in a Limit Order Book](https://www.tandfonline.com/doi/abs/10.1080/14697680701381228), Quantitative Finance, 2008.
10. Francis X. Diebold and Roberto S. Mariano, [Comparing Predictive Accuracy](https://www.tandfonline.com/doi/abs/10.1080/07350015.1995.10524599), Journal of Business and Economic Statistics, 1995.
11. Marcos Lopez de Prado, [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086), Wiley, 2018.
12. J. M. Bates and C. W. J. Granger, [The Combination of Forecasts](https://www.jstor.org/stable/3008764), OR, 1969.
13. Narasimhan Jegadeesh and Sheridan Titman, [Returns to Buying Winners and Selling Losers](https://www.jstor.org/stable/2328882), Journal of Finance, 1993.
14. Mark B. Garman and Michael J. Klass, [On the Estimation of Security Price Volatilities from Historical Data](https://www.jstor.org/stable/2352358), Journal of Business, 1980.
15. Paul A. Tetlock, [Giving Content to Investor Sentiment](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2007.01232.x), Journal of Finance, 2007.
16. R. Tyrrell Rockafellar and Stanislav Uryasev, [Optimization of Conditional Value-at-Risk](https://www.risk.net/journal-of-risk/2161159/optimization-conditional-value-risk), Journal of Risk, 2000.
17. Frank A. Sortino and Lee N. Price, [Performance Measurement in a Downside Risk Framework](https://www.pm-research.com/content/iijpormgmt/21/1/59), Journal of Investing, 1994.
18. David H. Wolpert, [Stacked Generalization](https://www.sciencedirect.com/science/article/pii/S0893608005800231), Neural Networks, 1992.

---

*All quoted metrics are from the June 25, 2026 runtime. Results will differ on subsequent runs due to live price injection and updated market data. The platform is designed for research reproducibility: saving all artifacts in Step 17 ensures any run can be reconstructed from the stored JSON, CSV, and visualization files.*
