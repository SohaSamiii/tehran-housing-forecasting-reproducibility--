# Data and Methods Reference

## Panel structure

The source workbook has one row per district and Jalali month. The modelling sample is t = 13–76, where t = 1 corresponds to 1396/01. There are 22 districts and 64 modelling months.

## Target and forecasting equation

Let P(r,t) denote the nominal mean housing price in district r and month t. The target is:

`delta_y(r,t) = log(1 + P(r,t)) - log(1 + P(r,t-1))`.

The model estimates this one-month log change. The predicted level price is reconstructed as:

`P_hat(r,t) = max(exp(log(1 + P(r,t-1)) + delta_y_hat(r,t)) - 1, 0)`.

The anchor `log(1 + P(r,t-1))` is taken directly from the previous observation. The code never uses the current target price to construct the anchor.

## Predictor timing

All autoregressive, moving-average, momentum, volatility, and macroeconomic predictors are computed using values observed no later than t-1. The calendar season of t is known at the forecast origin and is represented by three seasonal dummy variables with winter as the reference category.

The four market-level proxies are:

- `ECI = 0.6 * USDREL + 0.4 * HC(season)`
- `CCF = MASREL * ECI * gamma(season)`
- `HED = HC(season)`
- `SDI = 0.35 * STAB + 0.35 * AIR(season) + 0.30 * (1 - PRESS)`

The seasonal constants are prespecified relative scores, not observed emission factors. The proxies are not direct engineering measurements or LCA scores.

## Exact linear dependencies

In the full controlled design, several terms are algebraically redundant:

- `DEV1 = LP_L1 - NAT_L1`.
- `LP_MA3 = (2 * LP_L1 - MOM1 + LP_L3) / 3`.
- `HED` is determined by seasonal indicators and the intercept.
- `ECI` is determined by `USDREL`, seasonal indicators, and the intercept.
- `SDI` is determined by `STAB`, `PRESS`, seasonal indicators, and the intercept.

The rank-reduced controlled comparison removes `LP_MA3`, `DEV1`, `HED`, `ECI`, and `SDI`, retaining CCF as the additional proxy feature. This avoids presenting redundant variables as independent information dimensions in the controlled linear model.

## Validation and inference

The outer validation windows are: train t=13–33 / validate t=34–39; train t=13–39 / validate t=40–45; train t=13–45 / validate t=46–51; train t=13–51 / validate t=52–57; train t=13–57 / validate t=58–63. Each outer training period uses three trailing, contiguous, three-month inner validation blocks to select Ridge alpha from a 41-point logarithmic grid from 1e-6 to 1e4.

Final training uses t=13–63 and the untouched test uses t=64–76. Robust scaling is fitted only on the corresponding training data.

Paired bootstrap comparisons use 2,000 replications by default. Three schemes are reported: IID district-month resampling, moving three-month block resampling with all districts kept together within each sampled month, and district-cluster resampling. The monthly block bootstrap is the most relevant of these for common temporal shocks, but none of these conditional prediction bootstraps includes uncertainty from refitting the full model-selection procedure.

The month-shuffle permutation diagnostic jointly permutes the four proxy series across months and compares the proxy-augmented model with the baseline. It does not test whether proxies add information beyond macro-ratio controls.

## Spatial and scenario terminology

The district-omission analysis removes one district from model training and scores the full 13-month test panel. It is a training-composition sensitivity analysis, not a strict spatial holdout score for the omitted district alone.

Scenario perturbations modify the constructed seasonal and proxy parameters and recompute dependent proxy values. Scenario responses are model-based sensitivities, not estimates of causal effects. Price terciles are based on mean district log prices in the training period.
