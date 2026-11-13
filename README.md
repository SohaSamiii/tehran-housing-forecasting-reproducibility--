# Tehran Housing Prices and Market-Level Environmental Proxies

This repository contains the analysis code, input workbook, notebook, and generated outputs for the empirical study of one-month-ahead residential housing-price prediction across Tehran's 22 municipal districts.

The analysis asks whether four reconstructed, price-input-free market-level proxies—ECI, CCF, HED, and SDI—provide predictive information beyond lagged housing prices and macroeconomic conditions. The proxies are constructed from lagged exchange-rate and construction-material indicators together with prespecified seasonal parameters. They are **not** building-level measurements of energy use, embodied carbon, operational emissions, or life-cycle assessment results.

The analysis is predictive rather than causal. No coefficient, SHAP value, scenario response, or model comparison in this repository should be interpreted as a causal environmental price premium.

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── pyproject.toml
├── run_analysis.py
├── data/
│   ├── README.md
│   └── raw/Home_4.xlsx
├── docs/
│   ├── DATA_AND_METHODS.md
│   ├── DISCUSSION_NUMERIC_AUDIT.md
│   └── REPRODUCIBILITY_NOTES.md
├── notebooks/
│   └── 01_reproducible_analysis.ipynb
├── src/tehran_housing/
│   ├── __init__.py
│   └── pipeline.py
├── tests/
│   └── test_temporal_information.py
└── outputs/
    ├── data_audit.json
    ├── analysis_panel.csv
    ├── residual_diagnostics.json
    ├── reproducibility_manifest.json
    ├── tables/
    └── figures/
```

## Quick start

Use Python 3.11 or later. The exact lock file in this package records the versions used for the supplied run (Python 3.13.5). From the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install the tested dependency versions:

```bash
python -m pip install -r requirements-lock.txt
```

For a compatible environment with newer patch releases, use `requirements.txt`.

Run the complete analysis:

```bash
python run_analysis.py
```

The default command uses 2,000 bootstrap replications and 300 regional-scenario resamples. It also runs the prespecified Random Forest and XGBoost comparisons and SHAP attribution. To run only the Ridge-based core analysis:

```bash
python run_analysis.py --skip-nonlinear
```

To run nonlinear models but omit SHAP calculations:

```bash
python run_analysis.py --skip-shap
```

To specify another input workbook or output directory:

```bash
python run_analysis.py --data path/to/workbook.xlsx --output path/to/results
```

To run the notebook, open `notebooks/01_reproducible_analysis.ipynb` in JupyterLab and execute the cells from top to bottom. A short explanation of the operation appears immediately before each code cell.

## Study sample and forecast design

The raw workbook contains 89 months and 22 districts (1,958 district-month records). The primary analytical panel covers Jalali months 1397/01–1402/04 (t = 13–76), giving 64 months and 1,408 district-month observations. The training sample ends at 1401/03 (t = 63): 1,122 rows. The untouched chronological test period is 1401/04–1402/04 (t = 64–76): 13 months and 286 rows.

The target is the one-month change in log nominal housing price. A predicted log change is anchored to the **observed price in the immediately preceding month**. The code obtains this anchor directly from the lagged price array; it does not recover it by subtracting the target from the current outcome.

Hyperparameters are selected using nested expanding-window validation. The five outer validation windows are six months each. Each outer training window uses three contiguous, non-overlapping three-month inner validation blocks. Robust scaling is fitted separately within each training window using the training median and interquartile range. The final chronological test period is not used for hyperparameter selection.

## Main model specifications

- `baseline`: 17 lagged-price, momentum, district-deviation, and macroeconomic predictors.
- `baseline_season`: baseline plus three seasonal indicators (winter is the reference category).
- `macro_control`: baseline plus seasonal indicators and four unlabelled macro-ratio controls (USDREL, MASREL, STAB, PRESS).
- `full_proxy`: baseline plus the four reconstructed proxies (ECI, CCF, HED, SDI).
- `rank_macro_control`: macro-control model after removing exact linear duplicates from the baseline design.
- `rank_macro_plus_ccf`: rank-reduced macro-control model plus CCF, the only proxy that remains an additional linear dimension after the exact dependencies are removed.

This rank-adjusted comparison is important. HED is a deterministic seasonal score; ECI is a linear combination of USDREL and the seasonal score; SDI is a linear combination of STAB, PRESS, and the seasonal score; and the original baseline includes exact dependencies (`DEV1` and `LP_MA3`). CCF is a nonlinear interaction of the lagged material-cost ratio, ECI, and the seasonal multiplier. Accordingly, the controlled rank-adjusted comparison treats CCF as the only additional proxy dimension rather than interpreting the four proxy labels as four independent sources of information.

## Data-quality rules

The analysis script applies the documented factor-of-100 correction to all 22 districts in Jalali months 1401/02–1401/04. Five early USD months with inconsistent district values are set to missing and are not imputed. A retrospective centred seven-month median is used only to flag decimal-shift anomalies after the analytical period (t >= 77); these flagged observations are not used as model targets or predictors in the 64-month analytical sample. The script exports the anomaly list for audit.

Prices remain nominal. No winsorisation or inflation deflation is applied. The reported price unit is the source workbook's unit.

## Outputs

The script writes the following to `outputs/`:

- `data_audit.json`: sample checks, scale corrections, macro-series checks, and retrospective anomaly counts.
- `reproducibility_manifest.json`: Python/package versions, random seed, input workbook SHA-256, run settings, and the specification used for residual diagnostics.
- `analysis_panel.csv`: the 1,408-row modelling panel and constructed predictors.
- `tables/rolling_origin_validation.csv`: outer rolling-origin validation results.
- `tables/out_of_time_test_metrics.csv`: final test metrics for the model specifications and persistence benchmark.
- `tables/paired_bootstrap_comparisons.csv`: paired IID-row, three-month block, and district-cluster bootstrap summaries.
- `tables/discussion_numeric_audit.csv`: checked numerical values used in the Discussion, with source output and interpretation notes.
- `tables/original_proxy_contamination_audit.csv`: archived pre-reconstruction Spearman diagnostics for the obsolete price-dependent HED and SDI versions; these are not recalculated by the final pipeline.
- `tables/month_shuffle_permutation.csv`: a joint month-shuffle diagnostic for the full-proxy specification against the baseline.
- `tables/leave_one_district_out.csv`: district-omission influence analysis. Each refit omits one district from training but scores the complete test panel; this is **not** a strict held-out-district-only test.
- `tables/regional_scenario_sensitivity.csv`: proxy-parameter scenario responses by district price tercile, with test-month bootstrap intervals conditional on the fitted interaction model.
- `tables/scenario_sensitivity.csv`: overall proxy-parameter scenario sensitivity.
- `tables/nonlinear_model_comparison.csv`: prespecified Random Forest and XGBoost comparisons.
- `tables/shap_feature_attribution.csv` and `tables/shap_group_attribution.csv`: exploratory SHAP attribution outputs.
- `figures/`: figures regenerated from the data and fitted models. Figure 10 and `residual_diagnostics.json` use the final rank-reduced `rank_macro_plus_ccf` specification.

All bootstrap intervals are conditional on the fitted predictions unless otherwise stated. They do not include uncertainty from refitting the entire model-selection process. Month-block resampling retains all districts within each resampled month; it is included because macroeconomic and proxy series are common across districts in a month.

## Regional analysis terminology

The district-omission diagnostic removes each district from model training and recomputes aggregate test performance over all 22 districts. It evaluates sensitivity to the training composition. It should not be described as an out-of-sample score for the omitted district alone.

Price terciles are assigned using each district's mean log price over the training period. Scenario responses are model-based sensitivities to the constructed proxy parameters, not causal impacts. Their bootstrap intervals resample test months and are conditional on the fitted interaction model and fixed parameter values.

## Reproducibility checks

Run the automated checks locally with:

```bash
pytest
```

The tests verify the workbook structure, the 1,408-row analytical panel, lag-only feature construction, the explicit previous-price forecast anchor, and the rank of the reduced controlled design matrix. GitHub Actions runs these tests automatically on pushes and pull requests.

## Software and numerical reproducibility

The analysis uses pandas for workbook ingestion, NumPy/SciPy for numerical operations, scikit-learn for Ridge and Random Forest, XGBoost for boosting, SHAP for exploratory attribution, and Matplotlib for figures. A fixed random seed (`20261107`) is used for stochastic procedures. Small numerical differences can occur across operating systems, BLAS implementations, and library versions; the exact dependency ranges are listed in `requirements.txt`.

## Data and code permissions

The source workbook was supplied for this analysis. Its redistribution rights were not established by the workbook itself. Verify permission before publishing the data file in a public repository. No open-source licence is assigned in this package; add an appropriate code and data licence only after the authors have agreed on the terms.
