# Reproducibility Notes for Reviewers

1. Use the supplied workbook `data/raw/Home_4.xlsx`, worksheet `Sheet2`, unless an authorised replacement is supplied.
2. Run the notebook from top to bottom or execute `python run_analysis.py` from the repository root.
3. Do not use a random row-wise train/test split. The study's test period is chronological and remains untouched during model selection.
4. The forecast target is monthly log-price growth. The level forecast is anchored to the observed price at t-1 using a direct lag lookup.
5. Seasonal and proxy terms are constructed without contemporaneous property prices, price ranks, or housing inflation. Their status as market-level heuristic proxies must be retained in any interpretation.
6. When evaluating incremental information, use the rank-reduced macro-control model and its CCF-augmented counterpart. The original four-proxy model is retained as a descriptive benchmark, but three of its proxies are exact linear combinations of the included controls and seasonal variables.
7. Interpret month-block bootstrap intervals as conditional on fitted predictions. They do not capture full model-selection uncertainty.
8. Interpret the month-shuffle permutation test as a comparison against the baseline, not against the macro-ratio control.
9. The district-omission output removes one district from the training sample but evaluates the complete test panel. It is not a pure held-out-district evaluation.
10. Scenario results represent changes in model predictions under perturbations of constructed proxy parameters. They are not causal effects and should not be described as measured carbon, energy, or sustainability impacts.

The analysis outputs are written to `outputs/`. Existing contents of that directory are example run products; rerun the pipeline to regenerate them in the current environment.

11. Residual diagnostics and Figure 10 are computed from `rank_macro_plus_ccf`, the rank-reduced macro-control specification augmented with CCF. The full four-proxy model remains the descriptive benchmark for the primary test metrics and scenario sensitivity.
12. The original within-month Spearman correlations for the price-contaminated HED and SDI versions (1.000 and -0.975367) are retained as an archived pre-reconstruction diagnostic. The final pipeline does not reconstruct those obsolete price-dependent versions; the reported values are documented separately in `outputs/tables/original_proxy_contamination_audit.csv`.
13. In the rank-reduced comparison, the observed test-set RMSE change after adding CCF is distinct from the median bootstrap change. Report both separately: observed point change approximately -0.43%; three-month block-bootstrap median approximately -0.57%, with the 95% percentile interval shown in `outputs/tables/paired_bootstrap_comparisons.csv`.
