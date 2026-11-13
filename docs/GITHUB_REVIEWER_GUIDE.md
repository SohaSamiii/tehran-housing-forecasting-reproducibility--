# Reviewer guide

## Recommended review path

1. Read `README.md` for the research question, sample, model definitions, and limitations.
2. Read `docs/DATA_AND_METHODS.md` for the operational definitions and feature construction.
3. Open `notebooks/01_reproducible_analysis.ipynb` and run the cells in order. Each analytical code cell is preceded by a short stage description.
4. Alternatively, run `python run_analysis.py` from the repository root.
5. Run `pytest` to execute the automated temporal-information and design-matrix checks.
6. Inspect `outputs/reproducibility_manifest.json` for the input workbook hash and software versions, then review the CSV files in `outputs/tables/` and PNG files in `outputs/figures/`.
7. Use `docs/DISCUSSION_NUMERIC_AUDIT.md` and `outputs/tables/discussion_numeric_audit.csv` to trace numerical claims in the Discussion to their generating outputs. The legacy HED/SDI contamination correlations are explicitly labelled as archived pre-reconstruction diagnostics.

## Important interpretation limits

- The proxies are heuristic market-level indices, not building-level energy, carbon, or life-cycle measurements.
- The model comparison is predictive, not causal.
- The month-shuffle permutation test compares the original full-proxy model with the baseline; it does not test incremental value over the macro-ratio control.
- The controlled rank-reduced comparison is the relevant specification for assessing whether CCF adds information beyond the macro-ratio controls and seasonal terms.
- District omission evaluates sensitivity to training composition because the complete test panel is scored after each refit; it is not a strict held-out-district test.
- Bootstrap intervals are conditional on fitted predictions and do not include full model-selection uncertainty.

## Data availability

The raw workbook is included for convenience in this review package. Before making the repository public, the authors should confirm that they have permission to redistribute the workbook. If redistribution is not permitted, remove `data/raw/Home_4.xlsx`, retain `data/README.md`, and update the README with an authorised data-access procedure.
