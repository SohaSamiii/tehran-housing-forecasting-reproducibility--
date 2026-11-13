# Repository release notes

## Reviewer package — corrected reproducibility build

This package includes the writable-array correction in `load_study_data`.
The price matrix is copied into an owned, C-contiguous `float64` NumPy array
before the prespecified factor-of-100 corrections are applied. An automated
test verifies that the loaded price matrix remains writable.

Residual diagnostics and Figure 10 are generated from
`rank_macro_plus_ccf`, the rank-reduced macro-control model augmented with
CCF. The full four-proxy specification remains the reference for the full-proxy
scenario analysis, SHAP attribution, and nonlinear model comparison.

The repository test suite contains six tests covering workbook dimensions,
writable price storage, lag-only feature construction, direct previous-price
anchoring, same-month macro exclusion, and full-column-rank checks for the
rank-reduced model designs.
