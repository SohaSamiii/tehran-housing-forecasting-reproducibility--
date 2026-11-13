from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tehran_housing.pipeline import (  # noqa: E402
    FIRST_ANALYSIS_MONTH,
    LAST_ANALYSIS_MONTH,
    TRAIN_END_MONTH,
    TEST_START_MONTH,
    build_analysis_frame,
    load_study_data,
)

DATA_PATH = ROOT / "data" / "raw" / "Home_4.xlsx"


def _study():
    return load_study_data(DATA_PATH)


def test_loaded_price_array_is_writable_after_scale_correction():
    study = _study()
    assert study.prices.flags.writeable
    assert study.prices.shape == (89, 22)
    assert np.isfinite(study.prices).all()
    assert (study.prices > 0).all()


def test_workbook_and_analysis_sample_dimensions():
    study = _study()
    frame = build_analysis_frame(study).rows
    assert study.raw.shape[0] == 1958
    assert study.raw["district"].nunique() == 22
    assert study.raw["t"].nunique() == 89
    assert frame.shape[0] == 1408
    assert frame["t"].min() == FIRST_ANALYSIS_MONTH
    assert frame["t"].max() == LAST_ANALYSIS_MONTH
    assert ((frame["t"] <= TRAIN_END_MONTH).sum()) == 1122
    assert ((frame["t"] >= TEST_START_MONTH).sum()) == 286


def test_current_month_target_does_not_enter_its_own_features():
    study = _study()
    original = build_analysis_frame(study).rows
    changed_prices = study.prices.copy()
    target_month = 30
    district = 5
    changed_prices[target_month - 1, district - 1] *= 1.25
    altered_study = replace(study, prices=changed_prices)
    altered = build_analysis_frame(altered_study).rows

    row_original = original.query("t == @target_month and district == @district").iloc[0]
    row_altered = altered.query("t == @target_month and district == @district").iloc[0]
    feature_columns = [
        c for c in original.select_dtypes(include=[np.number]).columns
        if c not in {"price", "target_log_growth"}
    ]
    np.testing.assert_allclose(
        row_original[feature_columns].to_numpy(dtype=float),
        row_altered[feature_columns].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )
    assert not np.isclose(row_original["target_log_growth"], row_altered["target_log_growth"])


def test_anchor_is_directly_taken_from_previous_observed_price():
    study = _study()
    frame = build_analysis_frame(study).rows
    month = 64
    district = 1
    row = frame.query("t == @month and district == @district").iloc[0]
    expected = np.log1p(study.prices[month - 2, district - 1])
    assert np.isclose(row["anchor_log_price"], expected)
    assert np.isclose(row["anchor_price"], study.prices[month - 2, district - 1])


def test_same_month_macro_value_does_not_enter_current_month_features():
    study = _study()
    original = build_analysis_frame(study).rows
    altered_macro = study.macro.copy()
    target_month = 40
    altered_macro.loc[target_month, "usd"] *= 1.50
    altered_study = replace(study, macro=altered_macro)
    altered = build_analysis_frame(altered_study).rows

    a = original.query("t == @target_month and district == 7").iloc[0]
    b = altered.query("t == @target_month and district == 7").iloc[0]
    cols = ["USDREL", "MASREL", "STAB", "PRESS", "USD1", "dUSD3", "ECI", "CCF", "SDI"]
    np.testing.assert_allclose(a[cols].to_numpy(dtype=float), b[cols].to_numpy(dtype=float),
                               rtol=0, atol=1e-12)


def test_rank_reduced_control_design_has_no_exact_linear_dependencies():
    frame = build_analysis_frame(_study())
    rows = frame.rows
    train = rows["t"].between(FIRST_ANALYSIS_MONTH, TRAIN_END_MONTH).to_numpy()
    for spec in ["rank_macro_control", "rank_macro_plus_ccf"]:
        x = rows.loc[train, frame.specifications[spec]].to_numpy(dtype=float)
        med = np.median(x, axis=0)
        q25 = np.percentile(x, 25, axis=0)
        q75 = np.percentile(x, 75, axis=0)
        scale = np.where(q75 - q25 == 0, 1.0, q75 - q25)
        scaled = (x - med) / scale
        design = np.column_stack([np.ones(scaled.shape[0]), scaled])
        assert np.linalg.matrix_rank(design) == design.shape[1]
