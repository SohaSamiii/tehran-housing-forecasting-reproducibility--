"""Reproducible analysis pipeline for the Tehran housing-price study.

The pipeline uses only information available at or before each forecast origin.
It writes audit tables, model metrics, bootstrap summaries, regional diagnostics,
scenario sensitivities, and publication-ready figures to the selected output folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import warnings
import hashlib
import platform
from importlib import metadata as package_metadata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 20261107
N_MONTHS = 89
N_DISTRICTS = 22
FIRST_ANALYSIS_MONTH = 13
LAST_ANALYSIS_MONTH = 76
TRAIN_END_MONTH = 63
TEST_START_MONTH = 64
ALPHAS = np.logspace(-6, 4, 41)

HC_MAP = {1: .8, 2: .8, 3: .8, 4: 1.4, 5: 1.4, 6: 1.4,
          7: .9, 8: .9, 9: .9, 10: 1.5, 11: 1.5, 12: 1.5}
GAMMA_MAP = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.2, 5: 1.2, 6: 1.2,
             7: 1.1, 8: 1.1, 9: 1.1, 10: 1.3, 11: 1.3, 12: 1.3}
AIR_MAP = {1: 1.0, 2: 1.0, 3: 1.0, 4: .8, 5: .8, 6: .8,
           7: .5, 8: .5, 9: .5, 10: .3, 11: .3, 12: .3}

B_FEATURES = [
    "LP_L1", "LP_L3", "LP_L6", "LP_L12",
    "LP_MA3", "LP_MA6", "LP_MA12", "MOM1", "VOL6",
    "NAT_L1", "DEV1", "USD1", "dUSD3", "GOLD1", "BORS1", "TAV1", "MAS1",
]
S_FEATURES = ["S_spring", "S_summer", "S_autumn"]
M_FEATURES = ["USDREL", "MASREL", "STAB", "PRESS"]
E_FEATURES = ["ECI", "CCF", "HED", "SDI"]


@dataclass
class StudyData:
    raw: pd.DataFrame
    prices: np.ndarray
    macro: pd.DataFrame
    audit: dict[str, Any]
    decimal_flags: pd.DataFrame


@dataclass
class AnalysisFrame:
    rows: pd.DataFrame
    specifications: dict[str, list[str]]


@dataclass
class FittedResult:
    name: str
    alpha: float
    model: Ridge
    medians: np.ndarray
    scales: np.ndarray
    feature_names: list[str]
    metrics: dict[str, float]
    predictions: pd.DataFrame
    alpha_profile: pd.DataFrame


def jalali_label(t: int) -> str:
    year = 1396 + (int(t) - 1) // 12
    month = (int(t) - 1) % 12 + 1
    return f"{year}/{month:02d}"


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lookup = {str(c).strip().upper(): c for c in frame.columns}
    expected = {
        "YEAR": "year", "MONTH": "month", "REGION": "district",
        "AVERAGE PRICE": "price", "SHAKHES MASALEH": "mas",
        "TAVAROM MASKAN": "tav", "SHAKHES BORS": "bors",
        "USD": "usd", "GOLD": "gold",
    }
    missing = [key for key in expected if key not in lookup]
    if missing:
        raise ValueError(f"Missing expected workbook columns: {missing}")
    return frame.rename(columns={lookup[k]: v for k, v in expected.items()})[list(expected.values())]


def _decimal_shift_audit(price_matrix: np.ndarray) -> pd.DataFrame:
    """Flag retrospective decimal-shift anomalies before constructing the model sample."""
    records: list[dict[str, Any]] = []
    logp = np.where(price_matrix > 0, np.log(price_matrix), np.nan)
    cross_median = np.nanmedian(logp, axis=1)
    centred = logp - cross_median[:, None]
    for district in range(price_matrix.shape[1]):
        series = pd.Series(centred[:, district])
        rolling = series.rolling(window=7, center=True, min_periods=3).median().to_numpy()
        residual = series.to_numpy() - rolling
        for idx in range(price_matrix.shape[0]):
            if not np.isfinite(residual[idx]) or abs(residual[idx]) <= 0.40:
                continue
            k = int(np.rint(residual[idx] / np.log(10.0)))
            original = price_matrix[idx, district]
            if k != 0 and abs(residual[idx] - k * np.log(10.0)) <= 0.40:
                cleaned = original * (10.0 ** (-k))
                action = f"decimal-shift correction x10^{(-k)}; diagnostic only"
            else:
                cleaned = np.nan
                action = "unrecoverable; diagnostic only"
            records.append({
                "t": idx + 1, "jalali_month": jalali_label(idx + 1),
                "district": district + 1, "original_price": original,
                "diagnostic_price": cleaned, "centred_residual": residual[idx],
                "action": action,
            })
    return pd.DataFrame(records)


def load_study_data(workbook_path: str | Path, sheet_name: str = "Sheet2") -> StudyData:
    """Load, validate, and apply the prespecified price-scale and macro-series rules."""
    workbook_path = Path(workbook_path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")
    raw = _normalise_columns(pd.read_excel(workbook_path, sheet_name=sheet_name, engine="openpyxl"))
    for col in ["year", "month", "district"]:
        raw[col] = pd.to_numeric(raw[col], errors="raise").astype(int)
    for col in ["price", "mas", "tav", "bors", "usd", "gold"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["t"] = (raw["year"] - 1396) * 12 + raw["month"]
    raw["jalali_month"] = raw["t"].map(jalali_label)

    if raw.duplicated(["t", "district"]).any():
        duplicates = raw.loc[raw.duplicated(["t", "district"], keep=False), ["t", "district"]]
        raise ValueError(f"Duplicate district-month rows found:\n{duplicates.head()}")
    if raw["district"].nunique() != N_DISTRICTS:
        raise ValueError(f"Expected {N_DISTRICTS} districts; found {raw['district'].nunique()}.")
    if raw["t"].nunique() != N_MONTHS:
        raise ValueError(f"Expected {N_MONTHS} months; found {raw['t'].nunique()}.")
    if len(raw) != N_MONTHS * N_DISTRICTS:
        raise ValueError(f"Expected {N_MONTHS*N_DISTRICTS} rows; found {len(raw)}.")

    # Build an owned, writable NumPy array. Under pandas Copy-on-Write,
    # DataFrame.to_numpy() may expose a read-only view of the underlying data.
    # The subsequent scale correction is intentionally in-place, so make an
    # explicit independent copy before modifying any values.
    price_frame = raw.pivot(index="t", columns="district", values="price").reindex(
        index=range(1, N_MONTHS + 1), columns=range(1, N_DISTRICTS + 1)
    )
    price = np.array(
        price_frame.to_numpy(dtype=np.float64, copy=True),
        dtype=np.float64,
        copy=True,
        order="C",
    )
    if not price.flags.writeable:
        price.setflags(write=True)
    if np.isnan(price).any() or (price <= 0).any():
        raise ValueError("The analysis panel contains missing or non-positive raw prices.")

    scale_months = [62, 63, 64]
    scale_indices = np.asarray(scale_months, dtype=int) - 1
    scale_before = price[scale_indices, :].copy()
    price[scale_indices, :] = price[scale_indices, :] / 100.0
    scale_after = price[np.array(scale_months) - 1, :]
    decimal_flags = _decimal_shift_audit(price)
    in_sample_flags = decimal_flags.loc[decimal_flags["t"] <= LAST_ANALYSIS_MONTH]
    if not in_sample_flags.empty:
        raise ValueError(
            "Decimal-shift diagnostics flagged observations inside the analytical period. "
            "Resolve these data-quality issues before fitting the forecasting models."
        )

    macro = raw.groupby("t")[["mas", "tav", "bors", "usd", "gold"]].mean().reindex(range(1, N_MONTHS + 1))
    usd_inconsistent_months = [1, 3, 4, 5, 6]
    macro.loc[usd_inconsistent_months, "usd"] = np.nan

    macro_spread = raw.groupby("t")[["mas", "tav", "bors", "usd", "gold"]].agg(
        lambda s: float(s.max() - s.min()) if s.notna().any() else np.nan
    )
    audit = {
        "input_file": workbook_path.name,
        "sheet_name": sheet_name,
        "n_rows": int(len(raw)),
        "n_months": int(raw["t"].nunique()),
        "n_districts": int(raw["district"].nunique()),
        "analysis_start": jalali_label(FIRST_ANALYSIS_MONTH),
        "analysis_end": jalali_label(LAST_ANALYSIS_MONTH),
        "training_end": jalali_label(TRAIN_END_MONTH),
        "test_start": jalali_label(TEST_START_MONTH),
        "test_end": jalali_label(LAST_ANALYSIS_MONTH),
        "scale_error_months": [jalali_label(t) for t in scale_months],
        "scale_error_rows_corrected": int(len(scale_months) * N_DISTRICTS),
        "scale_error_min_before": float(np.nanmin(scale_before)),
        "scale_error_max_before": float(np.nanmax(scale_before)),
        "scale_error_min_after": float(np.nanmin(scale_after)),
        "scale_error_max_after": float(np.nanmax(scale_after)),
        "usd_inconsistent_months_set_missing": [jalali_label(t) for t in usd_inconsistent_months],
        "decimal_shift_flags_in_analysis_period": int((decimal_flags["t"] <= LAST_ANALYSIS_MONTH).sum()) if len(decimal_flags) else 0,
        "decimal_shift_flags_after_analysis_period": int((decimal_flags["t"] > LAST_ANALYSIS_MONTH).sum()) if len(decimal_flags) else 0,
        "decimal_shift_correctable_after_analysis_period": int(
            ((decimal_flags["t"] > LAST_ANALYSIS_MONTH) & decimal_flags["action"].str.startswith("decimal-shift")).sum()
        ) if len(decimal_flags) else 0,
        "decimal_shift_unrecoverable_after_analysis_period": int(
            ((decimal_flags["t"] > LAST_ANALYSIS_MONTH) & decimal_flags["action"].str.startswith("unrecoverable")).sum()
        ) if len(decimal_flags) else 0,
        "max_macro_within_month_range": {c: float(macro_spread[c].max()) for c in macro_spread.columns},
        "nominal_prices": True,
        "winsorisation_applied": False,
    }
    return StudyData(raw=raw, prices=price, macro=macro, audit=audit, decimal_flags=decimal_flags)


def _expanding_mean(values: np.ndarray, end_exclusive: int) -> float:
    available = values[:end_exclusive]
    available = available[np.isfinite(available)]
    return float(np.mean(available)) if available.size else np.nan


def build_analysis_frame(study: StudyData) -> AnalysisFrame:
    """Construct all predictors from data available no later than month t-1."""
    p = study.prices
    y = np.log1p(p)
    dy = np.full_like(y, np.nan, dtype=float)
    dy[1:, :] = y[1:, :] - y[:-1, :]
    macro = study.macro
    usd = macro["usd"].to_numpy(dtype=float)
    mas = macro["mas"].to_numpy(dtype=float)
    tav = macro["tav"].to_numpy(dtype=float)
    bors = macro["bors"].to_numpy(dtype=float)
    gold = macro["gold"].to_numpy(dtype=float)
    dlog_mas = np.diff(np.log(mas))

    records: list[dict[str, Any]] = []
    for t1 in range(FIRST_ANALYSIS_MONTH, LAST_ANALYSIS_MONTH + 1):
        t0 = t1 - 1
        month = (t1 - 1) % 12 + 1
        season = 1 if month <= 3 else 2 if month <= 6 else 3 if month <= 9 else 4
        hc = HC_MAP[month]
        gamma = GAMMA_MAP[month]
        air = AIR_MAP[month]
        usd_rel = usd[t0 - 1] / _expanding_mean(usd, t0)
        mas_rel = mas[t0 - 1] / _expanding_mean(mas, t0)
        recent_usd_changes = np.array([
            np.log(usd[t0 - 1 - lag]) - np.log(usd[t0 - 2 - lag])
            for lag in range(3)
        ], dtype=float)
        if recent_usd_changes.size != 3 or not np.isfinite(recent_usd_changes).all():
            raise ValueError(f"USD stability feature unavailable for target month {t1}.")
        stability = 1.0 / (1.0 + 100.0 * np.std(recent_usd_changes, ddof=1))
        pressure_idx = t0 - 2
        past_material_growth = dlog_mas[:pressure_idx + 1]
        past_material_growth = past_material_growth[np.isfinite(past_material_growth)]
        press = float(np.mean(past_material_growth <= dlog_mas[pressure_idx]))

        for district in range(1, N_DISTRICTS + 1):
            r0 = district - 1
            row: dict[str, Any] = {
                "t": t1,
                "jalali_month": jalali_label(t1),
                "district": district,
                "season": season,
                "price": float(p[t0, r0]),
                "target_log_growth": float(dy[t0, r0]),
                "anchor_log_price": float(y[t0 - 1, r0]),
                "anchor_price": float(p[t0 - 1, r0]),
                "HC_s": hc,
                "gamma_s": gamma,
                "AIR_s": air,
                "USDREL": float(usd_rel),
                "MASREL": float(mas_rel),
                "STAB": float(stability),
                "PRESS": float(press),
            }
            for lag in [1, 3, 6, 12]:
                row[f"LP_L{lag}"] = float(y[t0 - lag, r0])
            for window in [3, 6, 12]:
                row[f"LP_MA{window}"] = float(np.mean(y[t0 - window:t0, r0]))
            row["MOM1"] = float(y[t0 - 1, r0] - y[t0 - 2, r0])
            row["VOL6"] = float(np.std(dy[t0 - 6:t0, r0], ddof=1))
            nat_l1 = float(np.nanmean(y[t0 - 1, :]))
            row["NAT_L1"] = nat_l1
            row["DEV1"] = float(y[t0 - 1, r0] - nat_l1)
            row["USD1"] = float(np.log(usd[t0 - 1]))
            row["dUSD3"] = float(np.log(usd[t0 - 1]) - np.log(usd[t0 - 4]))
            row["GOLD1"] = float(np.log(gold[t0 - 1]))
            row["BORS1"] = float(np.log(bors[t0 - 1]))
            row["TAV1"] = float(tav[t0 - 1])
            row["MAS1"] = float(np.log(mas[t0 - 1]))
            row["S_spring"] = float(season == 1)
            row["S_summer"] = float(season == 2)
            row["S_autumn"] = float(season == 3)
            row["ECI"] = float(.6 * usd_rel + .4 * hc)
            row["CCF"] = float(mas_rel * row["ECI"] * gamma)
            row["HED"] = float(hc)
            row["SDI"] = float(.35 * stability + .35 * air + .30 * (1.0 - press))
            records.append(row)

    frame = pd.DataFrame.from_records(records)
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        bad = numeric.columns[~np.isfinite(numeric.to_numpy()).all(axis=0)].tolist()
        raise ValueError(f"Non-finite values found in analysis features: {bad}")
    specs = {
        "baseline": B_FEATURES,
        "baseline_season": B_FEATURES + S_FEATURES,
        "macro_control": B_FEATURES + S_FEATURES + M_FEATURES,
        "full_proxy": B_FEATURES + E_FEATURES,
    }
    rank_base = [x for x in B_FEATURES if x not in {"LP_MA3", "DEV1"}]
    specs["rank_macro_control"] = rank_base + S_FEATURES + M_FEATURES
    specs["rank_macro_plus_ccf"] = rank_base + S_FEATURES + M_FEATURES + ["CCF"]
    return AnalysisFrame(rows=frame, specifications=specs)


def robust_scale_fit(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(train_x, axis=0)
    q25 = np.nanpercentile(train_x, 25, axis=0)
    q75 = np.nanpercentile(train_x, 75, axis=0)
    scales = q75 - q25
    scales = np.where(scales == 0.0, 1.0, scales)
    return medians, scales


def _scale(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    medians, scales = robust_scale_fit(train_x)
    return (train_x - medians) / scales, (eval_x - medians) / scales, medians, scales


def _inner_blocks(max_train_month: int) -> list[tuple[int, int]]:
    return [
        (max_train_month - 8, max_train_month - 6),
        (max_train_month - 5, max_train_month - 3),
        (max_train_month - 2, max_train_month),
    ]


def tune_alpha(
    x: np.ndarray,
    y: np.ndarray,
    months: np.ndarray,
    allowed_train: np.ndarray,
    alphas: np.ndarray = ALPHAS,
) -> tuple[float, pd.DataFrame]:
    """Select alpha using three contiguous 3-month inner validation blocks.

    The ridge path is evaluated from one eigendecomposition per inner fold. This is
    algebraically equivalent to fitting Ridge separately at every alpha, while making
    repeated leave-one-district-out validation practical.
    """
    max_train_month = int(months[allowed_train].max())
    fold_mse = []
    for start, end in _inner_blocks(max_train_month):
        tr = allowed_train & (months <= start - 1)
        va = allowed_train & (months >= start) & (months <= end)
        if tr.sum() == 0 or va.sum() == 0:
            raise ValueError("An inner validation block has no training or validation rows.")
        xtr, xva, _, _ = _scale(x[tr], x[va])
        ytr = y[tr]
        x_mean = xtr.mean(axis=0)
        y_mean = float(ytr.mean())
        xc = xtr - x_mean
        yc = ytr - y_mean
        gram = xc.T @ xc
        cross = xc.T @ yc
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        projected = eigenvectors.T @ cross
        coefficients = eigenvectors @ (projected[:, None] / (eigenvalues[:, None] + alphas[None, :]))
        intercepts = y_mean - x_mean @ coefficients
        predictions = (xva @ coefficients) + intercepts
        mse = np.mean((y[va, None] - predictions) ** 2, axis=0)
        fold_mse.append(np.sqrt(mse))
    mean_rmse = np.mean(np.vstack(fold_mse), axis=0)
    profile = pd.DataFrame({"alpha": alphas.astype(float), "mean_inner_RMSElog": mean_rmse})
    best = profile.loc[profile["mean_inner_RMSElog"].idxmin(), "alpha"]
    return float(best), profile


def _metrics(actual_price: np.ndarray, predicted_price: np.ndarray,
             actual_growth: np.ndarray, predicted_growth: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(actual_price, predicted_price)))
    return {
        "R2": float(r2_score(actual_price, predicted_price)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(actual_price, predicted_price)),
        "MAPE_pct": float(np.mean(np.abs(actual_price - predicted_price) / actual_price) * 100.0),
        "RMSElog": float(np.sqrt(mean_squared_error(actual_growth, predicted_growth))),
        "Growth_R2": float(1.0 - np.sum((actual_growth - predicted_growth) ** 2)
                            / np.sum((actual_growth - actual_growth.mean()) ** 2)),
    }


def fit_specification(
    rows: pd.DataFrame,
    feature_names: list[str],
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    name: str,
    fixed_alpha: float | None = None,
) -> FittedResult:
    x = rows[feature_names].to_numpy(dtype=float)
    y = rows["target_log_growth"].to_numpy(dtype=float)
    months = rows["t"].to_numpy(dtype=int)
    x_train, x_eval = x[train_mask], x[eval_mask]
    if fixed_alpha is None:
        alpha, profile = tune_alpha(x, y, months, train_mask)
    else:
        alpha = float(fixed_alpha)
        profile = pd.DataFrame({"alpha": [alpha], "mean_inner_RMSElog": [np.nan]})
    x_train_s, x_eval_s, medians, scales = _scale(x_train, x_eval)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x_train_s, y[train_mask])
    growth_hat = model.predict(x_eval_s)
    eval_rows = rows.loc[eval_mask].copy()
    predicted_price = np.maximum(np.expm1(eval_rows["anchor_log_price"].to_numpy() + growth_hat), 0.0)
    actual_price = eval_rows["price"].to_numpy(dtype=float)
    actual_growth = eval_rows["target_log_growth"].to_numpy(dtype=float)
    metrics = _metrics(actual_price, predicted_price, actual_growth, growth_hat)
    predictions = eval_rows[["t", "jalali_month", "district", "price", "anchor_price", "anchor_log_price", "target_log_growth"]].copy()
    predictions["predicted_growth"] = growth_hat
    predictions["predicted_price"] = predicted_price
    predictions["price_residual"] = actual_price - predicted_price
    predictions["growth_residual"] = actual_growth - growth_hat
    return FittedResult(name, alpha, model, medians, scales, feature_names, metrics, predictions, profile)


def run_rolling_validation(rows: pd.DataFrame, specifications: dict[str, list[str]]) -> pd.DataFrame:
    folds = [(13, 33, 34, 39), (13, 39, 40, 45), (13, 45, 46, 51),
             (13, 51, 52, 57), (13, 57, 58, 63)]
    output: list[dict[str, Any]] = []
    months = rows["t"].to_numpy(dtype=int)
    for spec_name in ["baseline", "baseline_season", "macro_control", "full_proxy",
                      "rank_macro_control", "rank_macro_plus_ccf"]:
        feature_names = specifications[spec_name]
        x = rows[feature_names].to_numpy(dtype=float)
        y = rows["target_log_growth"].to_numpy(dtype=float)
        for fold_id, (train_start, train_end, val_start, val_end) in enumerate(folds, start=1):
            train_mask = (months >= train_start) & (months <= train_end)
            val_mask = (months >= val_start) & (months <= val_end)
            result = fit_specification(rows, feature_names, train_mask, val_mask,
                                       f"{spec_name}_fold{fold_id}")
            output.append({"specification": spec_name, "fold": fold_id,
                           "train_start_t": train_start, "train_end_t": train_end,
                           "validation_start_t": val_start, "validation_end_t": val_end,
                           "alpha": result.alpha, **result.metrics})
    return pd.DataFrame(output)


def run_final_models(rows: pd.DataFrame, specifications: dict[str, list[str]]) -> dict[str, FittedResult]:
    months = rows["t"].to_numpy(dtype=int)
    train_mask = (months >= FIRST_ANALYSIS_MONTH) & (months <= TRAIN_END_MONTH)
    test_mask = (months >= TEST_START_MONTH) & (months <= LAST_ANALYSIS_MONTH)
    results = {}
    for spec_name, feature_names in specifications.items():
        results[spec_name] = fit_specification(rows, feature_names, train_mask, test_mask, spec_name)
    return results


def persistence_predictions(rows: pd.DataFrame, test_mask: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    test = rows.loc[test_mask].copy()
    pred = test["anchor_price"].to_numpy(dtype=float)
    metrics = _metrics(test["price"].to_numpy(dtype=float), pred,
                       test["target_log_growth"].to_numpy(dtype=float),
                       np.zeros(test.shape[0], dtype=float))
    out = test[["t", "jalali_month", "district", "price", "anchor_price"]].copy()
    out["predicted_price"] = pred
    out["price_residual"] = out["price"] - pred
    return metrics, out


def paired_bootstrap(
    actual: np.ndarray,
    pred_reference: np.ndarray,
    pred_comparison: np.ndarray,
    test_months: np.ndarray,
    districts: np.ndarray,
    reps: int = 2000,
    seed: int = SEED,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap of percentage RMSE change; block draws keep all districts per month."""
    rng = np.random.default_rng(seed)
    month_values = np.unique(test_months)
    month_to_idx = {m: np.flatnonzero(test_months == m) for m in month_values}
    district_values = np.unique(districts)
    samples: dict[str, list[float]] = {"iid_rows": [], "month_blocks_3": [], "district_clusters": []}

    def delta(ix: np.ndarray) -> float:
        rmse_ref = np.sqrt(np.mean((actual[ix] - pred_reference[ix]) ** 2))
        rmse_cmp = np.sqrt(np.mean((actual[ix] - pred_comparison[ix]) ** 2))
        return float(100.0 * (rmse_cmp / rmse_ref - 1.0))

    for _ in range(reps):
        ix = rng.integers(0, len(actual), len(actual))
        samples["iid_rows"].append(delta(ix))

        selected_months: list[int] = []
        while len(selected_months) < len(month_values):
            start = int(rng.integers(0, len(month_values) - 2))
            selected_months.extend(month_values[start:start + 3].tolist())
        ix = np.concatenate([month_to_idx[m] for m in selected_months[:len(month_values)]])
        samples["month_blocks_3"].append(delta(ix))

        selected_districts = rng.choice(district_values, size=len(district_values), replace=True)
        ix = np.concatenate([np.flatnonzero(districts == d) for d in selected_districts])
        samples["district_clusters"].append(delta(ix))

    summary = {}
    for method, values in samples.items():
        arr = np.asarray(values, dtype=float)
        summary[method] = {
            "median_pct": float(np.median(arr)),
            "ci_low_pct": float(np.percentile(arr, 2.5)),
            "ci_high_pct": float(np.percentile(arr, 97.5)),
            "share_improvement": float(np.mean(arr < 0.0)),
        }
    return summary



def run_month_shuffle_permutation(
    rows: pd.DataFrame,
    baseline_result: FittedResult,
    full_result: FittedResult,
    baseline_features: list[str],
    full_features: list[str],
    reps: int = 500,
    seed: int = SEED,
) -> pd.DataFrame:
    """Jointly shuffle the monthly ordering of all proxies; compare full model with baseline.

    The test is a prespecified diagnostic against the baseline specification. It does not
    test whether the proxy transformations add information beyond macro-ratio controls.
    """
    months = rows["t"].to_numpy(dtype=int)
    train = (months >= FIRST_ANALYSIS_MONTH) & (months <= TRAIN_END_MONTH)
    test = (months >= TEST_START_MONTH) & (months <= LAST_ANALYSIS_MONTH)
    baseline_pred = baseline_result.predictions["predicted_price"].to_numpy(dtype=float)
    actual = rows.loc[test, "price"].to_numpy(dtype=float)
    baseline_rmse = float(np.sqrt(np.mean((actual - baseline_pred) ** 2)))
    observed_full = full_result.predictions["predicted_price"].to_numpy(dtype=float)
    observed_delta = float(np.sqrt(np.mean((actual - observed_full) ** 2)) - baseline_rmse)

    all_months = np.sort(rows["t"].unique())
    month_to_proxy = rows.drop_duplicates("t").set_index("t")[E_FEATURES]
    rng = np.random.default_rng(seed)
    permuted_deltas = []
    x_base = rows[baseline_features].to_numpy(dtype=float)
    for _ in range(reps):
        shuffled = rng.permutation(all_months)
        mapping = {int(target): int(source) for target, source in zip(all_months, shuffled)}
        permuted = rows[full_features].copy()
        source_rows = rows["t"].map(mapping)
        proxy_lookup = month_to_proxy
        for proxy in E_FEATURES:
            permuted[proxy] = source_rows.map(proxy_lookup[proxy])
        x = permuted.to_numpy(dtype=float)
        xtr, xte, med, scale = _scale(x[train], x[test])
        model = Ridge(alpha=full_result.alpha, fit_intercept=True)
        model.fit(xtr, rows.loc[train, "target_log_growth"].to_numpy(dtype=float))
        growth_hat = model.predict(xte)
        test_rows = rows.loc[test]
        price_hat = np.maximum(np.expm1(test_rows["anchor_log_price"].to_numpy() + growth_hat), 0.0)
        permuted_deltas.append(float(np.sqrt(np.mean((actual - price_hat) ** 2)) - baseline_rmse))

    values = np.asarray(permuted_deltas)
    p_value = (1.0 + float(np.sum(values <= observed_delta))) / (reps + 1.0)
    return pd.DataFrame([{
        "observed_delta_RMSE_full_minus_baseline": observed_delta,
        "mean_permuted_delta_RMSE": float(values.mean()),
        "permuted_delta_RMSE_sd": float(values.std(ddof=1)),
        "permutation_p_value_one_sided": p_value,
        "replications": reps,
        "interpretation_scope": "proxy ordering versus baseline; not incremental value beyond macro controls",
    }])


def run_lodo(rows: pd.DataFrame, specifications: dict[str, list[str]]) -> pd.DataFrame:
    """Refit after omitting each district from training; score the full test panel.

    This is a district-omission influence analysis. It is not a strict spatial
    holdout score restricted to the omitted district, so outputs are labelled accordingly.
    """
    months = rows["t"].to_numpy(dtype=int)
    districts = rows["district"].to_numpy(dtype=int)
    records = []
    compare_specs = ["baseline", "macro_control", "full_proxy"]
    test_mask = (months >= TEST_START_MONTH) & (months <= LAST_ANALYSIS_MONTH)
    for omitted in range(1, N_DISTRICTS + 1):
        train_mask = (months >= FIRST_ANALYSIS_MONTH) & (months <= TRAIN_END_MONTH) & (districts != omitted)
        for spec_name in compare_specs:
            result = fit_specification(rows, specifications[spec_name], train_mask, test_mask,
                                       f"{spec_name}_omit_district_{omitted}")
            records.append({"district_omitted_from_training": omitted, "specification": spec_name,
                            "alpha": result.alpha, **result.metrics})
    lodo = pd.DataFrame(records)
    wide = lodo.pivot(index="district_omitted_from_training", columns="specification", values="RMSE").reset_index()
    wide["full_vs_baseline_change_pct"] = 100 * (wide["full_proxy"] / wide["baseline"] - 1)
    wide["full_vs_macro_control_change_pct"] = 100 * (wide["full_proxy"] / wide["macro_control"] - 1)
    return wide


def _scenario_proxy_update(test_rows: pd.DataFrame, scenario: str, change: float) -> pd.DataFrame:
    out = test_rows.copy()
    hc = out["HC_s"].to_numpy(dtype=float)
    gamma = out["gamma_s"].to_numpy(dtype=float)
    eci = out["ECI"].to_numpy(dtype=float)
    ccf = out["CCF"].to_numpy(dtype=float)
    hed = out["HED"].to_numpy(dtype=float)
    sdi = out["SDI"].to_numpy(dtype=float)
    masrel = out["MASREL"].to_numpy(dtype=float)
    usdrel = out["USDREL"].to_numpy(dtype=float)

    if scenario in {"S1", "S4"}:
        gamma = gamma * (1.0 - change)
    if scenario in {"S2", "S4"}:
        hc = hc * (1.0 - change)
        eci = 0.6 * usdrel + 0.4 * hc
        hed = hc.copy()
    if scenario in {"S3", "S4"}:
        sdi = np.minimum(sdi * (1.0 + change), 1.0)
    ccf = masrel * eci * gamma
    out["ECI"] = eci
    out["CCF"] = ccf
    out["HED"] = hed
    out["SDI"] = sdi
    return out


def run_scenario_sensitivity(rows: pd.DataFrame, full_result: FittedResult,
                             feature_names: list[str], reps: int = 300,
                             seed: int = SEED) -> pd.DataFrame:
    test_mask = (rows["t"].to_numpy() >= TEST_START_MONTH) & (rows["t"].to_numpy() <= LAST_ANALYSIS_MONTH)
    test = rows.loc[test_mask].copy().reset_index(drop=True)
    base_x = test[feature_names].to_numpy(dtype=float)
    base_xs = (base_x - full_result.medians) / full_result.scales
    base_growth = full_result.model.predict(base_xs)
    base_price = np.maximum(np.expm1(test["anchor_log_price"].to_numpy() + base_growth), 0.0)
    records = []
    rng = np.random.default_rng(seed)
    months = np.sort(test["t"].unique())
    month_groups = {m: np.flatnonzero(test["t"].to_numpy() == m) for m in months}
    for change in [0.10, 0.20, 0.30, 0.50]:
        for scenario in ["S1", "S2", "S3", "S4"]:
            changed = _scenario_proxy_update(test, scenario, change)
            x_new = changed[feature_names].to_numpy(dtype=float)
            pred_growth = full_result.model.predict((x_new - full_result.medians) / full_result.scales)
            pred_price = np.maximum(np.expm1(test["anchor_log_price"].to_numpy() + pred_growth), 0.0)
            point = float((np.mean(pred_price) / np.mean(base_price) - 1.0) * 100.0)
            draws = []
            for _ in range(reps):
                selected = rng.choice(months, size=len(months), replace=True)
                ix = np.concatenate([month_groups[m] for m in selected])
                draws.append(float((np.mean(pred_price[ix]) / np.mean(base_price[ix]) - 1.0) * 100.0))
            records.append({
                "scenario": scenario, "parameter_change_pct": int(change * 100),
                "mean_predicted_price_change_pct": point,
                "month_bootstrap_ci_low_pct": float(np.percentile(draws, 2.5)),
                "month_bootstrap_ci_high_pct": float(np.percentile(draws, 97.5)),
                "interval_scope": "test-month resampling; conditional on fitted model and fixed parameters",
            })
    return pd.DataFrame(records)


def run_regional_scenario_sensitivity(rows: pd.DataFrame,
                                      specifications: dict[str, list[str]],
                                      reps: int = 300, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Estimate price-tier interactions and proxy scenario responses by training-period tercile."""
    training = rows.loc[rows["t"] <= TRAIN_END_MONTH]
    tier_mean = training.assign(log_price=np.log1p(training["price"])).groupby("district")["log_price"].mean()
    tier_code = pd.qcut(tier_mean.rank(method="first"), q=3, labels=[0, 1, 2]).astype(int)
    tier_map = tier_code.to_dict()
    data = rows.copy()
    data["price_tier"] = data["district"].map(tier_map).astype(int)
    for proxy in E_FEATURES:
        data[f"{proxy}_tier_low"] = data[proxy] * (data["price_tier"] == 0).astype(float)
        data[f"{proxy}_tier_high"] = data[proxy] * (data["price_tier"] == 2).astype(float)
    base = specifications["full_proxy"]
    interaction_features = base + [f"{proxy}_tier_{tier}" for proxy in E_FEATURES for tier in ["low", "high"]]
    months = data["t"].to_numpy(dtype=int)
    train_mask = (months >= FIRST_ANALYSIS_MONTH) & (months <= TRAIN_END_MONTH)
    test_mask = (months >= TEST_START_MONTH) & (months <= LAST_ANALYSIS_MONTH)
    interaction_result = fit_specification(data, interaction_features, train_mask, test_mask, "full_proxy_tier_interactions")

    test = data.loc[test_mask].copy().reset_index(drop=True)
    base_pred = interaction_result.predictions["predicted_price"].to_numpy(dtype=float)
    records = []
    rng = np.random.default_rng(seed)
    months_unique = np.sort(test["t"].unique())
    month_groups = {m: np.flatnonzero(test["t"].to_numpy() == m) for m in months_unique}
    for change in [0.10, 0.30, 0.50]:
        for scenario in ["S1", "S2", "S3"]:
            changed = _scenario_proxy_update(test, scenario, change)
            for proxy in E_FEATURES:
                changed[f"{proxy}_tier_low"] = changed[proxy] * (changed["price_tier"] == 0).astype(float)
                changed[f"{proxy}_tier_high"] = changed[proxy] * (changed["price_tier"] == 2).astype(float)
            x_new = changed[interaction_features].to_numpy(dtype=float)
            pred_growth = interaction_result.model.predict((x_new - interaction_result.medians) / interaction_result.scales)
            pred = np.maximum(np.expm1(changed["anchor_log_price"].to_numpy() + pred_growth), 0.0)
            for tier_id, tier_label in [(0, "low-price"), (1, "mid-price"), (2, "high-price")]:
                ix_tier = np.flatnonzero(test["price_tier"].to_numpy() == tier_id)
                point = float((np.mean(pred[ix_tier]) / np.mean(base_pred[ix_tier]) - 1.0) * 100.0)
                draws = []
                for _ in range(reps):
                    selected = rng.choice(months_unique, size=len(months_unique), replace=True)
                    ix_parts = [np.intersect1d(month_groups[m], ix_tier, assume_unique=False)
                               for m in selected]
                    ix_parts = [part for part in ix_parts if part.size]
                    if ix_parts:
                        ix = np.concatenate(ix_parts)
                        draws.append(float((np.mean(pred[ix]) / np.mean(base_pred[ix]) - 1.0) * 100.0))
                records.append({
                    "parameter_change_pct": int(change * 100), "scenario": scenario,
                    "district_tier": tier_label, "mean_price_change_pct": point,
                    "month_bootstrap_ci_low_pct": float(np.percentile(draws, 2.5)) if draws else np.nan,
                    "month_bootstrap_ci_high_pct": float(np.percentile(draws, 97.5)) if draws else np.nan,
                })
    membership = pd.DataFrame({"district": list(tier_map.keys()), "tier_code": list(tier_map.values())})
    membership["tier"] = membership["tier_code"].map({0: "low-price", 1: "mid-price", 2: "high-price"})
    membership["mean_log_price_training"] = membership["district"].map(tier_mean)
    interaction_metrics = {"alpha": interaction_result.alpha,
                           "n_features": len(interaction_features),
                           **interaction_result.metrics}
    return pd.DataFrame(records), membership.sort_values("district"), interaction_metrics


def run_shap_attribution(rows: pd.DataFrame, specifications: dict[str, list[str]],
                         ridge_result: FittedResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute exploratory SHAP attributions for Ridge, Random Forest, and XGBoost."""
    try:
        import shap
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("Install shap and xgboost to run the SHAP attribution stage.") from exc

    feature_names = specifications["full_proxy"]
    months = rows["t"].to_numpy(dtype=int)
    train = (months >= FIRST_ANALYSIS_MONTH) & (months <= TRAIN_END_MONTH)
    test = (months >= TEST_START_MONTH) & (months <= LAST_ANALYSIS_MONTH)
    x = rows[feature_names].to_numpy(dtype=float)
    x_train_scaled = (x[train] - ridge_result.medians) / ridge_result.scales
    x_test_scaled = (x[test] - ridge_result.medians) / ridge_result.scales
    x_train_raw = x[train]
    x_test_raw = x[test]
    y_train = rows.loc[train, "target_log_growth"].to_numpy(dtype=float)

    values_by_model: dict[str, np.ndarray] = {}
    ridge_explainer = shap.LinearExplainer(ridge_result.model, x_train_scaled)
    ridge_values = ridge_explainer.shap_values(x_test_scaled)
    values_by_model["Ridge"] = np.asarray(ridge_values)

    forest = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                   random_state=SEED, n_jobs=-1)
    forest.fit(x_train_raw, y_train)
    forest_values = shap.TreeExplainer(forest).shap_values(x_test_raw)
    values_by_model["RandomForest"] = np.asarray(forest_values)

    boosted = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
                           objective="reg:squarederror", random_state=SEED, n_jobs=-1)
    boosted.fit(x_train_raw, y_train)
    boosted_values = shap.TreeExplainer(boosted).shap_values(x_test_raw)
    values_by_model["XGBoost"] = np.asarray(boosted_values)

    group_map = {}
    for name in feature_names:
        if name in E_FEATURES:
            group_map[name] = "Environmental proxy"
        elif name in M_FEATURES or name in {"USD1", "dUSD3", "GOLD1", "BORS1", "TAV1", "MAS1"}:
            group_map[name] = "Macroeconomic series"
        else:
            group_map[name] = "Lagged price and momentum"

    importance_rows = []
    group_rows = []
    for model_name, shap_values in values_by_model.items():
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        total = float(mean_abs.sum())
        for feature, value in zip(feature_names, mean_abs):
            importance_rows.append({"model": model_name, "feature": feature,
                                    "group": group_map[feature], "mean_abs_SHAP": float(value),
                                    "share_total_abs_SHAP_pct": float(value / total * 100.0) if total else np.nan})
        for group in sorted(set(group_map.values())):
            group_value = float(sum(mean_abs[i] for i, feature in enumerate(feature_names)
                                    if group_map[feature] == group))
            group_rows.append({"model": model_name, "group": group,
                               "mean_abs_SHAP": group_value,
                               "share_total_abs_SHAP_pct": float(group_value / total * 100.0) if total else np.nan})
    return pd.DataFrame(importance_rows), pd.DataFrame(group_rows)


def run_nonlinear_models(rows: pd.DataFrame, specifications: dict[str, list[str]]) -> pd.DataFrame:
    """Fit the prespecified, untuned Random Forest and XGBoost comparators."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("Install xgboost to run the nonlinear comparator stage.") from exc
    months = rows["t"].to_numpy(dtype=int)
    train = (months >= FIRST_ANALYSIS_MONTH) & (months <= TRAIN_END_MONTH)
    test = (months >= TEST_START_MONTH) & (months <= LAST_ANALYSIS_MONTH)
    y = rows["target_log_growth"].to_numpy(dtype=float)
    records = []
    configurations = {
        "baseline": specifications["baseline"],
        "macro_control": specifications["macro_control"],
        "full_proxy": specifications["full_proxy"],
    }
    learners = {
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=8,
                         min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
                   objective="reg:squarederror", random_state=SEED, n_jobs=-1),
    }
    for learner_name, estimator in learners.items():
        for spec_name, feature_names in configurations.items():
            x = rows[feature_names].to_numpy(dtype=float)
            model = estimator.__class__(**estimator.get_params())
            model.fit(x[train], y[train])
            growth_hat = model.predict(x[test])
            test_rows = rows.loc[test]
            pred_price = np.maximum(np.expm1(test_rows["anchor_log_price"].to_numpy() + growth_hat), 0.0)
            metric = _metrics(test_rows["price"].to_numpy(), pred_price,
                              test_rows["target_log_growth"].to_numpy(), growth_hat)
            records.append({"learner": learner_name, "specification": spec_name,
                            "n_features": len(feature_names), **metric})
    return pd.DataFrame(records)


def run_residual_diagnostics(predictions: pd.DataFrame) -> dict[str, float]:
    residual = predictions["growth_residual"].to_numpy(dtype=float)
    fitted = predictions["predicted_growth"].to_numpy(dtype=float)
    monthly_mean = predictions.groupby("t")["growth_residual"].mean().sort_index().to_numpy()
    shapiro = stats.shapiro(residual)
    if len(monthly_mean) > 1 and np.std(monthly_mean[:-1]) > 0 and np.std(monthly_mean[1:]) > 0:
        lag1 = float(np.corrcoef(monthly_mean[:-1], monthly_mean[1:])[0, 1])
    else:
        lag1 = np.nan
    try:
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import het_breuschpagan
        design = sm.add_constant(fitted)
        lm_stat, lm_pvalue, _, _ = het_breuschpagan(residual, design)
    except ImportError:
        lm_stat, lm_pvalue = np.nan, np.nan
    return {
        "mean_growth_residual_observed_minus_predicted": float(np.mean(residual)),
        "mean_level_residual_observed_minus_predicted": float(predictions["price_residual"].mean()),
        "shapiro_W": float(shapiro.statistic),
        "shapiro_p": float(shapiro.pvalue),
        "breusch_pagan_LM_descriptive": float(lm_stat),
        "breusch_pagan_p_descriptive": float(lm_pvalue),
        "lag1_corr_monthly_mean_growth_residual": lag1,
    }


def _save_figures(output_dir: Path, rows: pd.DataFrame, cv: pd.DataFrame,
                  models: dict[str, FittedResult], boot: pd.DataFrame,
                  lodo: pd.DataFrame, scenarios: pd.DataFrame,
                  regional: pd.DataFrame, residual_stats: dict[str, float],
                  shap_groups: pd.DataFrame | None = None) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name in ["baseline", "macro_control", "full_proxy", "rank_macro_control", "rank_macro_plus_ccf"]:
        profile = models[name].alpha_profile
        ax.plot(profile["alpha"], profile["mean_inner_RMSElog"], label=name.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xlabel("Ridge penalty (alpha)")
    ax.set_ylabel("Mean inner-validation RMSE (log growth)")
    ax.set_title("Nested expanding-window alpha selection")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_02_alpha_selection.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, group in cv.groupby("specification"):
        ax.plot(group["fold"], group["RMSE"], marker="o", label=name.replace("_", " "))
    ax.set_xlabel("Outer rolling-origin fold")
    ax.set_ylabel("RMSE (nominal price units)")
    ax.set_title("Rolling-origin validation")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_03_rolling_validation.png", dpi=220)
    plt.close(fig)

    pred = models["full_proxy"].predictions.sort_values(["t", "district"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(pred["price"], pred["predicted_price"], s=16, alpha=.65)
    lo = min(pred["price"].min(), pred["predicted_price"].min())
    hi = max(pred["price"].max(), pred["predicted_price"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Observed price")
    ax.set_ylabel("Predicted price")
    ax.set_title("Untouched chronological test set: full proxy model")
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_04_test_predictions.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if len(boot):
        yloc = np.arange(len(boot))
        point = boot["median_pct"].to_numpy()
        low = point - boot["ci_low_pct"].to_numpy()
        high = boot["ci_high_pct"].to_numpy() - point
        ax.errorbar(point, yloc, xerr=np.vstack([low, high]), fmt="o", capsize=3)
        ax.axvline(0, linestyle="--")
        ax.set_yticks(yloc, boot["comparison"])
    ax.set_xlabel("RMSE change (%) of comparison model vs reference")
    ax.set_title("Paired bootstrap uncertainty")
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_05_bootstrap.png", dpi=220)
    plt.close(fig)

    rank_model = models["rank_macro_plus_ccf"]
    coef = pd.DataFrame({"feature": rank_model.feature_names,
                         "coefficient": rank_model.model.coef_})
    coef = coef.reindex(coef["coefficient"].abs().sort_values(ascending=True).index)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh(coef["feature"], coef["coefficient"])
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("Standardised Ridge coefficient")
    ax.set_title("Rank-adjusted macro control + CCF model")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_06_rank_adjusted_coefficients.png", dpi=220)
    plt.close(fig)

    if shap_groups is not None and len(shap_groups):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        pivot = shap_groups.pivot(index="group", columns="model", values="share_total_abs_SHAP_pct")
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Share of total mean absolute SHAP (%)")
        ax.set_title("Exploratory SHAP attribution by feature group")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="Model")
        fig.tight_layout()
        fig.savefig(fig_dir / "figure_07_shap_group_attribution.png", dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(lodo):
        ax.plot(lodo["district_omitted_from_training"], lodo["full_vs_baseline_change_pct"], marker="o", label="Full vs baseline")
        ax.plot(lodo["district_omitted_from_training"], lodo["full_vs_macro_control_change_pct"], marker="s", label="Full vs macro control")
        ax.axhline(0, linestyle="--")
        ax.set_xticks(range(1, N_DISTRICTS + 1))
        ax.set_xlabel("District omitted from training; full test panel scored")
        ax.set_ylabel("Test RMSE change (%)")
        ax.legend()
    ax.set_title("District-omission influence diagnostic")
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_09a_lodo.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if len(regional):
        sub = regional[regional["parameter_change_pct"] == 30]
        labels = [f"{r.scenario} | {r.district_tier}" for r in sub.itertuples()]
        ax.barh(labels, sub["mean_price_change_pct"])
        ax.axvline(0, linestyle="--")
    ax.set_xlabel("Predicted price change relative to scenario baseline (%)")
    ax.set_title("Regional scenario sensitivity (30% perturbation)")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_09b_regional_sensitivity.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for scenario, group in scenarios.groupby("scenario"):
        ax.plot(group["parameter_change_pct"], group["mean_predicted_price_change_pct"], marker="o", label=scenario)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Parameter perturbation (%)")
    ax.set_ylabel("Mean predicted-price response (%)")
    ax.set_title("Proxy scenario sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_08_scenario_sensitivity.png", dpi=220)
    plt.close(fig)

    residual_predictions = models["rank_macro_plus_ccf"].predictions.sort_values(["t", "district"])
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    axs[0, 0].scatter(residual_predictions["predicted_growth"], residual_predictions["growth_residual"], s=14, alpha=.6)
    axs[0, 0].axhline(0, linestyle="--")
    axs[0, 0].set_xlabel("Predicted log growth")
    axs[0, 0].set_ylabel("Growth residual")
    axs[0, 0].set_title("Residuals vs fitted: rank-adjusted macro + CCF")
    axs[0, 1].hist(residual_predictions["growth_residual"], bins=18)
    axs[0, 1].set_title("Residual distribution")
    stats.probplot(residual_predictions["growth_residual"], dist="norm", plot=axs[1, 0])
    axs[1, 0].set_title("Normal Q-Q")
    axs[1, 1].scatter(np.sqrt(np.abs(residual_predictions["growth_residual"])), residual_predictions["predicted_growth"], s=14, alpha=.6)
    axs[1, 1].set_xlabel("Sqrt absolute residual")
    axs[1, 1].set_ylabel("Predicted log growth")
    axs[1, 1].set_title("Scale-location")
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_10_residual_diagnostics.png", dpi=220)
    plt.close(fig)


def run_pipeline(data_path: str | Path, output_dir: str | Path,
                 bootstrap_reps: int = 2000, region_reps: int = 300,
                 run_nonlinear: bool = True,
                 run_shap: bool = True) -> dict[str, Any]:
    """Execute the reproducibility workflow and write all outputs."""
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    study = load_study_data(data_path)
    frame = build_analysis_frame(study)
    rows = frame.rows
    months = rows["t"].to_numpy(dtype=int)
    test_mask = (months >= TEST_START_MONTH) & (months <= LAST_ANALYSIS_MONTH)

    pd.DataFrame([study.audit]).to_json(output_dir / "data_audit.json", orient="records", indent=2)
    study.decimal_flags.to_csv(tables_dir / "decimal_shift_diagnostics.csv", index=False)
    rows.to_csv(output_dir / "analysis_panel.csv", index=False)

    proxy_summary = pd.DataFrame([
        {"proxy": col, "mean": rows.drop_duplicates("t")[col].mean(),
         "sd": rows.drop_duplicates("t")[col].std(ddof=1),
         "min": rows.drop_duplicates("t")[col].min(),
         "median": rows.drop_duplicates("t")[col].median(),
         "max": rows.drop_duplicates("t")[col].max()}
        for col in E_FEATURES
    ])
    proxy_summary.to_csv(tables_dir / "proxy_descriptive_statistics.csv", index=False)

    cv = run_rolling_validation(rows, frame.specifications)
    cv.to_csv(tables_dir / "rolling_origin_validation.csv", index=False)

    models = run_final_models(rows, frame.specifications)
    metric_rows = []
    for name, fitted in models.items():
        metric_rows.append({"specification": name, "n_features": len(fitted.feature_names),
                            "alpha": fitted.alpha, **fitted.metrics})
        fitted.alpha_profile.to_csv(tables_dir / f"alpha_profile_{name}.csv", index=False)
        fitted.predictions.to_csv(tables_dir / f"test_predictions_{name}.csv", index=False)
    persistence_metric, persistence_pred = persistence_predictions(rows, test_mask)
    metric_rows.append({"specification": "persistence", "n_features": 0, "alpha": np.nan, **persistence_metric})
    persistence_pred.to_csv(tables_dir / "test_predictions_persistence.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(tables_dir / "out_of_time_test_metrics.csv", index=False)

    test_rows = rows.loc[test_mask]
    pred_lookup = {k: v.predictions["predicted_price"].to_numpy() for k, v in models.items()}
    actual = test_rows["price"].to_numpy(dtype=float)
    test_months = test_rows["t"].to_numpy(dtype=int)
    districts = test_rows["district"].to_numpy(dtype=int)
    comparisons = [
        ("full_proxy_vs_baseline", "baseline", "full_proxy"),
        ("full_proxy_vs_macro_control", "macro_control", "full_proxy"),
        ("rank_ccf_vs_rank_control", "rank_macro_control", "rank_macro_plus_ccf"),
    ]
    boot_rows = []
    boot_detail = {}
    for label, ref_name, comparison_name in comparisons:
        summary = paired_bootstrap(actual, pred_lookup[ref_name], pred_lookup[comparison_name],
                                   test_months, districts, reps=bootstrap_reps,
                                   seed=SEED + len(boot_rows))
        boot_detail[label] = summary
        for method, values in summary.items():
            boot_rows.append({"comparison": label, "bootstrap_scheme": method, **values})
    bootstrap = pd.DataFrame(boot_rows)
    bootstrap.to_csv(tables_dir / "paired_bootstrap_comparisons.csv", index=False)

    permutation = run_month_shuffle_permutation(
        rows, models["baseline"], models["full_proxy"],
        frame.specifications["baseline"], frame.specifications["full_proxy"],
        reps=500, seed=SEED,
    )
    permutation.to_csv(tables_dir / "month_shuffle_permutation.csv", index=False)

    lodo = run_lodo(rows, frame.specifications)
    lodo.to_csv(tables_dir / "leave_one_district_out.csv", index=False)

    scenarios = run_scenario_sensitivity(rows, models["full_proxy"],
                                         frame.specifications["full_proxy"], reps=region_reps)
    scenarios.to_csv(tables_dir / "scenario_sensitivity.csv", index=False)
    regional, tier_membership, tier_model_metrics = run_regional_scenario_sensitivity(
        rows, frame.specifications, reps=region_reps
    )
    regional.to_csv(tables_dir / "regional_scenario_sensitivity.csv", index=False)
    tier_membership.to_csv(tables_dir / "district_price_tiers.csv", index=False)
    pd.DataFrame([tier_model_metrics]).to_csv(tables_dir / "tier_interaction_model_metrics.csv", index=False)

    rank_diagnostics = []
    for name in ["rank_macro_control", "rank_macro_plus_ccf"]:
        x = rows[frame.specifications[name]].to_numpy(dtype=float)
        train = (months >= FIRST_ANALYSIS_MONTH) & (months <= TRAIN_END_MONTH)
        x_scaled, _, _, _ = _scale(x[train], x[test_mask])
        augmented = np.column_stack([np.ones(x_scaled.shape[0]), x_scaled])
        rank_diagnostics.append({
            "specification": name,
            "n_columns_including_intercept": int(augmented.shape[1]),
            "matrix_rank": int(np.linalg.matrix_rank(augmented)),
            "condition_number": float(np.linalg.cond(augmented)),
        })
    pd.DataFrame(rank_diagnostics).to_csv(tables_dir / "rank_and_condition_diagnostics.csv", index=False)

    # Residual diagnostics correspond to the controlled rank-reduced model used
    # for the incremental CCF comparison in the manuscript.
    residual = run_residual_diagnostics(models["rank_macro_plus_ccf"].predictions)
    with open(output_dir / "residual_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(residual, f, indent=2)

    nonlinear = pd.DataFrame()
    shap_importance = pd.DataFrame()
    shap_groups = pd.DataFrame()
    if run_nonlinear:
        nonlinear = run_nonlinear_models(rows, frame.specifications)
        nonlinear.to_csv(tables_dir / "nonlinear_model_comparison.csv", index=False)
        if run_shap:
            shap_importance, shap_groups = run_shap_attribution(
                rows, frame.specifications, models["full_proxy"]
            )
            shap_importance.to_csv(tables_dir / "shap_feature_attribution.csv", index=False)
            shap_groups.to_csv(tables_dir / "shap_group_attribution.csv", index=False)

    _save_figures(output_dir, rows, cv, models, bootstrap, lodo, scenarios, regional,
                  residual, shap_groups if not shap_groups.empty else None)

    summary = {
        "seed": SEED,
        "n_analysis_rows": int(len(rows)),
        "n_train_rows": int(((months >= FIRST_ANALYSIS_MONTH) & (months <= TRAIN_END_MONTH)).sum()),
        "n_test_rows": int(test_mask.sum()),
        "train_months": [FIRST_ANALYSIS_MONTH, TRAIN_END_MONTH],
        "test_months": [TEST_START_MONTH, LAST_ANALYSIS_MONTH],
        "model_metrics": metrics.to_dict(orient="records"),
        "bootstrap": boot_detail,
        "residual_diagnostics": residual,
        "residual_diagnostic_specification": "rank_macro_plus_ccf",
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    package_names = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
                     "openpyxl", "statsmodels", "xgboost", "shap"]
    manifest = {
        "python": platform.python_version(),
        "seed": SEED,
        "input_file": str(Path(data_path).name),
        "input_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "packages": {name: package_metadata.version(name) for name in package_names},
        "bootstrap_repetitions": int(bootstrap_reps),
        "regional_scenario_repetitions": int(region_reps),
        "nonlinear_models_run": bool(run_nonlinear),
        "shap_run": bool(run_shap and run_nonlinear),
        "residual_diagnostic_specification": "rank_macro_plus_ccf",
    }
    with open(output_dir / "reproducibility_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return {"study": study, "frame": frame, "cv": cv, "models": models,
            "metrics": metrics, "bootstrap": bootstrap, "permutation": permutation, "lodo": lodo,
            "scenarios": scenarios, "regional_scenarios": regional,
            "tier_membership": tier_membership, "tier_model_metrics": tier_model_metrics,
            "residual_diagnostics": residual, "nonlinear": nonlinear,
            "shap_importance": shap_importance, "shap_groups": shap_groups}