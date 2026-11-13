"""Command-line entry point for the reproducibility workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tehran_housing.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the empirical results for the Tehran housing-price study."
    )
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data" / "raw" / "Home_4.xlsx",
        help="Path to the supplied Excel workbook."
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs",
        help="Directory in which tables, diagnostics, and figures are saved."
    )
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--region-reps", type=int, default=300)
    parser.add_argument(
        "--skip-nonlinear", action="store_true",
        help="Skip the prespecified Random Forest and XGBoost comparison."
    )
    parser.add_argument(
        "--skip-shap", action="store_true",
        help="Skip SHAP attribution calculations."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 100 or args.region_reps < 100:
        raise SystemExit("Use at least 100 replications; manuscript settings are 2000 and 300.")
    results = run_pipeline(
        data_path=args.data,
        output_dir=args.output,
        bootstrap_reps=args.bootstrap_reps,
        region_reps=args.region_reps,
        run_nonlinear=not args.skip_nonlinear,
        run_shap=not args.skip_shap,
    )
    print("Analysis completed.")
    print(f"Rows: {len(results['frame'].rows)}")
    print(f"Output directory: {args.output.resolve()}")
    print(results["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
