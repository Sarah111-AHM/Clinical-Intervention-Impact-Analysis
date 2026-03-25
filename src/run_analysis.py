"""
Clinical Intervention Analysis - Main Runner
============================================
Run this script to execute the full analysis pipeline:
  1. Data preprocessing & validation
  2. Descriptive statistics & visualisations
  3. Statistical tests
  4. Machine learning models
  5. Save all outputs to /reports/
"""

import sys
from pathlib import Path

# Allow running from /src or from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_preprocessing import full_pipeline
from descriptive_analysis import (
    compute_descriptive_stats,
    compute_group_stats,
    run_statistical_tests,
    generate_all_plots,
)
from ml_models import run_full_ml_pipeline

DATA_PATH    = ROOT / "data" / "competition_dataset.csv"
REPORTS_PATH = ROOT / "reports"


def main():
    print("=" * 65)
    print("   CLINICAL INTERVENTION ANALYSIS — FULL PIPELINE")
    print("=" * 65)

    # ── 1. Load & preprocess ──────────────────────────────────────────
    print("\n[1/4] Data Preprocessing")
    df_raw, df_clean, df_feat, X, y_comp, y_stay, val_report = full_pipeline(
        str(DATA_PATH)
    )
    print(f"  Missing values : {sum(val_report['missing_values'].values())}")
    print(f"  Duplicates     : {val_report['duplicates']}")

    # ── 2. Descriptive stats ──────────────────────────────────────────
    print("\n[2/4] Descriptive Statistics")
    stats_tbl = compute_descriptive_stats(df_feat)
    print(stats_tbl.to_string())

    print("\n  Group-level statistics:")
    grp_stats = compute_group_stats(df_feat)
    print(grp_stats.to_string(index=False))

    # ── 3. Statistical tests ──────────────────────────────────────────
    print("\n[3/4] Statistical Tests")
    tests_tbl = run_statistical_tests(df_feat)
    print(tests_tbl.to_string(index=False))

    # ── 3b. EDA visualisations ─────────────────────────────────────────
    print("\n  Generating EDA plots...")
    plot_paths = generate_all_plots(df_feat, str(REPORTS_PATH))
    for name, path in plot_paths.items():
        print(f"    {name:<30} → {path}")

    # ── 4. Machine Learning ────────────────────────────────────────────
    print("\n[4/4] Machine Learning Models")
    ml_paths = run_full_ml_pipeline(X, y_comp, y_stay, str(REPORTS_PATH))
    for name, path in ml_paths.items():
        print(f"    {name:<35} → {path}")

    print("\n" + "=" * 65)
    print("  ✔ Analysis complete!")
    print(f"  Reports saved to: {REPORTS_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
