"""
Clinical Intervention Analysis - Data Preprocessing Module
===========================================================
Handles data loading, cleaning, validation, and feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    """Load clinical dataset from CSV file."""
    df = pd.read_csv(filepath)
    print(f"✔ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """Run data quality checks and return a report."""
    report = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates": df.duplicated().sum(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the clinical dataset:
      - Strip string whitespace
      - Standardise categorical capitalisation
      - Remove duplicate rows
      - Clip numeric outliers beyond 3 IQR fences
    """
    df = df.copy()

    # Strip whitespace on string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Standardise group / sex / complication labels
    if "group" in df.columns:
        df["group"] = df["group"].str.capitalize()
    if "sex" in df.columns:
        df["sex"] = df["sex"].str.capitalize()
    if "complication" in df.columns:
        df["complication"] = df["complication"].str.capitalize()

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  ↳ Removed {removed} duplicate row(s).")

    # Clip numeric outliers (IQR × 3 rule)
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        if clipped:
            df[col] = df[col].clip(lower, upper)
            print(f"  ↳ Clipped {clipped} outlier(s) in '{col}'.")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features for modelling:
      - age_group  : binned age bracket
      - stay_long  : binary flag for hospital stay > 7 days
      - sex_enc    : numeric encoding of sex (Male=1, Female=0)
      - group_enc  : numeric encoding of group (Intervention=1, Control=0)
    """
    df = df.copy()

    # Age brackets
    bins = [0, 40, 55, 65, 120]
    labels = ["<40", "40-55", "55-65", "65+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    # Long-stay binary target
    df["stay_long"] = (df["hospital_stay"] > 7).astype(int)

    # Label encodings
    df["sex_enc"] = (df["sex"] == "Male").astype(int)
    df["group_enc"] = (df["group"] == "Intervention").astype(int)

    # Binary complication target
    df["complication_bin"] = (df["complication"] == "Yes").astype(int)

    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    Return (X, y_complication, y_stay) ready for sklearn.

    X features: age, sex_enc, group_enc, hospital_stay
    y_complication : binary complication label
    y_stay         : binary long-stay label
    """
    feature_cols = ["age", "sex_enc", "group_enc", "hospital_stay"]
    X = df[feature_cols]
    y_comp = df["complication_bin"]
    y_stay = df["stay_long"]
    return X, y_comp, y_stay


def full_pipeline(filepath: str):
    """End-to-end preprocessing pipeline."""
    df_raw = load_data(filepath)
    report = validate_data(df_raw)
    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean)
    X, y_comp, y_stay = get_feature_matrix(df_feat)
    return df_raw, df_clean, df_feat, X, y_comp, y_stay, report


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    path = base / "data" / "competition_dataset.csv"
    _, _, df, X, yc, ys, rpt = full_pipeline(str(path))
    print("\n--- Feature matrix head ---")
    print(X.head())
    print("\n--- Validation report ---")
    for k, v in rpt.items():
        print(f"  {k}: {v}")
