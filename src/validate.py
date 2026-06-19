"""
Validation stage: reads raw churn data, cleans it, asserts data quality
invariants, and writes a clean version to data/processed/.

This stage exists to fail loudly and early if the data violates assumptions
the rest of the pipeline depends on -- rather than letting bad data
silently flow into training.
"""

import sys
import pandas as pd

RAW_PATH = "data/raw/telco_churn.csv"
PROCESSED_PATH = "data/processed/telco_churn_clean.csv"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges has blank strings for some brand-new customers -- coerce + drop
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # customerID is an identifier, not a feature
    df = df.drop(columns=["customerID"])

    # Encode target: Yes/No -> 1/0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def validate(df: pd.DataFrame) -> None:
    """Raise AssertionError with a clear message if any invariant is violated."""

    errors = []

    if df.isnull().any().any():
        bad_cols = df.columns[df.isnull().any()].tolist()
        errors.append(f"Null values found in columns: {bad_cols}")

    if not df["Churn"].isin([0, 1]).all():
        errors.append("Churn column contains values outside {0, 1}")

    if (df["tenure"] < 0).any():
        errors.append("Negative tenure values found")

    if (df["MonthlyCharges"] < 0).any():
        errors.append("Negative MonthlyCharges values found")

    if (df["TotalCharges"] < 0).any():
        errors.append("Negative TotalCharges values found")

    if len(df) < 1000:
        errors.append(f"Row count suspiciously low: {len(df)} (expected ~7000)")

    if errors:
        message = "Data validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise AssertionError(message)


def main():
    print(f"Reading raw data from {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"Raw shape: {df.shape}")

    df = clean(df)
    print(f"Cleaned shape: {df.shape}")

    try:
        validate(df)
    except AssertionError as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)

    print("All validation checks passed.")

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Wrote clean data to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()