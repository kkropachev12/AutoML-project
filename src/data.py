"""Utilities for loading and preparing the Telco Customer Churn dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "telco_customer_churn.csv"
DEFAULT_DATASET_INFO_PATH = PROJECT_ROOT / "artifacts" / "dataset_info.json"


def load_raw_data(
    path: str | Path = DEFAULT_DATA_PATH,
    dataset_info_path: str | Path = DEFAULT_DATASET_INFO_PATH,
) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV file."""
    data_path = Path(path)
    df = pd.read_csv(data_path)
    save_dataset_info(df, dataset_info_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco Customer Churn data."""
    cleaned_df = df.copy()

    if "TotalCharges" in cleaned_df.columns:
        total_charges = pd.to_numeric(cleaned_df["TotalCharges"], errors="coerce")
        cleaned_df["TotalCharges"] = total_charges.fillna(total_charges.median())

    return cleaned_df


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split cleaned data into features and target."""
    if "Churn" not in df.columns:
        raise ValueError("Column 'Churn' is required to prepare X and y.")

    y = df["Churn"].copy()
    X = df.drop(columns=["Churn"]).drop(columns=["customerID"], errors="ignore")

    return X, y


def save_dataset_info(
    df: pd.DataFrame,
    path: str | Path = DEFAULT_DATASET_INFO_PATH,
) -> dict[str, object]:
    """Save a short dataset summary to a JSON artifact."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    churn_distribution = {}
    if "Churn" in df.columns:
        churn_distribution = {
            str(label): int(count)
            for label, count in df["Churn"].value_counts(dropna=False).items()
        }

    dataset_info = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "churn_distribution": churn_distribution,
    }

    artifact_path.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return dataset_info
