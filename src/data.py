"""Utilities for loading and preparing the Telco Customer Churn dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.artifacts import save_json
from src.config import DEFAULT_DATA_PATH, DEFAULT_DATASET_INFO_PATH
from src.metrics import target_to_binary


def load_raw_data(
    path: str | Path = DEFAULT_DATA_PATH,
    dataset_info_path: str | Path = DEFAULT_DATASET_INFO_PATH,
) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV file."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    save_dataset_info(df, dataset_info_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric values without learning statistics from the full dataset."""
    cleaned_df = df.copy()

    if "TotalCharges" in cleaned_df.columns:
        total_charges = pd.to_numeric(cleaned_df["TotalCharges"], errors="coerce")
        cleaned_df["TotalCharges"] = total_charges

    return cleaned_df


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split cleaned data into features and target."""
    if "Churn" not in df.columns:
        raise ValueError("Column 'Churn' is required to prepare X and y.")

    y = df["Churn"].copy()
    target_to_binary(y)
    if y.nunique() != 2:
        raise ValueError("Training requires both churn classes.")
    X = df.drop(columns=["Churn"]).drop(columns=["customerID"], errors="ignore")

    return X, y


def save_dataset_info(
    df: pd.DataFrame,
    path: str | Path = DEFAULT_DATASET_INFO_PATH,
) -> dict[str, object]:
    """Save a short dataset summary to a JSON artifact."""
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

    save_json(dataset_info, path)

    return dataset_info
