from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.data import clean_data, load_raw_data, prepare_data


def _write_sample_telco_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "customerID": ["0001-A", "0002-B", "0003-C", "0004-D"],
            "gender": ["Female", "Male", "Male", "Female"],
            "tenure": [1, 34, 2, 45],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
            "TotalCharges": ["29.85", "1889.5", "bad-value", " "],
            "Churn": ["No", "No", "Yes", "Yes"],
        }
    )
    df.to_csv(path, index=False)


def test_load_raw_data_loads_csv(tmp_path):
    csv_path = tmp_path / "telco_customer_churn.csv"
    _write_sample_telco_csv(csv_path)

    df = load_raw_data(csv_path, tmp_path / "dataset_info.json")

    assert not df.empty
    assert "Churn" in df.columns


def test_clean_data_does_not_break(tmp_path):
    csv_path = tmp_path / "telco_customer_churn.csv"
    _write_sample_telco_csv(csv_path)
    df = load_raw_data(csv_path, tmp_path / "dataset_info.json")

    cleaned_df = clean_data(df)

    assert cleaned_df.shape == df.shape
    assert cleaned_df.isna().sum().sum() == 0


def test_clean_data_converts_total_charges_to_numeric(tmp_path):
    csv_path = tmp_path / "telco_customer_churn.csv"
    _write_sample_telco_csv(csv_path)
    df = load_raw_data(csv_path, tmp_path / "dataset_info.json")

    cleaned_df = clean_data(df)

    assert is_numeric_dtype(cleaned_df["TotalCharges"])


def test_prepare_data_returns_features_and_target(tmp_path):
    csv_path = tmp_path / "telco_customer_churn.csv"
    _write_sample_telco_csv(csv_path)
    df = clean_data(load_raw_data(csv_path, tmp_path / "dataset_info.json"))

    X, y = prepare_data(df)

    assert len(X) == len(y)
    assert "Churn" not in X.columns
    assert "customerID" not in X.columns


def test_prepare_data_target_has_two_classes(tmp_path):
    csv_path = tmp_path / "telco_customer_churn.csv"
    _write_sample_telco_csv(csv_path)
    df = clean_data(load_raw_data(csv_path, tmp_path / "dataset_info.json"))

    _, y = prepare_data(df)

    assert y.nunique() == 2
