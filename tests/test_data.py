import json

import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

from src.data import clean_data, load_raw_data, prepare_data


def test_load_raw_data_loads_csv(synthetic_telco_csv, tmp_path):
    dataset_info_path = tmp_path / "dataset_info.json"

    df = load_raw_data(synthetic_telco_csv, dataset_info_path)

    assert not df.empty
    assert "Churn" in df.columns
    assert dataset_info_path.exists()

    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    assert dataset_info["row_count"] == len(df)
    assert dataset_info["column_count"] == len(df.columns)
    assert set(dataset_info["churn_distribution"]) == {"No", "Yes"}


def test_clean_data_does_not_break(synthetic_telco_csv, tmp_path):
    df = load_raw_data(synthetic_telco_csv, tmp_path / "dataset_info.json")

    cleaned_df = clean_data(df)

    assert cleaned_df.shape == df.shape
    assert cleaned_df["TotalCharges"].isna().sum() == 1
    assert df.loc[0, "TotalCharges"] == "bad-value"


def test_clean_data_converts_total_charges_to_numeric(synthetic_telco_csv, tmp_path):
    df = load_raw_data(synthetic_telco_csv, tmp_path / "dataset_info.json")

    cleaned_df = clean_data(df)

    assert is_numeric_dtype(cleaned_df["TotalCharges"])
    assert cleaned_df["TotalCharges"].isna().sum() == 1


def test_prepare_data_returns_features_and_target(synthetic_telco_csv, tmp_path):
    df = clean_data(load_raw_data(synthetic_telco_csv, tmp_path / "dataset_info.json"))

    X, y = prepare_data(df)

    assert len(X) == len(y)
    assert "Churn" not in X.columns
    assert "customerID" not in X.columns


def test_prepare_data_target_has_two_classes(synthetic_telco_csv, tmp_path):
    df = clean_data(load_raw_data(synthetic_telco_csv, tmp_path / "dataset_info.json"))

    _, y = prepare_data(df)

    assert y.nunique() == 2


@pytest.mark.parametrize("labels", [["No", None], ["Yes", "Maybe"], ["No", "No"]])
def test_prepare_data_rejects_invalid_target(labels):
    with pytest.raises(ValueError, match="Churn|churn"):
        prepare_data(pd.DataFrame({"tenure": [1, 2], "Churn": labels}))
