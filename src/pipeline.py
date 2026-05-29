"""Baseline model pipeline for Telco Customer Churn."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing() -> ColumnTransformer:
    """Build preprocessing for numeric and categorical Telco features."""
    numeric_features = make_column_selector(dtype_include=np.number)
    categorical_features = make_column_selector(dtype_exclude=np.number)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_baseline_pipeline() -> Pipeline:
    """Build a baseline preprocessing and classification pipeline."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
