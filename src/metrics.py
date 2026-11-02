"""Shared binary classification metrics for baseline and AutoML."""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def target_to_binary(y: Any) -> pd.Series:
    """Convert Telco labels or binary values to zero and one."""
    series = pd.Series(y).reset_index(drop=True)
    if series.isna().any():
        raise ValueError("Churn must not contain missing values.")
    labels = set(series.unique())
    if labels.issubset({"No", "Yes"}):
        return series.map({"No": 0, "Yes": 1}).astype(int)
    if labels.issubset({0, 1}):
        return series.astype(int)
    raise ValueError("Churn must contain Yes/No or 0/1 values.")


def positive_class_probability(model: Pipeline, X: pd.DataFrame) -> NDArray[np.float64]:
    """Return the churn probability using the classifier's actual class order."""
    classes = list(model.classes_)
    if "Yes" in classes:
        positive_class_index = classes.index("Yes")
    elif 1 in classes:
        positive_class_index = classes.index(1)
    else:
        raise ValueError("Model has no positive churn class (Yes or 1).")
    return model.predict_proba(X)[:, positive_class_index]


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Calculate the same five metrics for every candidate model."""
    y_test_binary = target_to_binary(y_test)
    y_pred_binary = target_to_binary(model.predict(X_test))
    y_score = positive_class_probability(model, X_test)
    return {
        "accuracy": float(accuracy_score(y_test_binary, y_pred_binary)),
        "precision": float(
            precision_score(y_test_binary, y_pred_binary, zero_division=0)
        ),
        "recall": float(recall_score(y_test_binary, y_pred_binary, zero_division=0)),
        "f1_score": float(f1_score(y_test_binary, y_pred_binary, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_binary, y_score)),
    }
