"""Train the baseline Telco Customer Churn model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data import DEFAULT_DATA_PATH, clean_data, load_raw_data, prepare_data
from src.pipeline import build_baseline_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "baseline_model.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "baseline_metrics.json"


def _target_to_binary(y: Any) -> pd.Series:
    """Convert common Telco target labels to binary values for metrics."""
    series = pd.Series(y).reset_index(drop=True)
    if set(series.dropna().unique()).issubset({"No", "Yes"}):
        return series.map({"No": 0, "Yes": 1}).astype(int)
    return series.astype(int)


def _positive_class_probability(model, X: pd.DataFrame) -> list[float]:
    """Return predicted probability for the positive churn class."""
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    if "Yes" in classes:
        positive_class_index = classes.index("Yes")
    elif 1 in classes:
        positive_class_index = classes.index(1)
    else:
        positive_class_index = len(classes) - 1

    return probabilities[:, positive_class_index]


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calculate baseline classification metrics."""
    y_pred = model.predict(X_test)
    y_test_binary = _target_to_binary(y_test)
    y_pred_binary = _target_to_binary(y_pred)
    y_score = _positive_class_probability(model, X_test)

    return {
        "accuracy": float(accuracy_score(y_test_binary, y_pred_binary)),
        "precision": float(
            precision_score(y_test_binary, y_pred_binary, zero_division=0)
        ),
        "recall": float(recall_score(y_test_binary, y_pred_binary, zero_division=0)),
        "f1_score": float(f1_score(y_test_binary, y_pred_binary, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_binary, y_score)),
    }


def train_baseline_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Train and save the baseline churn prediction model."""
    raw_df = load_raw_data(data_path)
    cleaned_df = clean_data(raw_df)
    X, y = prepare_data(cleaned_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_baseline_pipeline()
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    model_artifact_path = Path(model_path)
    model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_artifact_path)

    metrics_artifact_path = Path(metrics_path)
    metrics_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_artifact_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metrics


def main() -> None:
    """Run baseline training from the command line."""
    metrics = train_baseline_model()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
