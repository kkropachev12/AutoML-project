"""Train Telco Customer Churn models."""

from __future__ import annotations

import argparse
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

from src.automl import train_automl_model
from src.data import DEFAULT_DATA_PATH, clean_data, load_raw_data, prepare_data
from src.pipeline import build_baseline_pipeline
from src.visualize import DEFAULT_PLOTS_DIR, save_baseline_plots


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
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
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
    y_pred = model.predict(X_test)
    y_test_binary = _target_to_binary(y_test)
    y_pred_binary = _target_to_binary(y_pred)
    y_score = _positive_class_probability(model, X_test)
    save_baseline_plots(cleaned_df, y_test_binary, y_pred_binary, y_score, plots_dir)

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


def train_models(
    data_path: str | Path = DEFAULT_DATA_PATH,
    automl_trials: int = 20,
) -> dict[str, Any]:
    """Train baseline and AutoML models."""
    baseline_metrics = train_baseline_model(data_path=data_path)
    automl_result = train_automl_model(data_path=data_path, n_trials=automl_trials)

    return {
        "baseline_metrics": baseline_metrics,
        "automl": {
            "best_params": automl_result.best_params,
            "best_metrics": automl_result.best_metrics,
            "validation_roc_auc": automl_result.validation_roc_auc,
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train Telco churn models.")
    parser.add_argument(
        "--automl-trials",
        type=int,
        default=20,
        help="Number of Optuna trials for AutoML.",
    )
    return parser.parse_args()


def main() -> None:
    """Run baseline and AutoML training from the command line."""
    args = parse_args()
    result = train_models(automl_trials=args.automl_trials)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
