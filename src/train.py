"""Train Telco Customer Churn models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from src.artifacts import save_json, save_model
from src.automl import train_automl_model
from src.config import (
    AUTOML_TRIALS,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data import (
    DEFAULT_DATA_PATH,
    DEFAULT_DATASET_INFO_PATH,
    clean_data,
    load_raw_data,
    prepare_data,
)
from src.metrics import evaluate_model, positive_class_probability, target_to_binary
from src.pipeline import build_baseline_pipeline
from src.visualize import DEFAULT_PLOTS_DIR, save_baseline_plots


def train_baseline_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
    dataset_info_path: str | Path = DEFAULT_DATASET_INFO_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Train and save the baseline churn prediction model."""
    raw_df = load_raw_data(data_path, dataset_info_path=dataset_info_path)
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
    y_test_binary = target_to_binary(y_test)
    y_pred_binary = target_to_binary(y_pred)
    y_score = positive_class_probability(model, X_test)
    save_baseline_plots(cleaned_df, y_test_binary, y_pred_binary, y_score, plots_dir)

    save_model(model, model_path)
    save_json(metrics, metrics_path)

    return metrics


def train_models(
    data_path: str | Path = DEFAULT_DATA_PATH,
    automl_trials: int = AUTOML_TRIALS,
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
        default=AUTOML_TRIALS,
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
