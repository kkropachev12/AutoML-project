"""Optional MLflow tracking helpers for AutoML experiments."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
DEFAULT_EXPERIMENT_NAME = "telco-churn-automl"


def _flatten_params(
    params: dict[str, Any],
    prefix: str | None = None,
) -> dict[str, str | int | float | bool]:
    """Flatten nested model parameters into MLflow-compatible values."""
    flattened: dict[str, str | int | float | bool] = {}

    for key, value in params.items():
        param_name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_params(value, prefix=param_name))
        elif isinstance(value, (str, int, float, bool)):
            flattened[param_name] = value
        elif value is not None:
            flattened[param_name] = str(value)

    return flattened


def _log_existing_artifact(mlflow: Any, path: str | Path) -> None:
    """Log one artifact path only when it exists."""
    artifact_path = Path(path)
    if artifact_path.is_file():
        mlflow.log_artifact(str(artifact_path))


def _log_existing_artifact_dir(mlflow: Any, path: str | Path) -> None:
    """Log one artifact directory only when it exists."""
    artifact_path = Path(path)
    if artifact_path.is_dir():
        mlflow.log_artifacts(str(artifact_path))


def log_automl_run(
    best_params: dict[str, Any],
    best_metrics: dict[str, float],
    params_path: str | Path,
    metrics_path: str | Path,
    plots_dir: str | Path,
    model_path: str | Path | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> bool:
    """Log AutoML model quality to MLflow without making MLflow mandatory.

    Returns ``True`` when logging finishes successfully and ``False`` when
    MLflow is unavailable or raises an error.
    """
    try:
        mlflow = importlib.import_module("mlflow")

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="automl-best-model"):
            for name, value in _flatten_params(best_params).items():
                mlflow.log_param(name, value)

            for name, value in best_metrics.items():
                mlflow.log_metric(name, float(value))

            _log_existing_artifact(mlflow, params_path)
            _log_existing_artifact(mlflow, metrics_path)
            if model_path is not None:
                _log_existing_artifact(mlflow, model_path)
            _log_existing_artifact_dir(mlflow, plots_dir)

        return True
    except Exception as error:  # pragma: no cover - exact MLflow errors vary
        LOGGER.warning("MLflow logging skipped: %s", error)
        return False
