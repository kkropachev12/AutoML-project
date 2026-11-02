"""Simple Optuna AutoML for Telco Customer Churn."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.artifacts import save_json, save_model
from src.config import (
    AUTOML_TRIALS,
    DEFAULT_BEST_METRICS_PATH,
    DEFAULT_BEST_MODEL_PATH,
    DEFAULT_BEST_PARAMS_PATH,
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
from src.metrics import evaluate_model as evaluate_automl_model
from src.metrics import positive_class_probability, target_to_binary
from src.monitor import (
    DEFAULT_RESOURCE_USAGE_PATH,
    finish_resource_monitoring,
    start_resource_monitoring,
)
from src.pipeline import build_model_pipeline
from src.tracking import log_automl_run
from src.visualize import DEFAULT_PLOTS_DIR, save_automl_interpretation_plots


@dataclass
class AutoMLResult:
    """Result of the AutoML training run."""

    best_model: Pipeline
    best_params: dict[str, Any]
    best_metrics: dict[str, float]
    validation_roc_auc: float


def _suggest_model(trial: optuna.Trial, random_state: int) -> tuple[str, object]:
    """Suggest one candidate model and its hyperparameters."""
    model_name = trial.suggest_categorical(
        "model",
        ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"],
    )

    if model_name == "LogisticRegression":
        classifier = LogisticRegression(
            C=trial.suggest_float("logistic_regression_c", 0.01, 10.0, log=True),
            penalty=trial.suggest_categorical("logistic_regression_penalty", ["l2"]),
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )
    elif model_name == "RandomForestClassifier":
        classifier = RandomForestClassifier(
            n_estimators=trial.suggest_int("random_forest_n_estimators", 50, 200),
            max_depth=trial.suggest_int("random_forest_max_depth", 2, 20),
            min_samples_split=trial.suggest_int(
                "random_forest_min_samples_split", 2, 10
            ),
            min_samples_leaf=trial.suggest_int("random_forest_min_samples_leaf", 1, 5),
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        classifier = GradientBoostingClassifier(
            n_estimators=trial.suggest_int("gradient_boosting_n_estimators", 50, 200),
            learning_rate=trial.suggest_float(
                "gradient_boosting_learning_rate", 0.01, 0.3, log=True
            ),
            max_depth=trial.suggest_int("gradient_boosting_max_depth", 2, 5),
            min_samples_leaf=trial.suggest_int(
                "gradient_boosting_min_samples_leaf", 1, 5
            ),
            random_state=random_state,
        )

    return model_name, classifier


def train_automl_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_BEST_MODEL_PATH,
    params_path: str | Path = DEFAULT_BEST_PARAMS_PATH,
    metrics_path: str | Path = DEFAULT_BEST_METRICS_PATH,
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
    resource_usage_path: str | Path = DEFAULT_RESOURCE_USAGE_PATH,
    dataset_info_path: str | Path = DEFAULT_DATASET_INFO_PATH,
    n_trials: int = AUTOML_TRIALS,
    test_size: float = TEST_SIZE,
    validation_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> AutoMLResult:
    """Train candidate models with Optuna and save the best AutoML artifacts."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")

    resource_snapshot = start_resource_monitoring()
    try:
        raw_df = load_raw_data(data_path, dataset_info_path=dataset_info_path)
        cleaned_df = clean_data(raw_df)
        X, y = prepare_data(cleaned_df)

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full,
            y_train_full,
            test_size=validation_size,
            random_state=random_state,
            stratify=y_train_full,
        )

        def objective(trial: optuna.Trial) -> float:
            _, classifier = _suggest_model(trial, random_state=random_state)
            model = build_model_pipeline(classifier)
            model.fit(X_train, y_train)
            y_valid_score = positive_class_probability(model, X_valid)
            return float(roc_auc_score(target_to_binary(y_valid), y_valid_score))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = dict(study.best_trial.params)
        best_hyperparameters = {
            name: value for name, value in best_params.items() if name != "model"
        }
        best_model_name, best_classifier = _suggest_model(
            study.best_trial,
            random_state=random_state,
        )
        best_model = build_model_pipeline(best_classifier)
        best_model.fit(X_train_full, y_train_full)
        best_metrics = evaluate_automl_model(best_model, X_test, y_test)

        save_model(best_model, model_path)

        best_params_artifact = {
            "model": best_model_name,
            "params": best_hyperparameters,
            "validation_roc_auc": float(study.best_value),
        }
        save_json(best_params_artifact, params_path)
        save_json(best_metrics, metrics_path)
        save_automl_interpretation_plots(best_model, X_test, y_test, study, plots_dir)
        log_automl_run(
            best_params=best_params_artifact,
            best_metrics=best_metrics,
            params_path=params_path,
            metrics_path=metrics_path,
            plots_dir=plots_dir,
            model_path=model_path,
        )

        return AutoMLResult(
            best_model=best_model,
            best_params=best_params_artifact,
            best_metrics=best_metrics,
            validation_roc_auc=float(study.best_value),
        )
    finally:
        finish_resource_monitoring(resource_snapshot, resource_usage_path)
