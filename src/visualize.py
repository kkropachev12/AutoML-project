"""Visualization helpers for Telco Customer Churn reports."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "netology_automl_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.pipeline import Pipeline

from src.config import DEFAULT_PLOTS_DIR

CHURN_DISTRIBUTION_PATH = DEFAULT_PLOTS_DIR / "churn_distribution.png"
MONTHLY_CHARGES_BY_CHURN_PATH = DEFAULT_PLOTS_DIR / "monthly_charges_by_churn.png"
CONFUSION_MATRIX_PATH = DEFAULT_PLOTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = DEFAULT_PLOTS_DIR / "roc_curve.png"
FEATURE_IMPORTANCE_PATH = DEFAULT_PLOTS_DIR / "feature_importance.png"
OPTUNA_HISTORY_PATH = DEFAULT_PLOTS_DIR / "optuna_history.png"


def _save_current_figure(path: str | Path) -> Path:
    """Save the current matplotlib figure and close it."""
    plot_path = Path(path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


def plot_churn_distribution(
    df: pd.DataFrame,
    output_path: str | Path = CHURN_DISTRIBUTION_PATH,
) -> Path:
    """Plot the distribution of the target variable."""
    if "Churn" not in df.columns:
        raise ValueError("Column 'Churn' is required to plot churn distribution.")

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Churn", hue="Churn", palette="Set2", legend=False)
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")

    return _save_current_figure(output_path)


def plot_monthly_charges_by_churn(
    df: pd.DataFrame,
    output_path: str | Path = MONTHLY_CHARGES_BY_CHURN_PATH,
) -> Path:
    """Plot MonthlyCharges distribution grouped by churn class."""
    required_columns = {"MonthlyCharges", "Churn"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns for MonthlyCharges plot: {sorted(missing_columns)}"
        )

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn", palette="Set2")
    plt.title("Monthly Charges by Churn")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    plt.legend([], [], frameon=False)

    return _save_current_figure(output_path)


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    output_path: str | Path = CONFUSION_MATRIX_PATH,
) -> Path:
    """Plot a confusion matrix for binary churn predictions."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    return _save_current_figure(output_path)


def plot_roc_curve(
    y_true: Iterable[int],
    y_score: Iterable[float],
    output_path: str | Path = ROC_CURVE_PATH,
) -> Path:
    """Plot an ROC curve for churn probability scores."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}", color="#2563eb")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Random")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)

    return _save_current_figure(output_path)


def _clean_feature_names(feature_names: Iterable[str]) -> list[str]:
    """Make sklearn feature names shorter for plot labels."""
    return [
        name.replace("numeric__", "").replace("categorical__", "")
        for name in feature_names
    ]


def _top_feature_importances(
    feature_names: Iterable[str],
    importances: Iterable[float],
    max_features: int,
) -> pd.DataFrame:
    importance_df = pd.DataFrame(
        {
            "feature": _clean_feature_names(feature_names),
            "importance": np.asarray(list(importances), dtype=float),
        }
    )
    return (
        importance_df.sort_values("importance", ascending=False)
        .head(max_features)
        .sort_values("importance", ascending=True)
    )


def plot_feature_importance(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: str | Path = FEATURE_IMPORTANCE_PATH,
    max_features: int = 20,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Path:
    """Plot feature importance for the best AutoML model.

    Tree-based models use their native ``feature_importances_`` on transformed
    features. Other models fall back to permutation importance on source columns.
    """
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        feature_names = model.named_steps["preprocessing"].get_feature_names_out()
        importance_df = _top_feature_importances(
            feature_names,
            estimator.feature_importances_,
            max_features=max_features,
        )
        title = "AutoML Feature Importance"
        xlabel = "Model importance"
    else:
        permutation = permutation_importance(
            model,
            X,
            y,
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1,
        )
        importance_df = _top_feature_importances(
            X.columns,
            permutation.importances_mean,
            max_features=max_features,
        )
        title = "AutoML Permutation Importance"
        xlabel = "Mean ROC-AUC decrease"

    plt.figure(figsize=(9, max(5, 0.35 * len(importance_df) + 1.5)))
    sns.barplot(data=importance_df, x="importance", y="feature", color="#2563eb")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Feature")
    plt.grid(axis="x", alpha=0.25)

    return _save_current_figure(output_path)


def plot_optuna_history(
    study: optuna.Study,
    output_path: str | Path = OPTUNA_HISTORY_PATH,
) -> Path:
    """Plot Optuna trial values and best-so-far optimization history."""
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not completed_trials:
        raise ValueError("Optuna study must contain at least one completed trial.")

    trial_numbers = [trial.number for trial in completed_trials]
    trial_values = [float(trial.value) for trial in completed_trials]
    best_values = np.maximum.accumulate(trial_values)

    plt.figure(figsize=(8, 5))
    plt.plot(
        trial_numbers,
        trial_values,
        marker="o",
        linestyle="",
        color="#6b7280",
        label="Trial ROC-AUC",
    )
    plt.plot(
        trial_numbers,
        best_values,
        marker="o",
        color="#2563eb",
        label="Best ROC-AUC so far",
    )
    plt.title("Optuna Optimization History")
    plt.xlabel("Trial")
    plt.ylabel("Validation ROC-AUC")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")

    return _save_current_figure(output_path)


def save_baseline_plots(
    df: pd.DataFrame,
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float],
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
) -> dict[str, Path]:
    """Save all baseline report plots and return their paths."""
    output_dir = Path(plots_dir)

    return {
        "churn_distribution": plot_churn_distribution(
            df, output_dir / "churn_distribution.png"
        ),
        "monthly_charges_by_churn": plot_monthly_charges_by_churn(
            df, output_dir / "monthly_charges_by_churn.png"
        ),
        "confusion_matrix": plot_confusion_matrix(
            y_true, y_pred, output_dir / "confusion_matrix.png"
        ),
        "roc_curve": plot_roc_curve(y_true, y_score, output_dir / "roc_curve.png"),
    }


def save_automl_interpretation_plots(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    study: optuna.Study,
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
) -> dict[str, Path]:
    """Save interpretation plots for the selected AutoML model and Optuna run."""
    output_dir = Path(plots_dir)

    return {
        "feature_importance": plot_feature_importance(
            model,
            X_test,
            y_test,
            output_dir / "feature_importance.png",
        ),
        "optuna_history": plot_optuna_history(
            study,
            output_dir / "optuna_history.png",
        ),
    }
