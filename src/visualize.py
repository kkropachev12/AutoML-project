"""Visualization helpers for Telco Customer Churn reports."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "netology_automl_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "artifacts" / "plots"

CHURN_DISTRIBUTION_PATH = DEFAULT_PLOTS_DIR / "churn_distribution.png"
MONTHLY_CHARGES_BY_CHURN_PATH = DEFAULT_PLOTS_DIR / "monthly_charges_by_churn.png"
CONFUSION_MATRIX_PATH = DEFAULT_PLOTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = DEFAULT_PLOTS_DIR / "roc_curve.png"


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
