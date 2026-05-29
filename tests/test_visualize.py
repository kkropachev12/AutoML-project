import pandas as pd

from src.visualize import (
    plot_churn_distribution,
    plot_confusion_matrix,
    plot_monthly_charges_by_churn,
    plot_roc_curve,
    save_baseline_plots,
)


def _sample_churn_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 80.00, 91.25],
            "Churn": ["No", "No", "Yes", "No", "Yes", "Yes"],
        }
    )


def test_plot_churn_distribution_creates_file(tmp_path):
    output_path = tmp_path / "plots" / "churn_distribution.png"

    result_path = plot_churn_distribution(_sample_churn_data(), output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_monthly_charges_by_churn_creates_file(tmp_path):
    output_path = tmp_path / "plots" / "monthly_charges_by_churn.png"

    result_path = plot_monthly_charges_by_churn(_sample_churn_data(), output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_confusion_matrix_creates_file(tmp_path):
    output_path = tmp_path / "plots" / "confusion_matrix.png"

    result_path = plot_confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_roc_curve_creates_file(tmp_path):
    output_path = tmp_path / "plots" / "roc_curve.png"

    result_path = plot_roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.7, 0.9], output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_baseline_plots_creates_expected_files(tmp_path):
    result_paths = save_baseline_plots(
        _sample_churn_data(),
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        y_score=[0.1, 0.4, 0.7, 0.9],
        plots_dir=tmp_path / "plots",
    )

    assert set(result_paths) == {
        "churn_distribution",
        "monthly_charges_by_churn",
        "confusion_matrix",
        "roc_curve",
    }
    assert all(path.exists() for path in result_paths.values())
