from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from src.automl import train_automl_model


def _write_sample_telco_csv(path: Path) -> None:
    rows = []
    for index in range(40):
        churn = "Yes" if index % 2 == 0 else "No"
        rows.append(
            {
                "customerID": f"{index:04d}-TEST",
                "gender": "Female" if index % 3 == 0 else "Male",
                "tenure": index + 1,
                "Contract": "Month-to-month" if churn == "Yes" else "Two year",
                "PaymentMethod": (
                    "Electronic check" if churn == "Yes" else "Credit card"
                ),
                "MonthlyCharges": 80.0 if churn == "Yes" else 40.0,
                "TotalCharges": str((index + 1) * (80.0 if churn == "Yes" else 40.0)),
                "Churn": churn,
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)


def test_automl_runs_with_small_number_of_trials(tmp_path):
    csv_path = tmp_path / "telco_customer_churn.csv"
    _write_sample_telco_csv(csv_path)

    result = train_automl_model(
        data_path=csv_path,
        model_path=tmp_path / "models" / "best_model.joblib",
        params_path=tmp_path / "best_params.json",
        metrics_path=tmp_path / "best_metrics.json",
        n_trials=2,
    )

    assert isinstance(result.best_model, Pipeline)
    assert result.best_params["model"] in {
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
    }


def test_automl_creates_best_params_and_metrics_files(tmp_path):
    csv_path = tmp_path / "telco_customer_churn.csv"
    params_path = tmp_path / "best_params.json"
    metrics_path = tmp_path / "best_metrics.json"
    plots_dir = tmp_path / "plots"
    _write_sample_telco_csv(csv_path)

    result = train_automl_model(
        data_path=csv_path,
        model_path=tmp_path / "models" / "best_model.joblib",
        params_path=params_path,
        metrics_path=metrics_path,
        plots_dir=plots_dir,
        n_trials=2,
    )

    assert params_path.exists()
    assert metrics_path.exists()
    assert (plots_dir / "feature_importance.png").exists()
    assert (plots_dir / "optuna_history.png").exists()
    assert all(0 <= value <= 1 for value in result.best_metrics.values())
