import json

from sklearn.pipeline import Pipeline

from src import automl
from src.automl import train_automl_model


def test_automl_runs_with_small_number_of_trials(
    synthetic_telco_csv, monkeypatch, tmp_path
):
    monkeypatch.setattr(automl, "log_automl_run", lambda **kwargs: False)

    result = train_automl_model(
        data_path=synthetic_telco_csv,
        model_path=tmp_path / "models" / "best_model.joblib",
        params_path=tmp_path / "best_params.json",
        metrics_path=tmp_path / "best_metrics.json",
        plots_dir=tmp_path / "plots",
        resource_usage_path=tmp_path / "resource_usage.json",
        dataset_info_path=tmp_path / "dataset_info.json",
        n_trials=2,
    )

    assert isinstance(result.best_model, Pipeline)
    assert result.best_params["model"] in {
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
    }


def test_automl_creates_main_artifacts(synthetic_telco_csv, monkeypatch, tmp_path):
    monkeypatch.setattr(automl, "log_automl_run", lambda **kwargs: False)

    model_path = tmp_path / "models" / "best_model.joblib"
    params_path = tmp_path / "best_params.json"
    metrics_path = tmp_path / "best_metrics.json"
    dataset_info_path = tmp_path / "dataset_info.json"
    resource_usage_path = tmp_path / "resource_usage.json"
    plots_dir = tmp_path / "plots"

    result = train_automl_model(
        data_path=synthetic_telco_csv,
        model_path=model_path,
        params_path=params_path,
        metrics_path=metrics_path,
        dataset_info_path=dataset_info_path,
        resource_usage_path=resource_usage_path,
        plots_dir=plots_dir,
        n_trials=2,
    )

    assert model_path.exists()
    assert params_path.exists()
    assert metrics_path.exists()
    assert dataset_info_path.exists()
    assert resource_usage_path.exists()
    assert (plots_dir / "feature_importance.png").exists()
    assert (plots_dir / "optuna_history.png").exists()
    assert all(0 <= value <= 1 for value in result.best_metrics.values())

    resource_usage = json.loads(resource_usage_path.read_text(encoding="utf-8"))
    assert "training_time_seconds" in resource_usage
    assert "cpu_percent" in resource_usage
    assert "ram_usage_mb" in resource_usage
    assert resource_usage["training_time_seconds"] >= 0
    assert resource_usage["cpu_percent"] is None or resource_usage["cpu_percent"] >= 0
    assert resource_usage["ram_usage_mb"] is None or resource_usage["ram_usage_mb"] > 0
