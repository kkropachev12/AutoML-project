from pathlib import Path

from src import tracking
from src.tracking import log_automl_run


class _FakeRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _FakeMLflow:
    def __init__(self):
        self.experiment_name = None
        self.run_name = None
        self.params = {}
        self.metrics = {}
        self.artifacts = []
        self.artifact_dirs = []

    def set_experiment(self, experiment_name):
        self.experiment_name = experiment_name

    def start_run(self, run_name):
        self.run_name = run_name
        return _FakeRun()

    def log_param(self, name, value):
        self.params[name] = value

    def log_metric(self, name, value):
        self.metrics[name] = value

    def log_artifact(self, path):
        self.artifacts.append(Path(path).name)

    def log_artifacts(self, path):
        self.artifact_dirs.append(Path(path).name)


def test_log_automl_run_logs_params_metrics_and_artifacts(monkeypatch, tmp_path):
    fake_mlflow = _FakeMLflow()
    monkeypatch.setattr(
        tracking.importlib,
        "import_module",
        lambda module_name: fake_mlflow,
    )
    params_path = tmp_path / "best_params.json"
    metrics_path = tmp_path / "best_metrics.json"
    model_path = tmp_path / "models" / "best_model.joblib"
    plots_dir = tmp_path / "plots"
    model_path.parent.mkdir()
    plots_dir.mkdir()
    params_path.write_text("{}", encoding="utf-8")
    metrics_path.write_text("{}", encoding="utf-8")
    model_path.write_text("model", encoding="utf-8")
    (plots_dir / "feature_importance.png").write_text("plot", encoding="utf-8")

    logged = log_automl_run(
        best_params={
            "model": "RandomForestClassifier",
            "params": {"max_depth": 5},
            "validation_roc_auc": 0.81,
        },
        best_metrics={"accuracy": 0.8, "precision": 0.75, "roc_auc": 0.82},
        params_path=params_path,
        metrics_path=metrics_path,
        plots_dir=plots_dir,
        model_path=model_path,
    )

    assert logged is True
    assert fake_mlflow.experiment_name == "telco-churn-automl"
    assert fake_mlflow.run_name == "automl-best-model"
    assert fake_mlflow.params["model"] == "RandomForestClassifier"
    assert fake_mlflow.params["params.max_depth"] == 5
    assert fake_mlflow.metrics["accuracy"] == 0.8
    assert {"best_params.json", "best_metrics.json", "best_model.joblib"}.issubset(
        set(fake_mlflow.artifacts)
    )
    assert "plots" in fake_mlflow.artifact_dirs


def test_log_automl_run_skips_when_mlflow_is_unavailable(monkeypatch, tmp_path):
    def _raise_module_not_found(module_name):
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(tracking.importlib, "import_module", _raise_module_not_found)

    logged = log_automl_run(
        best_params={"model": "LogisticRegression"},
        best_metrics={"accuracy": 1.0},
        params_path=tmp_path / "best_params.json",
        metrics_path=tmp_path / "best_metrics.json",
        plots_dir=tmp_path / "plots",
    )

    assert logged is False
