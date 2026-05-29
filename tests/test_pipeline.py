import pandas as pd
from sklearn.pipeline import Pipeline

from src.pipeline import build_baseline_pipeline
from src.train import evaluate_model, train_baseline_model


def _sample_training_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "tenure": [1, 2, 5, 12, 24, 36, 48, 60],
            "MonthlyCharges": [80.0, 75.5, 72.0, 65.0, 55.0, 50.0, 45.0, 40.0],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "Month-to-month",
                "One year",
                "One year",
                "Two year",
                "Two year",
                "Two year",
            ],
            "PaymentMethod": [
                "Electronic check",
                "Electronic check",
                "Mailed check",
                "Credit card",
                "Credit card",
                "Bank transfer",
                "Bank transfer",
                "Credit card",
            ],
        }
    )
    y = pd.Series(["Yes", "Yes", "Yes", "No", "No", "No", "No", "No"])
    return X, y


def test_build_baseline_pipeline_creates_pipeline():
    pipeline = build_baseline_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "preprocessing" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_baseline_model_trains_on_small_example():
    X, y = _sample_training_data()
    pipeline = build_baseline_pipeline()

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert len(predictions) == len(y)


def test_baseline_metrics_are_between_zero_and_one():
    X, y = _sample_training_data()
    pipeline = build_baseline_pipeline()
    pipeline.fit(X, y)

    metrics = evaluate_model(pipeline, X, y)

    assert metrics
    assert all(0 <= value <= 1 for value in metrics.values())


def test_train_baseline_model_creates_expected_artifacts(synthetic_telco_csv, tmp_path):
    model_path = tmp_path / "models" / "baseline_model.joblib"
    metrics_path = tmp_path / "baseline_metrics.json"
    plots_dir = tmp_path / "plots"
    dataset_info_path = tmp_path / "dataset_info.json"

    metrics = train_baseline_model(
        data_path=synthetic_telco_csv,
        model_path=model_path,
        metrics_path=metrics_path,
        plots_dir=plots_dir,
        dataset_info_path=dataset_info_path,
        test_size=0.25,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert dataset_info_path.exists()
    assert (plots_dir / "churn_distribution.png").exists()
    assert (plots_dir / "monthly_charges_by_churn.png").exists()
    assert (plots_dir / "confusion_matrix.png").exists()
    assert (plots_dir / "roc_curve.png").exists()
    assert all(0 <= value <= 1 for value in metrics.values())
