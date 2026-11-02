"""Shared paths and training defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "telco_customer_churn.csv"
DEFAULT_DATASET_INFO_PATH = ARTIFACTS_DIR / "dataset_info.json"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "models" / "baseline_model.joblib"
DEFAULT_METRICS_PATH = ARTIFACTS_DIR / "baseline_metrics.json"
DEFAULT_BEST_MODEL_PATH = ARTIFACTS_DIR / "models" / "best_model.joblib"
DEFAULT_BEST_PARAMS_PATH = ARTIFACTS_DIR / "best_params.json"
DEFAULT_BEST_METRICS_PATH = ARTIFACTS_DIR / "best_metrics.json"
DEFAULT_RESOURCE_USAGE_PATH = ARTIFACTS_DIR / "resource_usage.json"
DEFAULT_PLOTS_DIR = ARTIFACTS_DIR / "plots"
RANDOM_STATE = 42
TEST_SIZE = 0.2
AUTOML_TRIALS = 20
