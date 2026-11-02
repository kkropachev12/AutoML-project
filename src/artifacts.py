"""Save training results in a consistent format."""

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Write a JSON artifact, creating its directory when necessary."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_model(model: Pipeline, path: str | Path) -> None:
    """Save the fitted preprocessing and classifier together."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
