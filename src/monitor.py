"""Lightweight resource monitoring for training runs."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.artifacts import save_json
from src.config import DEFAULT_RESOURCE_USAGE_PATH

LOGGER = logging.getLogger(__name__)


@dataclass
class ResourceMonitorSnapshot:
    """Initial state for a training resource monitoring interval."""

    started_at: float
    process: Any | None = None


def start_resource_monitoring() -> ResourceMonitorSnapshot:
    """Start measuring elapsed time and prime psutil CPU tracking."""
    snapshot = ResourceMonitorSnapshot(started_at=time.perf_counter())

    try:
        import psutil

        process = psutil.Process()
        process.cpu_percent(interval=None)
        snapshot.process = process
    except Exception as error:  # pragma: no cover - environment-specific failure
        LOGGER.warning("Resource monitoring started without psutil metrics: %s", error)

    return snapshot


def collect_resource_usage(
    snapshot: ResourceMonitorSnapshot,
) -> dict[str, float | None]:
    """Collect training time, CPU percent, and current RAM usage."""
    usage: dict[str, float | None] = {
        "training_time_seconds": max(time.perf_counter() - snapshot.started_at, 0.0),
        "cpu_percent": None,
        "ram_usage_mb": None,
    }

    if snapshot.process is None:
        return usage

    try:
        usage["cpu_percent"] = float(snapshot.process.cpu_percent(interval=None))
        usage["ram_usage_mb"] = float(
            snapshot.process.memory_info().rss / (1024 * 1024)
        )
    except Exception as error:  # pragma: no cover - environment-specific failure
        LOGGER.warning("Resource metrics collection skipped: %s", error)

    return usage


def save_resource_usage(
    usage: dict[str, float | None],
    output_path: str | Path = DEFAULT_RESOURCE_USAGE_PATH,
) -> bool:
    """Save resource usage to JSON without raising monitoring errors."""
    try:
        save_json(usage, output_path)
        return True
    except Exception as error:  # pragma: no cover - filesystem failures vary
        LOGGER.warning("Resource usage artifact was not saved: %s", error)
        return False


def finish_resource_monitoring(
    snapshot: ResourceMonitorSnapshot,
    output_path: str | Path = DEFAULT_RESOURCE_USAGE_PATH,
) -> dict[str, float | None]:
    """Collect and save resource usage without affecting the training pipeline."""
    try:
        usage = collect_resource_usage(snapshot)
        save_resource_usage(usage, output_path)
        return usage
    except Exception as error:  # pragma: no cover - defensive guard
        LOGGER.warning("Resource monitoring skipped: %s", error)
        return {
            "training_time_seconds": None,
            "cpu_percent": None,
            "ram_usage_mb": None,
        }
