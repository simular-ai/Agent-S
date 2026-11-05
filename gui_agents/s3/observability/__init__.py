"""Observability layer — metrics (Prometheus) + alerting (Slack)."""

from .metrics import (
    ObservabilityManager,
    track_task,
    track_action,
    measure_duration,
    TASKS_TOTAL,
    TASK_DURATION,
    ACTIONS_TOTAL,
)

__all__ = [
    "ObservabilityManager",
    "track_task",
    "track_action",
    "measure_duration",
    "TASKS_TOTAL",
    "TASK_DURATION",
    "ACTIONS_TOTAL",
]