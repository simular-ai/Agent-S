"""Agent-S3 persistence layer — generic task state store (SQLite).

Distinct from ``gui_agents.s3.taskstore`` (procedural-memory replay of
winning tool sequences). This module tracks the *lifecycle* of a task:
id, status, instruction, result, error, attempts, timestamps.
"""

from .task_store import TaskRecord, TaskStatus, TaskStore

__all__ = ["TaskRecord", "TaskStatus", "TaskStore"]