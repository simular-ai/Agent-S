"""Test-only operator classes for cross-process driver tests.

Both classes are importable by qualified name so ProcessPoolExecutor
workers can pickle/unpickle references to them via Driver's op_classes
injection point.
"""
from __future__ import annotations

from typing import ClassVar


class MarkerOperatorError(Exception):
    """Raised by _RaisingMarkerOperator. Picklable for cross-IPC tests."""


class _FilesystemMarkerOperator:
    """Writes a file to state.base_dir/_test_markers/<island>.touch when
    apply() is invoked. Used by tests that need to verify a worker
    subprocess actually executed for each island."""
    kind: ClassVar[str] = "mutation"

    def apply(self, state) -> None:
        marker_dir = state.base_dir / "_test_markers"
        marker_dir.mkdir(exist_ok=True)
        (marker_dir / f"{state.current_island}.touch").touch()


class _RaisingMarkerOperator:
    """Raises MarkerOperatorError when apply() is invoked. Used by tests
    that need to verify worker exceptions propagate across the process
    boundary."""
    kind: ClassVar[str] = "mutation"

    def apply(self, state) -> None:
        raise MarkerOperatorError(
            f"intentional test failure on island={state.current_island}"
        )
