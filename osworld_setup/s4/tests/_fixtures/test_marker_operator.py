from pathlib import Path
from unittest.mock import MagicMock

from tests._fixtures.marker_operator import (
    _FilesystemMarkerOperator, _RaisingMarkerOperator, MarkerOperatorError,
)


def _fake_state(island_name, base_dir):
    state = MagicMock()
    state.current_island = island_name
    state.base_dir = Path(base_dir)
    return state


def test_filesystem_marker_operator_writes_file(tmp_path):
    op = _FilesystemMarkerOperator()
    op.apply(_fake_state("haiku", tmp_path))
    assert (tmp_path / "_test_markers" / "haiku.touch").exists()


def test_filesystem_marker_operator_handles_multiple_islands(tmp_path):
    for name in ("haiku", "sonnet", "qwen"):
        _FilesystemMarkerOperator().apply(_fake_state(name, tmp_path))
    markers = sorted((tmp_path / "_test_markers").iterdir())
    assert [m.name for m in markers] == ["haiku.touch", "qwen.touch", "sonnet.touch"]


def test_raising_marker_operator_raises_typed_error(tmp_path):
    op = _RaisingMarkerOperator()
    state = _fake_state("haiku", tmp_path)
    raised = None
    try:
        op.apply(state)
    except MarkerOperatorError as e:
        raised = e
    assert raised is not None
    assert "intentional" in str(raised).lower()


def test_marker_operators_are_picklable():
    import pickle
    pickle.dumps(_FilesystemMarkerOperator)
    pickle.dumps(_RaisingMarkerOperator)
