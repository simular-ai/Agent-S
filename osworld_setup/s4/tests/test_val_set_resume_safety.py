"""Resume safety: aborts cleanly when val_set_file content has changed under a run."""
import pytest

from agent_s4.db import load_val_set, verify_val_set_unchanged


def test_passes_on_first_run(tmp_db):
    # Empty val_set — fresh DB. Loading any IDs should succeed.
    verify_val_set_unchanged(tmp_db, ["a", "b", "c"])  # no exception


def test_passes_when_unchanged(tmp_db):
    load_val_set(tmp_db, ["a", "b", "c"])
    verify_val_set_unchanged(tmp_db, ["a", "b", "c"])
    verify_val_set_unchanged(tmp_db, ["c", "b", "a"])  # set equality, order-independent


def test_rejects_when_added(tmp_db):
    load_val_set(tmp_db, ["a", "b"])
    with pytest.raises(ValueError, match="val_set has changed"):
        verify_val_set_unchanged(tmp_db, ["a", "b", "c"])


def test_rejects_when_removed(tmp_db):
    load_val_set(tmp_db, ["a", "b", "c"])
    with pytest.raises(ValueError, match="val_set has changed"):
        verify_val_set_unchanged(tmp_db, ["a", "b"])


def test_rejects_when_replaced(tmp_db):
    load_val_set(tmp_db, ["a", "b"])
    with pytest.raises(ValueError, match="val_set has changed"):
        verify_val_set_unchanged(tmp_db, ["x", "y"])


def test_error_message_lists_diff(tmp_db):
    load_val_set(tmp_db, ["keep1", "keep2", "removed_a"])
    with pytest.raises(ValueError) as ei:
        verify_val_set_unchanged(tmp_db, ["keep1", "keep2", "added_b"])
    msg = str(ei.value)
    assert "removed_a" in msg
    assert "added_b" in msg
