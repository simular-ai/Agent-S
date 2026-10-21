"""Tests for agent_s4.utils._load_val_tasks."""
import json

import pytest

from agent_s4.utils import _load_val_tasks


def test_returns_task_ids_from_file(tmp_path):
    eval_dir = tmp_path / "eval_sets"
    eval_dir.mkdir()
    f = eval_dir / "my_val.json"
    f.write_text(json.dumps(["t1", "t2", "t3"]))
    assert _load_val_tasks(str(tmp_path), "my_val.json") == ["t1", "t2", "t3"]


def test_returns_empty_for_none_string(tmp_path):
    assert _load_val_tasks(str(tmp_path), "none") == []


def test_returns_empty_for_python_none(tmp_path):
    assert _load_val_tasks(str(tmp_path), None) == []


def test_raises_for_missing_file(tmp_path):
    (tmp_path / "eval_sets").mkdir()
    with pytest.raises(FileNotFoundError):
        _load_val_tasks(str(tmp_path), "missing.json")
