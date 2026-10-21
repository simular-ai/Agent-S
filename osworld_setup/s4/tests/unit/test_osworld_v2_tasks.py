"""Dual-corpus task indexing/loading in agent_s4.eval.osworld_v2.

OSWorld V2 ships two task corpora under one evaluation_examples/ tree: the
v1-style examples/<domain>/<uuid>.json files, and the V2-native
task_class/task_<NNN>.py BaseTask subclasses (gated, so they may be absent
from a fresh checkout). These tests build a miniature tree on tmp_path so they
never depend on either benchmark being present.
"""
from __future__ import annotations

import json

import pytest

from agent_s4.eval import osworld_v2


def _make_tree(tmp_path, *, with_task_class=True):
    base = tmp_path / "evaluation_examples"
    domain = base / "examples" / "chrome"
    domain.mkdir(parents=True)
    (domain / "abc-123.json").write_text(json.dumps({
        "id": "abc-123", "instruction": "v1 style task", "config": [],
    }))

    if with_task_class:
        tc = base / "task_class"
        tc.mkdir()
        # Files the index must skip — neither is a task.
        (tc / "__init__.py").write_text("")
        (tc / "generated_task_utils.py").write_text("HELPER = 1\n")
        # Stands in for a real task_class module. Upstream's BaseTask
        # subclasses dict; this mimics that contract without importing
        # desktop_env, so the test stays hermetic (the installed desktop_env
        # is the v1 fork, which has no task_base module).
        (tc / "task_001.py").write_text(
            "class Task001(dict):\n"
            "    def __init__(self):\n"
            "        super().__init__(id='001', instruction='v2 native task')\n"
            "TASK_CLASS = Task001\n"
        )
        # V2's loader lives at the repo root, the parent of evaluation_examples.
        (tmp_path / "task_loader.py").write_text(
            "import importlib.util, sys, json, os\n"
            "def load_task_from_file(p):\n"
            "    if not p.endswith('.py'):\n"
            "        return json.load(open(p))\n"
            "    name = 'osworld_task_' + os.path.basename(p)[:-3]\n"
            "    spec = importlib.util.spec_from_file_location(name, p)\n"
            "    m = importlib.util.module_from_spec(spec)\n"
            "    sys.modules[name] = m\n"
            "    spec.loader.exec_module(m)\n"
            "    return m.TASK_CLASS()\n"
        )
    return base


@pytest.fixture(autouse=True)
def _reset_loader_cache():
    """_task_loader memoizes the module; each test gets its own tmp tree."""
    osworld_v2._TASK_LOADER = None
    yield
    osworld_v2._TASK_LOADER = None


def test_index_covers_both_corpora(tmp_path):
    base = _make_tree(tmp_path)
    index = osworld_v2.build_task_index(str(base))

    assert set(index) == {"abc-123", "001"}
    assert index["abc-123"].endswith("chrome/abc-123.json")
    # id is the filename suffix, matching the ids listed in test_v2.json
    assert index["001"].endswith("task_class/task_001.py")


def test_index_skips_non_task_modules(tmp_path):
    base = _make_tree(tmp_path)
    index = osworld_v2.build_task_index(str(base))

    assert "__init__" not in index
    assert "generated_task_utils" not in index
    assert not any("generated_task_utils" in p for p in index.values())


def test_index_without_task_class_is_v1_only(tmp_path):
    """A v1 tree (or a V2 checkout missing the gated task files) still works."""
    base = _make_tree(tmp_path, with_task_class=False)
    assert set(osworld_v2.build_task_index(str(base))) == {"abc-123"}


def test_load_task_dispatches_on_extension(tmp_path):
    base = _make_tree(tmp_path)
    index = osworld_v2.build_task_index(str(base))

    v1 = osworld_v2.load_task(index["abc-123"], str(base))
    assert isinstance(v1, dict) and v1["instruction"] == "v1 style task"

    v2 = osworld_v2.load_task(index["001"], str(base))
    # BaseTask subclasses dict, so run_single_task's example.get("id") and
    # DesktopEnv.reset(task_config=...) work for both corpora unchanged.
    assert isinstance(v2, dict)
    assert v2["id"] == "001"
    assert v2["instruction"] == "v2 native task"


def test_load_task_requires_task_loader_at_repo_root(tmp_path):
    """A .py task with no task_loader.py alongside the benchmark fails with an
    actionable message rather than an ImportError from deep in importlib."""
    base = _make_tree(tmp_path)
    (tmp_path / "task_loader.py").unlink()

    with pytest.raises(RuntimeError, match="task_loader"):
        osworld_v2.load_task(str(base / "task_class" / "task_001.py"), str(base))


@pytest.mark.parametrize("payload,expected", [
    ({"tasks": ["001", "002"]}, ["001", "002"]),          # V2 test_v2.json
    ({"chrome": ["a"], "gimp": ["b", "c"]}, ["a", "b", "c"]),  # v1 test_all
    (["x", "y"], ["x", "y"]),                             # flat list
])
def test_load_all_task_ids_handles_both_manifest_shapes(
        tmp_path, payload, expected):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(payload))
    assert osworld_v2.load_all_task_ids(str(p)) == expected
