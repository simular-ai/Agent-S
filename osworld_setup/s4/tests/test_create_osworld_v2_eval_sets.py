"""Tests for scripts/eval/create_osworld_v2_eval_sets.py.

The splits drive selection, so the properties that matter are: disjointness
(a train/val leak silently inflates the selection signal), determinism under
a fixed seed, and that scarce capability categories don't all land in one
split.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "eval" / "create_osworld_v2_eval_sets.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("create_osworld_v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["create_osworld_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


mkset = _load_script()


def _make_examples_root(tmp_path, n_tasks=40):
    """A miniature V2 evaluation_examples/ tree: a manifest, task_class files,
    and two category manifests with a rare category (3 tasks) and a common one."""
    root = tmp_path / "evaluation_examples"
    tc = root / "task_class"
    tc.mkdir(parents=True)
    ids = [f"{i:03d}" for i in range(1, n_tasks + 1)]
    for tid in ids:
        (tc / f"task_{tid}.py").write_text("TASK_CLASS = dict\n")
    (root / "test_v2.json").write_text(json.dumps({"tasks": ids}))
    (root / "rare_capability.json").write_text(
        json.dumps({"rare_capability": ids[:3]}))
    (root / "common_capability.json").write_text(
        json.dumps({"common_capability": ids[3:]}))
    return root, ids


def test_splits_are_disjoint_and_cover_the_pool(tmp_path, monkeypatch, capsys):
    root, ids = _make_examples_root(tmp_path)
    out = tmp_path / "eval_sets"
    monkeypatch.setattr(sys, "argv", [
        "create_osworld_v2_eval_sets.py",
        "--examples-root", str(root), "--out-dir", str(out),
        "--train", "20", "--val", "10", "--seed", "42",
    ])
    assert mkset.main() == 0

    train = json.loads((out / "osworld_v2_train_20.json").read_text())
    val = json.loads((out / "osworld_v2_val_10.json").read_text())
    test = json.loads((out / "osworld_v2_test.json").read_text())

    assert len(train) == 20 and len(val) == 10 and len(test) == 10
    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)
    assert set(train) | set(val) | set(test) == set(ids)


def test_same_seed_reproduces_splits(tmp_path, monkeypatch):
    root, _ = _make_examples_root(tmp_path)

    def run(out_dir, seed):
        monkeypatch.setattr(sys, "argv", [
            "create_osworld_v2_eval_sets.py",
            "--examples-root", str(root), "--out-dir", str(out_dir),
            "--train", "20", "--val", "10", "--seed", str(seed),
        ])
        assert mkset.main() == 0
        return json.loads((out_dir / "osworld_v2_train_20.json").read_text())

    assert run(tmp_path / "a", 42) == run(tmp_path / "b", 42)
    assert run(tmp_path / "c", 7) != run(tmp_path / "a", 42)


def test_rare_category_is_spread_not_clumped(tmp_path, monkeypatch):
    """Bucketing by rarest category is the whole point: a 3-task capability
    must not land entirely in one split."""
    root, ids = _make_examples_root(tmp_path)
    rare = set(ids[:3])
    out = tmp_path / "eval_sets"
    monkeypatch.setattr(sys, "argv", [
        "create_osworld_v2_eval_sets.py",
        "--examples-root", str(root), "--out-dir", str(out),
        "--train", "20", "--val", "10", "--seed", "42",
    ])
    assert mkset.main() == 0

    train = set(json.loads((out / "osworld_v2_train_20.json").read_text()))
    val = set(json.loads((out / "osworld_v2_val_10.json").read_text()))
    assert rare & train, "rare category absent from train"
    assert rare & val, "rare category absent from val"


def test_bucket_by_rarest_prefers_the_smallest_category():
    cats = {"big": ["1", "2", "3", "4"], "small": ["1"], "mid": ["1", "2"]}
    buckets = mkset._bucket_by_rarest(["1", "2", "3", "9"], cats)
    assert buckets["1"] == "small"    # in all three -> smallest wins
    assert buckets["2"] == "mid"
    assert buckets["3"] == "big"
    assert buckets["9"] == "uncategorized"   # in no manifest


def test_manifest_ids_without_task_files_are_dropped(tmp_path, monkeypatch,
                                                     capsys):
    """The gated task_class files may be missing; ids with no file on disk
    can't be run, so they must not leak into a split."""
    root, ids = _make_examples_root(tmp_path, n_tasks=10)
    (root / "task_class" / f"task_{ids[0]}.py").unlink()
    out = tmp_path / "eval_sets"
    monkeypatch.setattr(sys, "argv", [
        "create_osworld_v2_eval_sets.py",
        "--examples-root", str(root), "--out-dir", str(out),
        "--train", "5", "--val", "2", "--seed", "1",
    ])
    assert mkset.main() == 0
    everything = set()
    for f in out.glob("osworld_v2_*.json"):
        everything |= set(json.loads(f.read_text()))
    assert ids[0] not in everything
    assert "no task_class file" in capsys.readouterr().err


def test_stratified_sample_returns_exactly_n():
    """max(1, ...) per-bucket allocation can overshoot when buckets are many
    and small; the correction loop must still land on exactly n."""
    ids = [f"{i:03d}" for i in range(30)]
    buckets = {tid: f"cat{i % 11}" for i, tid in enumerate(ids)}
    for n in (1, 5, 11, 29):
        picked = mkset.stratified_sample(ids, buckets, n, random.Random(0))
        assert len(picked) == n, f"n={n} produced {len(picked)}"
        assert len(set(picked)) == n, "duplicate ids in sample"


def test_missing_manifest_errors_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "create_osworld_v2_eval_sets.py",
        "--examples-root", str(tmp_path / "nope"),
        "--out-dir", str(tmp_path / "out"),
    ])
    assert mkset.main() == 1
