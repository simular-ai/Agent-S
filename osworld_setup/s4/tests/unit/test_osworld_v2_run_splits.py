import hashlib
import json
from pathlib import Path

import pytest

from agent_s4.config.experiment import ExperimentConfig
from agent_s4 import orchestrator


def _digest(values):
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def _workspace(tmp_path, monkeypatch):
    split_dir = tmp_path / "eval_sets"
    split_dir.mkdir()
    values = {
        "osworld_v2_train_50.json": ["001", "002"],
        "osworld_v2_val_25.json": ["003"],
        "osworld_v2_test.json": ["004", "005"],
    }
    for name, task_ids in values.items():
        (split_dir / name).write_text(json.dumps(task_ids))
    pin = tmp_path / "release_pin.json"
    pin.write_text(json.dumps({
        "split_sha256": {
            name: _digest(task_ids) for name, task_ids in values.items()
        },
    }))
    monkeypatch.setattr(orchestrator, "_V2_PIN_PATH", pin)
    return split_dir


def test_v2_canonical_runtime_splits_pass(tmp_path, monkeypatch):
    _workspace(tmp_path, monkeypatch)
    cfg = ExperimentConfig(islands=[], benchmark="osworld_v2")
    orchestrator.validate_v2_run_splits(cfg, tmp_path)


def test_v2_noncanonical_splits_require_explicit_diagnostic_flag(
        tmp_path, monkeypatch):
    split_dir = _workspace(tmp_path, monkeypatch)
    (split_dir / "tiny_train.json").write_text(json.dumps(["001"]))
    cfg = ExperimentConfig(
        islands=[], benchmark="osworld_v2", train_set_file="tiny_train.json",
    )
    with pytest.raises(RuntimeError, match="formal OSWorld V2 runs"):
        orchestrator.validate_v2_run_splits(cfg, tmp_path)
    cfg.allow_v2_noncanonical_splits = True
    orchestrator.validate_v2_run_splits(cfg, tmp_path)


def test_v2_diagnostic_flag_never_allows_test_leakage(tmp_path, monkeypatch):
    split_dir = _workspace(tmp_path, monkeypatch)
    (split_dir / "leaked.json").write_text(json.dumps(["004"]))
    cfg = ExperimentConfig(
        islands=[], benchmark="osworld_v2", train_set_file="leaked.json",
        allow_v2_noncanonical_splits=True,
    )
    with pytest.raises(RuntimeError, match="held-out test IDs"):
        orchestrator.validate_v2_run_splits(cfg, tmp_path)


def test_v2_heldout_must_match_canonical_even_for_diagnostics(
        tmp_path, monkeypatch):
    split_dir = _workspace(tmp_path, monkeypatch)
    (split_dir / "other_test.json").write_text(json.dumps(["005"]))
    cfg = ExperimentConfig(
        islands=[], benchmark="osworld_v2",
        held_out_tasks_path=Path("eval_sets/other_test.json"),
        allow_v2_noncanonical_splits=True,
    )
    with pytest.raises(RuntimeError, match="must exactly match"):
        orchestrator.validate_v2_run_splits(cfg, tmp_path)


def test_v2_duplicate_configured_ids_are_rejected(tmp_path, monkeypatch):
    split_dir = _workspace(tmp_path, monkeypatch)
    (split_dir / "duplicate.json").write_text(json.dumps(["001", "001"]))
    cfg = ExperimentConfig(
        islands=[], benchmark="osworld_v2", train_set_file="duplicate.json",
        allow_v2_noncanonical_splits=True,
    )
    with pytest.raises(RuntimeError, match="unique JSON string list"):
        orchestrator.validate_v2_run_splits(cfg, tmp_path)


def test_v2_orchestrator_shared_preflight_fails_before_run(monkeypatch, tmp_path):
    from agent_s4 import osworld_v2_preflight

    monkeypatch.setattr(
        osworld_v2_preflight, "validate_default_workspace",
        lambda root, **kwargs: ["runtime missing", "pin untracked"],
    )
    cfg = ExperimentConfig(islands=[], benchmark="osworld_v2")
    with pytest.raises(RuntimeError, match="runtime missing"):
        orchestrator.validate_v2_workspace_before_run(cfg, tmp_path)


def test_v2_orchestrator_preflight_scopes_proxy_check_to_run_tasks(
    monkeypatch, tmp_path
):
    """The driver's gate checks the proxy only for train + val, matching the
    launchers' preflight; it never runs the held-out test split."""
    from agent_s4 import osworld_v2_preflight

    split_dir = tmp_path / "eval_sets"
    split_dir.mkdir()
    (split_dir / "train.json").write_text(json.dumps(["s_001", "s_002"]))
    (split_dir / "val.json").write_text(json.dumps(["017", "033"]))
    seen = {}
    monkeypatch.setattr(
        osworld_v2_preflight, "validate_default_workspace",
        lambda root, **kwargs: seen.update(kwargs) or [],
    )
    cfg = ExperimentConfig(
        islands=[], benchmark="osworld_v2", train_set_file="train.json",
        val_set_file="val.json", allow_v2_noncanonical_splits=True,
    )
    orchestrator.validate_v2_workspace_before_run(cfg, tmp_path)
    assert seen["selected_task_ids"] == {"s_001", "s_002", "017", "033"}


def test_v2_orchestrator_preflight_unreadable_split_checks_every_task(
    monkeypatch, tmp_path
):
    from agent_s4 import osworld_v2_preflight

    seen = {}
    monkeypatch.setattr(
        osworld_v2_preflight, "validate_default_workspace",
        lambda root, **kwargs: seen.update(kwargs) or [],
    )
    cfg = ExperimentConfig(islands=[], benchmark="osworld_v2")
    orchestrator.validate_v2_workspace_before_run(cfg, tmp_path)
    assert seen["selected_task_ids"] is None


def test_non_v2_orchestrator_skips_shared_v2_preflight(monkeypatch, tmp_path):
    from agent_s4 import osworld_v2_preflight

    monkeypatch.setattr(
        osworld_v2_preflight, "validate_default_workspace",
        lambda root: pytest.fail("must not run for V1"),
    )
    orchestrator.validate_v2_workspace_before_run(
        ExperimentConfig(islands=[], benchmark="osworld",
                         train_set_file="train.json", val_set_file="val.json"),
        tmp_path,
    )
