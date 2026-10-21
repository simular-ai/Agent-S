"""Tests for v2 crossover operator behavior.

Verifies the two key changes from the island-evolution-v2 plan Task 7:

1. Crossover picks BOTH parents from the same island's population — it no
   longer consults state.peers() (recently-arrived migrants in this
   island's pop are the cross-environment synthesis channel).
2. The builder runs with a freshly-provisioned scratch VM (previously
   crossover passed env_extra=None — scratch VM was missing in v1).

The fixture mocks every external dependency (harnesses, AWS scratch VM,
git, DB side-effects beyond the insert_variant call). The tests are
narrow and target the two behavioral assertions called out in the plan:
peers NOT called; provision+teardown both called when scratch VM is
enabled.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_s4.harness.base import BuilderResult, EvalResults


def _make_state(tmp_path: Path, *, step: int = 6,
                scratch_vm_enabled: bool = False,
                benchmark: str = "osworld"):
    """Build a MagicMock LoopState exercising Crossover.apply without
    needing a real DB, clone dir, or config object."""
    state = MagicMock()
    state.step = step
    state.current_island = "haiku"
    state.clone_dir = tmp_path / "clone"
    state.base_dir = tmp_path
    state.eval_cache_dir = tmp_path / "eval_cache"

    cfg = state.config
    cfg.benchmark = benchmark
    cfg.scratch_vm_enabled = scratch_vm_enabled
    cfg.selection_signal = "val"
    cfg.train_set_file = "eval_island_100.json"
    cfg.val_set_file = "none"
    cfg.eval_timeout = 60
    cfg.builder_timeout = 60
    cfg.region = "us-east-1"
    cfg.resolve_eval_model.return_value = "haiku"
    cfg.resolve_builder_model.return_value = "opus"
    cfg.resolve_builder_harness_key.return_value = "claude_code"
    cfg.resolve_evaluator_harness_key.return_value = "claude_code"

    # current_config returns an island_cfg with a prompt_template path —
    # the finally block reads from it to restore CLAUDE.md.
    island_cfg = MagicMock(name="island_cfg")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("# island prompt\n")
    island_cfg.prompt_template = str(prompt_path)
    state.current_config.return_value = island_cfg

    # Build a minimal clone_dir tree.
    state.clone_dir.mkdir(parents=True, exist_ok=True)
    (state.clone_dir / "shadow_interface").mkdir(exist_ok=True)
    (state.clone_dir / "shadow_interface" / "stub.py").write_text("")

    # state.peers should NEVER be called in v2 — set a real return so
    # accidental consultation is observable through call_count.
    state.peers.return_value = []

    return state


def _write_train_set(base_dir: Path, train_file: str, task_ids: list[str]):
    eval_dir = base_dir / "eval_sets"
    eval_dir.mkdir(exist_ok=True)
    (eval_dir / train_file).write_text(json.dumps(task_ids))


def _fake_builder(cost: float = 0.5) -> MagicMock:
    m = MagicMock()
    m.launch.return_value = BuilderResult(
        exit_code=0,
        cost_usd=cost,
        log_path=Path("/tmp/builder.log"),
        crashed=False,
    )
    return m


def _fake_evaluator() -> MagicMock:
    m = MagicMock()
    m.run_tasks.return_value = EvalResults(
        pass_rate=0.5, total=2, passed=1, failed=1, results=[],
        timed_out=False,
    )
    return m


def _common_patches(local_pop):
    """Shared patch set for crossover tests. Returns a dict mapping
    target attribute name in the crossover module to a MagicMock."""
    parent_a, parent_b = local_pop[0], local_pop[1]
    return {
        "get_population": MagicMock(return_value=local_pop),
        # Each call to tournament returns a list with one variant — first
        # returns parent_a, second returns parent_b.
        "tournament": MagicMock(side_effect=[[parent_a], [parent_b]]),
        "insert_variant": MagicMock(),
        "insert_eval_result": MagicMock(),
        "update_variant": MagicMock(),
        "append_history": MagicMock(),
        "recompute_score_from_train_set": MagicMock(return_value=1),
        "recompute_val_score": MagicMock(return_value=0),
        "resolve_builder_harness": MagicMock(return_value=_fake_builder()),
        "resolve_evaluator_harness": MagicMock(return_value=_fake_evaluator()),
        "_collect_per_task_records": MagicMock(return_value={
            "t1": {"passed": 1, "raw_score": 1.0, "fitness_score": 1.0,
                   "details_json": None, "failure_type": None},
            "t2": {"passed": 0, "raw_score": 0.0, "fitness_score": 0.0,
                   "details_json": None, "failure_type": None},
        }),
        "record_builder_cost": MagicMock(),
        "record_eval_costs": MagicMock(),
        "git_archive_variant": MagicMock(),
        "git_tag_variant": MagicMock(),
        "get_next_variant_num": MagicMock(return_value=3),
        # Real fn reads eval_cache from disk + the DB; mock it here (the real
        # behavior — dest_name routing + val filter — is covered in
        # tests/test_stage_results.py).
        "stage_results_for_builder": MagicMock(),
    }


def _apply_with_patches(state, patches, scratch_provision_mock,
                        scratch_teardown_mock):
    """Run Crossover.apply() inside the patch context. The operator's
    filesystem staging (archive -> copytree into clone) is patched with a
    custom side_effect that creates the parent dirs on the real disk so
    the subsequent os.listdir + copy loop has something to read."""
    from agent_s4.operators.crossover import Crossover
    import os as _os

    clone_dir = str(state.clone_dir)

    def _fake_git_archive(_clone, _tag, dst):
        # Simulate git archive: create a stub shadow_interface/ tree
        # inside dst so the operator's `shutil.copytree(dst/shadow_interface, pa_dir)`
        # has source content.
        si = _os.path.join(dst, "shadow_interface")
        _os.makedirs(si, exist_ok=True)
        with open(_os.path.join(si, "stub.py"), "w") as f:
            f.write("# parent stub\n")

    patches["git_archive_variant"].side_effect = _fake_git_archive

    started = [patch(f"agent_s4.operators.crossover.{name}", val)
               for name, val in patches.items()]
    started.append(patch(
        "agent_s4.operators.crossover.provision_scratch_vm",
        scratch_provision_mock,
    ))
    started.append(patch(
        "agent_s4.operators.crossover.teardown_scratch_vm",
        scratch_teardown_mock,
    ))
    started.append(patch(
        "agent_s4.operators.crossover.scratch_vm_env",
        MagicMock(return_value={"SCRATCH_VM_IP": "1.2.3.4"}),
    ))

    for p in started:
        p.start()
    try:
        Crossover().apply(state)
    finally:
        for p in started:
            p.stop()


@pytest.mark.parametrize("exit_code,crashed", [(1, False), (0, True)])
def test_failed_builder_does_not_evaluate_or_record_child(tmp_path, exit_code, crashed):
    state = _make_state(tmp_path)
    pop = [
        {"id": f"haiku-if-{i:03d}", "tag": f"evol/haiku/if-{i:03d}",
         "score": 1, "val_score": 1, "generation": 1, "island": "haiku"}
        for i in (1, 2)
    ]
    patches = _common_patches(pop)
    patches["resolve_builder_harness"].return_value.launch.return_value = BuilderResult(
        exit_code=exit_code, crashed=crashed, cost_usd=0,
        log_path=tmp_path / "failed-builder.log",
    )
    teardown = MagicMock()
    with pytest.raises(RuntimeError, match="refusing crossover evaluation"):
        _apply_with_patches(state, patches, MagicMock(return_value=None), teardown)
    patches["resolve_evaluator_harness"].assert_not_called()
    patches["git_tag_variant"].assert_not_called()
    patches["insert_variant"].assert_not_called()
    patches["insert_eval_result"].assert_not_called()
    teardown.assert_called_once()


def test_crossover_both_parents_local(tmp_path):
    """Crossover picks parent_a AND parent_b from the same island's pop —
    state.peers MUST NOT be consulted in v2."""
    state = _make_state(tmp_path, step=6, scratch_vm_enabled=False)
    _write_train_set(tmp_path, "eval_island_100.json", ["t1", "t2"])

    local_pop = [
        {"id": "haiku-if-001", "tag": "evol/haiku/if-001", "score": 80,
         "val_score": 50, "eval_level": 100, "generation": 1,
         "island": "haiku"},
        {"id": "haiku-if-002", "tag": "evol/haiku/if-002", "score": 90,
         "val_score": 60, "eval_level": 100, "generation": 1,
         "island": "haiku"},
    ]
    patches = _common_patches(local_pop)
    provision = MagicMock(return_value=None)
    teardown = MagicMock()
    _apply_with_patches(state, patches, provision, teardown)

    assert state.peers.call_count == 0, (
        f"state.peers should not be consulted in v2 crossover; "
        f"got {state.peers.call_count} calls"
    )

    # Confirm both tournament calls were made (one per parent).
    assert patches["tournament"].call_count == 2

    # Confirm child variant was inserted with parent_b_id from the local
    # pop — proves parent_b came from the local population.
    assert patches["insert_variant"].call_count == 1
    variant_dict = patches["insert_variant"].call_args[0][1]
    assert variant_dict["parent_id"] == "haiku-if-001"
    assert variant_dict["parent_b_id"] == "haiku-if-002"
    assert variant_dict["origin"] == "crossover"
    assert variant_dict["step"] == 6
    assert variant_dict["island"] == "haiku"


def test_crossover_provisions_scratch_vm_when_enabled(tmp_path):
    """When scratch_vm_enabled, crossover provisions and tears down a
    scratch VM. Teardown is in finally — runs even if mid-operator
    errors raise."""
    state = _make_state(
        tmp_path, step=6, scratch_vm_enabled=True,
        benchmark="osworld_v2",
    )
    _write_train_set(tmp_path, "eval_island_100.json", ["t1", "t2"])

    local_pop = [
        {"id": "haiku-if-001", "tag": "evol/haiku/if-001", "score": 80,
         "val_score": 50, "eval_level": 100, "generation": 1,
         "island": "haiku"},
        {"id": "haiku-if-002", "tag": "evol/haiku/if-002", "score": 90,
         "val_score": 60, "eval_level": 100, "generation": 1,
         "island": "haiku"},
    ]
    patches = _common_patches(local_pop)
    fake_handle = MagicMock(name="scratch_handle")
    provision = MagicMock(return_value=fake_handle)
    teardown = MagicMock()

    _apply_with_patches(state, patches, provision, teardown)

    provision.assert_called_once_with(
        str(state.clone_dir), region="us-east-1", benchmark="osworld_v2",
    )
    assert teardown.called, (
        "teardown_scratch_vm should be called in the finally block"
    )
    # teardown should receive the same handle returned by provision.
    teardown.assert_called_with(fake_handle)


def test_crossover_v2_scratch_enabled_fails_closed_on_no_handle(tmp_path):
    state = _make_state(
        tmp_path, step=6, scratch_vm_enabled=True, benchmark="osworld_v2",
    )
    _write_train_set(tmp_path, "eval_island_100.json", ["t1", "t2"])
    local_pop = [
        {"id": "haiku-if-001", "tag": "evol/haiku/if-001", "score": 80,
         "val_score": 50, "eval_level": 100, "generation": 1,
         "island": "haiku"},
        {"id": "haiku-if-002", "tag": "evol/haiku/if-002", "score": 90,
         "val_score": 60, "eval_level": 100, "generation": 1,
         "island": "haiku"},
    ]
    patches = _common_patches(local_pop)
    with pytest.raises(RuntimeError, match="refusing a silent trajectory-only run"):
        _apply_with_patches(
            state, patches, MagicMock(return_value=None), MagicMock(),
        )


def test_crossover_skips_scratch_vm_when_disabled(tmp_path):
    """When scratch_vm_enabled=False, provision is not called; teardown
    is still invoked in finally (with None) to keep the finally branch
    uniform."""
    state = _make_state(tmp_path, step=6, scratch_vm_enabled=False)
    _write_train_set(tmp_path, "eval_island_100.json", ["t1", "t2"])

    local_pop = [
        {"id": "haiku-if-001", "tag": "evol/haiku/if-001", "score": 80,
         "val_score": 50, "eval_level": 100, "generation": 1,
         "island": "haiku"},
        {"id": "haiku-if-002", "tag": "evol/haiku/if-002", "score": 90,
         "val_score": 60, "eval_level": 100, "generation": 1,
         "island": "haiku"},
    ]
    patches = _common_patches(local_pop)
    provision = MagicMock(return_value=None)
    teardown = MagicMock()

    _apply_with_patches(state, patches, provision, teardown)

    assert not provision.called, (
        "provision_scratch_vm should NOT be called when scratch_vm_enabled=False"
    )
    # Even when disabled, teardown is invoked uniformly in finally — it
    # short-circuits on None per scratch_vm.teardown_scratch_vm.
    assert teardown.called
    teardown.assert_called_with(None)


def test_crossover_stages_both_parent_trajectories(tmp_path):
    """Crossover stages BOTH parents' train trajectories into sibling dirs
    (eval_results_parent_a/b) and names both — plus train scores and a
    clarifying line — in the builder prompt."""
    state = _make_state(tmp_path, step=6, scratch_vm_enabled=False)
    _write_train_set(tmp_path, "eval_island_100.json", ["t1", "t2"])

    local_pop = [
        {"id": "haiku-if-001", "tag": "evol/haiku/if-001", "score": 80,
         "val_score": 50, "eval_level": 100, "generation": 1,
         "island": "haiku"},
        {"id": "haiku-if-002", "tag": "evol/haiku/if-002", "score": 90,
         "val_score": 60, "eval_level": 100, "generation": 1,
         "island": "haiku"},
    ]
    patches = _common_patches(local_pop)
    _apply_with_patches(state, patches, MagicMock(return_value=None), MagicMock())

    # Both parents staged, into distinct sibling dirs, parent_a first.
    stage = patches["stage_results_for_builder"]
    assert stage.call_count == 2
    a_args, a_kw = stage.call_args_list[0]
    b_args, b_kw = stage.call_args_list[1]
    assert a_args[0] == "haiku-if-001"
    assert a_kw["dest_name"] == "eval_results_parent_a"
    assert b_args[0] == "haiku-if-002"
    assert b_kw["dest_name"] == "eval_results_parent_b"

    # Builder prompt names both trajectory dirs, shows TRAIN scores (not val),
    # and clarifies what the trajectories are.
    prompt = patches["resolve_builder_harness"].return_value.launch.call_args.kwargs["prompt"]
    assert "eval_results_parent_a/" in prompt
    assert "eval_results_parent_b/" in prompt
    assert "train=80" in prompt and "train=90" in prompt
    assert "val=" not in prompt
    assert "trajectories" in prompt
