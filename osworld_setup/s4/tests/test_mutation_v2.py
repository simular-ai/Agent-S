"""Tests for v2 mutation operator behavior.

Verifies the two key changes from the island-evolution-v2 plan Task 5:

1. Mutation.apply() writes the dispatch step (state.step) into the new
   variants.step column via insert_variant. This is what lets the
   driver's resume protocol find orphans from a crashed step.
2. The tier-cascade promotion block is gone; mutation evaluates on the
   single canonical train set (state.config.train_set_file) plus val,
   and does not call fill_eval or _next_tier.

The fixture mocks every external dependency (harnesses, AWS, git, DB
side-effects beyond the insert_variant call we're capturing). The test
is intentionally narrow — it asserts only that step is written and
that the cascade helpers are no longer invoked.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


from agent_s4.harness.base import BuilderResult, EvalResults


def _make_state(tmp_path: Path, step: int = 7, *,
                benchmark: str = "osworld", scratch_vm_enabled: bool = False):
    """Build a MagicMock LoopState that exercises Mutation.apply without
    needing a real DB, clone dir, or config object."""
    state = MagicMock()
    state.step = step
    state.current_island = "haiku"
    state.clone_dir = tmp_path / "clone"
    state.base_dir = tmp_path
    # eval_cache_dir is accessed via `.name` (Path property) — use a real Path.
    state.eval_cache_dir = tmp_path / "eval_cache"

    # Config — explicit values matter; MagicMock would auto-create attrs
    # but the boolean checks (scratch_vm_enabled, val_set_file == "none")
    # need real values.
    cfg = state.config
    cfg.benchmark = benchmark
    cfg.scratch_vm_enabled = scratch_vm_enabled
    cfg.selection_signal = "val"
    cfg.parent_selection = "tournament"
    cfg.train_set_file = "eval_island_100.json"
    cfg.val_set_file = "none"
    cfg.eval_timeout = 60
    cfg.builder_timeout = 60
    cfg.region = "us-east-1"
    cfg.resolve_eval_model.return_value = "haiku"
    cfg.resolve_builder_model.return_value = "opus"
    cfg.resolve_builder_harness_key.return_value = "claude_code"
    cfg.resolve_evaluator_harness_key.return_value = "claude_code"

    # Parent variant returned by the mocked tournament call.
    state.current_config.return_value = MagicMock(name="island_cfg")

    # Build a minimal clone_dir tree so the operator's pre-checkout
    # cleanup (`shutil.rmtree` of stale results/eval_results) is a no-op.
    state.clone_dir.mkdir(parents=True, exist_ok=True)

    return state


def _write_train_set(base_dir: Path, train_file: str, task_ids: list[str]):
    eval_dir = base_dir / "eval_sets"
    eval_dir.mkdir(exist_ok=True)
    import json
    (eval_dir / train_file).write_text(json.dumps(task_ids))


def _fake_builder(cost: float = 0.42, crashed: bool = False) -> MagicMock:
    m = MagicMock()
    m.launch.return_value = BuilderResult(
        exit_code=0 if not crashed else 1,
        cost_usd=cost,
        log_path=Path("/tmp/builder.log"),
        crashed=crashed,
    )
    return m


def _fake_evaluator(timed_out: bool = False) -> MagicMock:
    m = MagicMock()
    m.run_tasks.return_value = EvalResults(
        pass_rate=0.5, total=2, passed=1, failed=1, results=[],
        timed_out=timed_out,
    )
    return m


def test_mutation_writes_step_column(tmp_path):
    """Mutation.apply must write the dispatch step into the variants
    insert_variant call (the new variants.step column)."""
    from agent_s4.operators.mutation import Mutation

    state = _make_state(
        tmp_path, step=7, benchmark="osworld_v2",
        scratch_vm_enabled=True,
    )
    _write_train_set(tmp_path, "eval_island_100.json", ["t1", "t2"])

    parent = {
        "id": "haiku-seed", "tag": "evol/haiku/seed",
        "score": 1, "val_score": 0, "eval_level": 100,
        "generation": 0, "island": "haiku",
    }

    patches = {
        "get_population": MagicMock(return_value=[parent]),
        "tournament": MagicMock(return_value=[parent]),
        "insert_variant": MagicMock(),
        "insert_eval_result": MagicMock(),
        "update_variant": MagicMock(),
        "get_variant": MagicMock(return_value={**parent, "score": 1}),
        "append_history": MagicMock(),
        "recompute_score_from_train_set": MagicMock(return_value=1),
        "recompute_val_score": MagicMock(return_value=0),
        "resolve_builder_harness": MagicMock(return_value=_fake_builder()),
        "resolve_evaluator_harness": MagicMock(return_value=_fake_evaluator()),
        "provision_scratch_vm": MagicMock(return_value=MagicMock()),
        "teardown_scratch_vm": MagicMock(),
        "scratch_vm_env": MagicMock(return_value=None),
        "_collect_per_task_records": MagicMock(return_value={
            "t1": {"passed": 1, "raw_score": 1.0, "fitness_score": 1.0,
                   "details_json": None, "failure_type": None},
            "t2": {"passed": 0, "raw_score": 0.0, "fitness_score": 0.0,
                   "details_json": None, "failure_type": None},
        }),
        "record_builder_cost": MagicMock(),
        "record_eval_costs": MagicMock(),
        "stage_results_for_builder": MagicMock(),
        "git_checkout_variant": MagicMock(),
        "git_tag_variant": MagicMock(),
        "revert_out_of_scope": MagicMock(),
        "get_next_variant_num": MagicMock(return_value=1),
        "_recoverable_tagged_child": MagicMock(return_value=None),
    }
    started = [patch(f"agent_s4.operators.mutation.{name}", val)
               for name, val in patches.items()]
    for p in started:
        p.start()
    try:
        Mutation().apply(state)
    finally:
        for p in started:
            p.stop()

    # Capture the variant dict passed to insert_variant.
    assert patches["insert_variant"].call_count == 1, (
        f"expected exactly one insert_variant call, got "
        f"{patches['insert_variant'].call_count}"
    )
    args, _ = patches["insert_variant"].call_args
    variant_dict = args[1]
    assert variant_dict["step"] == 7, (
        f"insert_variant should receive step=7 from state.step; "
        f"got step={variant_dict.get('step')!r} in {variant_dict}"
    )
    patches["provision_scratch_vm"].assert_called_once_with(
        str(state.clone_dir), region="us-east-1", benchmark="osworld_v2",
    )


def test_mutation_no_longer_calls_tier_cascade(tmp_path):
    """The promotion-cascade block (fill_eval + _next_tier) must be gone.

    Mutation.apply now evaluates each variant on the single canonical
    train_set_file and never re-evaluates at a higher tier. This guards
    against accidentally re-introducing the cascade during merge.
    """
    from agent_s4.operators import mutation as mut_mod
    from agent_s4.operators.mutation import Mutation

    state = _make_state(tmp_path, step=1)
    _write_train_set(tmp_path, "eval_island_100.json", ["t1", "t2"])

    parent = {
        "id": "haiku-seed", "tag": "evol/haiku/seed",
        "score": 2, "val_score": 0, "eval_level": 100,
        "generation": 0, "island": "haiku",
    }
    # `fill_eval` and `_next_tier` should NOT be looked up at all in the
    # refactored module. Confirm via module-level introspection.
    assert not hasattr(mut_mod, "_next_tier"), (
        "_next_tier should be removed from mutation.py"
    )
    assert not hasattr(mut_mod, "_load_eval_tasks_at_level"), (
        "_load_eval_tasks_at_level should be removed from mutation.py"
    )

    patches = {
        "get_population": MagicMock(return_value=[parent]),
        "tournament": MagicMock(return_value=[parent]),
        "insert_variant": MagicMock(),
        "insert_eval_result": MagicMock(),
        "update_variant": MagicMock(),
        "get_variant": MagicMock(return_value={**parent, "score": 2}),
        "append_history": MagicMock(),
        "recompute_score_from_train_set": MagicMock(return_value=2),
        "recompute_val_score": MagicMock(return_value=0),
        "resolve_builder_harness": MagicMock(return_value=_fake_builder()),
        "resolve_evaluator_harness": MagicMock(return_value=_fake_evaluator()),
        "provision_scratch_vm": MagicMock(return_value=None),
        "teardown_scratch_vm": MagicMock(),
        "scratch_vm_env": MagicMock(return_value=None),
        "_collect_per_task_records": MagicMock(return_value={
            tid: {"passed": 1, "raw_score": 1.0, "fitness_score": 1.0,
                  "details_json": None, "failure_type": None}
            for tid in ("t1", "t2")
        }),
        "record_builder_cost": MagicMock(),
        "record_eval_costs": MagicMock(),
        "stage_results_for_builder": MagicMock(),
        "git_checkout_variant": MagicMock(),
        "git_tag_variant": MagicMock(),
        "revert_out_of_scope": MagicMock(),
        "get_next_variant_num": MagicMock(return_value=1),
        "_recoverable_tagged_child": MagicMock(return_value=None),
    }
    started = [patch(f"agent_s4.operators.mutation.{name}", val)
               for name, val in patches.items()]
    for p in started:
        p.start()
    try:
        Mutation().apply(state)
    finally:
        for p in started:
            p.stop()

    # Sanity: insert was called once, with eval_level=0 (deprecated col)
    # and origin=mutation. Cascade would have caused multiple update_variant
    # calls bumping eval_level; we only allow the single score/val_score
    # update at the end.
    args, _ = patches["insert_variant"].call_args
    variant_dict = args[1]
    assert variant_dict["eval_level"] == 0
    assert variant_dict["origin"] == "mutation"


def test_mutation_resumes_tagged_child_without_rebuilding(tmp_path):
    """A tag + build cost resumes before a new tournament parent is drawn."""
    from agent_s4.operators.mutation import Mutation

    state = _make_state(tmp_path, step=1, benchmark="osworld_v2")
    _write_train_set(tmp_path, "eval_island_100.json", ["t1", "t2"])
    parent = {
        "id": "haiku-seed", "tag": "evol/haiku/seed",
        "score": 2, "val_score": 0, "eval_level": 0,
        "generation": 0, "island": "haiku",
    }
    builder = _fake_builder()
    evaluator = _fake_evaluator()
    recovered = {
        "variant_num": 2,
        "child_id": "haiku-if-002",
        "tag": "evol/haiku/if-002",
        "builder_cost": 1.25,
        "parent": parent,
    }
    patches = {
        "get_population": MagicMock(return_value=[parent]),
        "tournament": MagicMock(return_value=[parent]),
        "insert_variant": MagicMock(),
        "insert_eval_result": MagicMock(),
        "update_variant": MagicMock(),
        "get_variant": MagicMock(return_value={**parent, "score": 2}),
        "append_history": MagicMock(),
        "recompute_score_from_train_set": MagicMock(return_value=1),
        "recompute_val_score": MagicMock(return_value=0),
        "resolve_builder_harness": MagicMock(return_value=builder),
        "resolve_evaluator_harness": MagicMock(return_value=evaluator),
        "_collect_per_task_records": MagicMock(return_value={
            tid: {"passed": 1, "raw_score": 1.0, "fitness_score": 1.0,
                  "details_json": None, "failure_type": None}
            for tid in ("t1", "t2")
        }),
        "record_builder_cost": MagicMock(),
        "record_eval_costs": MagicMock(),
        "stage_results_for_builder": MagicMock(),
        "git_checkout_variant": MagicMock(),
        "git_tag_variant": MagicMock(),
        "revert_out_of_scope": MagicMock(),
        "_recoverable_tagged_child": MagicMock(return_value=recovered),
    }
    started = [patch(f"agent_s4.operators.mutation.{name}", value)
               for name, value in patches.items()]
    for item in started:
        item.start()
    try:
        Mutation().apply(state)
    finally:
        for item in started:
            item.stop()

    builder.launch.assert_not_called()
    patches["tournament"].assert_not_called()
    patches["record_builder_cost"].assert_not_called()
    patches["stage_results_for_builder"].assert_not_called()
    patches["git_tag_variant"].assert_not_called()
    patches["git_checkout_variant"].assert_called_once_with(
        str(state.clone_dir), "evol/haiku/if-002",
    )
    evaluator.run_tasks.assert_called_once()
    variant = patches["insert_variant"].call_args.args[1]
    assert variant["id"] == "haiku-if-002"
    assert variant["builder_cost"] == 1.25


def test_mutation_latest_selection_extends_the_chain(tmp_path):
    """parent_selection='latest' must mutate the newest scored variant and
    never consult the tournament.

    The population deliberately puts the highest val_score on the seed and
    the lowest on the newest variant: a linear route extends the chain from
    step 2 regardless, which is the whole point of the mode and the one
    behaviour a tournament would get wrong.
    """
    from agent_s4.operators.mutation import Mutation

    state = _make_state(
        tmp_path, step=3, benchmark="osworld_v2", scratch_vm_enabled=False,
    )
    state.config.parent_selection = "latest"
    _write_train_set(tmp_path, "eval_island_100.json", ["t1", "t2"])

    def variant(vid, step, created_at, val):
        return {
            "id": vid, "tag": f"evol/haiku/{vid.split('-', 1)[1]}",
            "score": 1, "val_score": val, "eval_level": 100,
            "generation": step, "island": "haiku", "step": step,
            "created_at": created_at,
        }

    seed = variant("haiku-seed", 0, "2026-01-01T00:00:00", 9.0)
    if001 = variant("haiku-if-001", 1, "2026-01-01T01:00:00", 5.0)
    if002 = variant("haiku-if-002", 2, "2026-01-01T02:00:00", 1.0)
    # Insertion order is not selection order: the newest row is in the middle
    # so a test that passed by accident on list position would fail here.
    pop = [if001, seed, if002]

    patches = {
        "get_population": MagicMock(return_value=pop),
        "tournament": MagicMock(side_effect=AssertionError(
            "tournament must not be called under parent_selection='latest'"
        )),
        "insert_variant": MagicMock(),
        "insert_eval_result": MagicMock(),
        "update_variant": MagicMock(),
        "get_variant": MagicMock(return_value={**if002, "score": 1}),
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
        "stage_results_for_builder": MagicMock(),
        "git_checkout_variant": MagicMock(),
        "git_tag_variant": MagicMock(),
        "revert_out_of_scope": MagicMock(),
        "get_next_variant_num": MagicMock(return_value=3),
        "_recoverable_tagged_child": MagicMock(return_value=None),
    }
    started = [patch(f"agent_s4.operators.mutation.{name}", val)
               for name, val in patches.items()]
    for p in started:
        p.start()
    try:
        Mutation().apply(state)
    finally:
        for p in started:
            p.stop()

    patches["tournament"].assert_not_called()
    patches["git_checkout_variant"].assert_any_call(
        str(state.clone_dir), "evol/haiku/if-002",
    )
    patches["stage_results_for_builder"].assert_called_once()
    staged_parent = patches["stage_results_for_builder"].call_args[0][0]
    assert staged_parent == "haiku-if-002", (
        f"linear route must extend from the newest variant, staged "
        f"{staged_parent!r} instead"
    )
    args, _ = patches["insert_variant"].call_args
    assert args[1]["parent_id"] == "haiku-if-002"
