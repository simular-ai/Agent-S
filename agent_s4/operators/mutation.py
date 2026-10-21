"""Mutation operator.

v2 form (post island-evolution-v2 refactor): one call to apply() picks a
parent via tournament, runs the builder (via BuilderHarness), evaluates
the child once on the canonical train set + val set (via
EvaluatorHarness), and inserts the variant. The historical tier-cascade
promotion block is gone — every variant is evaluated on the same
train_set_file (see ExperimentConfig.train_set_file).

The builder + evaluator are resolved per-island via the harness registry
so a new harness can be registered without touching this
file.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import ClassVar

from agent_s4.costs import record_builder_cost, record_eval_costs
from agent_s4.db import (
    append_history,
    get_population,
    get_variant,
    insert_eval_result,
    insert_variant,
    recompute_score_from_train_set,
    recompute_val_score,
    update_variant,
)
from agent_s4.harness import (
    resolve_builder_harness,
    resolve_evaluator_harness,
)
from agent_s4.operators.base import Operator
from agent_s4.scratch_vm import (
    provision_scratch_vm,
    scratch_vm_env,
    stage_builder_scratch_bridge,
    teardown_scratch_vm,
)
from agent_s4.state import LoopState
from agent_s4.tournament import _collect_per_task_records, tournament

from agent_s4.benchmark_spec import ns
from agent_s4.constants import (
    BUILDER_LOGS_DIR,
    EVAL_SETS_DIR,
)
from agent_s4.git_utils import (
    git,
    git_checkout_variant,
    git_tag_variant,
    revert_out_of_scope,
)
from agent_s4.utils import (
    _load_val_tasks,
    get_next_variant_num,
    now_iso,
    stage_results_for_builder,
)

logger = logging.getLogger("agent_s4.operators.mutation")


class BuilderCrashError(Exception):
    """Raised by Mutation.apply() when the builder returns crashed=True
    (auth failure, rate-limit, short-exit). The runner catches this at
    the top level, logs, and exits the pane without advancing cadence."""


_MUTATION_BUILDER_PROMPT = (
    "Mutation step. Read CLAUDE.md for instructions. "
    "Study eval_results/ trajectories to identify difficulties and opportunities. Interpret scores according to CLAUDE.md. "
    "Improve shadow_interface/. Write observations to notebook.jsonl."
)


def _recoverable_tagged_child(
    conn,
    island_name: str,
    clone_dir: str,
    population: list[dict],
    parent: dict | None = None,
) -> dict | None:
    """Return the latest fully-built child left by a pre-insert crash.

    Mutation tags are written only after a successful builder run and its
    build-cost event, but the variant row is inserted only after evaluation.
    A VM-allocation or evaluator crash in that gap therefore leaves a tag and
    build-cost event without a population row.  Reusing that exact tag avoids
    paying for (and changing) a second builder mutation on dispatch resume.

    Only the highest numbered tag is eligible.  This deliberately ignores old
    lower-numbered audit tags once a later variant exists.  The exact commit
    subject also binds the orphan to the parent selected for this replay.
    """
    next_num = get_next_variant_num({"population": population}, clone_dir)
    candidate_num = next_num - 1
    if candidate_num < 1:
        return None

    child_id = f"{island_name}-if-{candidate_num:03d}"
    if any(v["id"] == child_id for v in population):
        return None

    tag = f"evol/{island_name}/if-{candidate_num:03d}"
    subject, rc = git(["show", "-s", "--format=%s", tag], cwd=clone_dir)
    subject = subject.strip()
    subject_prefix = f"{tag}: mutation of "
    if rc != 0 or not subject.startswith(subject_prefix):
        return None

    parent_id = subject[len(subject_prefix) :]
    recovered_parent = next(
        (variant for variant in population if variant["id"] == parent_id),
        None,
    )
    if recovered_parent is None:
        return None
    if parent is not None and parent["id"] != parent_id:
        return None

    row = conn.execute(
        """
        SELECT cost_usd
        FROM cost_events
        WHERE event_type = 'build' AND island = ? AND variant_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (island_name, child_id),
    ).fetchone()
    if row is None:
        return None

    return {
        "variant_num": candidate_num,
        "child_id": child_id,
        "tag": tag,
        "builder_cost": float(row[0]),
        "parent": recovered_parent,
    }


class Mutation(Operator):
    kind: ClassVar[str] = "mutation"

    def __init__(self) -> None:
        pass

    def apply(self, state: LoopState) -> None:
        island_name = state.current_island
        island_cfg = state.current_config()
        clone_dir = str(state.clone_dir)
        base_dir = str(state.base_dir)
        conn = state.conn

        # 0. Clear stale results so builder doesn't read prior eval
        for stale in ("results", "eval_results"):
            p = os.path.join(clone_dir, stale)
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)

        pop = get_population(conn, island_name)
        # Eligibility filter mirrors the configured selection signal:
        # under "val", a candidate must have val_score populated; under
        # "train", today's score-not-NULL filter applies. Both columns
        # are populated atomically by the same eval call, so in steady
        # state these filters select the same set.
        if state.config.selection_signal == "val":
            scored = [v for v in pop if v["val_score"] is not None]
        else:
            scored = [v for v in pop if v["score"] is not None]
        if not scored:
            logger.error("No scored variants in %s — cannot mutate", island_name)
            return

        # 1. Recover a fully-built pre-insert child before selecting a new
        # parent.  Re-running the tournament first is not safe: its random
        # draw can choose a different parent after a crash, which makes the
        # valid orphan tag invisible and needlessly launches another builder.
        recovery = _recoverable_tagged_child(
            conn,
            island_name,
            clone_dir,
            pop,
        )

        # Otherwise tournament-select a parent for a new mutation.
        eval_model = state.config.resolve_eval_model(island_cfg)
        if recovery is not None and recovery.get("parent") is not None:
            parent = recovery["parent"]
        elif state.config.parent_selection == "latest":
            # Linear route: the chain always extends from the newest scored
            # variant, whatever it scored. Ordering by step alone is not
            # enough -- the seed row predates the step column and carries its
            # NOT NULL DEFAULT 0, and a step that produced no scored child
            # leaves a gap -- so created_at and id break ties, and both rise
            # monotonically with insertion.
            parent = max(
                scored,
                key=lambda v: (v["step"] or 0, v["created_at"] or "", v["id"]),
            )
        else:
            winners = tournament(
                conn,
                scored,
                k=2,
                t=1,
                clone_dir=clone_dir,
                eval_model=eval_model,
                eval_timeout=state.config.eval_timeout,
                region=state.config.region,
                selection_signal=state.config.selection_signal,
                benchmark=state.config.benchmark,
            )
            parent = winners[0]
        logger.info(
            "Selected %s (step=%s, score=%s, val_score=%s) via %s",
            parent["id"],
            parent.get("step"),
            parent["score"],
            parent.get("val_score"),
            state.config.parent_selection,
        )

        if recovery is not None:
            variant_num = recovery["variant_num"]
            child_id = recovery["child_id"]
            tag = recovery["tag"]
            builder_cost = recovery["builder_cost"]
            logger.warning(
                "Resume: reusing tagged child %s from a pre-insert crash; "
                "skipping builder and continuing evaluation",
                child_id,
            )
            git_checkout_variant(clone_dir, tag)
        else:
            # 2. Checkout parent + stage results
            git_checkout_variant(clone_dir, parent["tag"])
            stage_results_for_builder(
                parent["id"],
                base_dir,
                clone_dir,
                conn,
                benchmark=state.config.benchmark,
            )

            # 3. Build child (with scratch VM for exploration)
            variant_num = get_next_variant_num({"population": pop}, clone_dir)
            child_id = f"{island_name}-if-{variant_num:03d}"
            logs_dir = os.path.join(
                base_dir,
                ns(BUILDER_LOGS_DIR, state.config.benchmark),
            )
            logger.info("Building %s (parent=%s)", child_id, parent["id"])

            if state.config.scratch_vm_enabled:
                stage_builder_scratch_bridge(clone_dir)
                scratch_handle = provision_scratch_vm(
                    clone_dir,
                    region=state.config.region,
                    benchmark=state.config.benchmark,
                )
                if scratch_handle is None:
                    append_history(
                        conn,
                        {
                            "type": "scratch_vm_provision_failed",
                            "island": island_name,
                            "parent": parent["id"],
                            "child": child_id,
                            "timestamp": now_iso(),
                        },
                    )
                    if state.config.benchmark == "osworld_v2":
                        raise RuntimeError(
                            "OSWorld V2 scratch VM is enabled but provisioning "
                            "returned no handle; refusing a silent "
                            "trajectory-only run"
                        )
            else:
                logger.info("Scratch VM disabled via config — builder runs without VM")
                scratch_handle = None
            builder_env_extra = (
                scratch_vm_env(scratch_handle) if scratch_handle else None
            )

            builder_harness_key = state.config.resolve_builder_harness_key(
                island_cfg,
            )
            builder = resolve_builder_harness(builder_harness_key)
            builder_model = state.config.resolve_builder_model(island_cfg)

            try:
                result = builder.launch(
                    clone_dir=state.clone_dir,
                    prompt=_MUTATION_BUILDER_PROMPT,
                    model=builder_model,
                    timeout=(
                        None
                        if state.config.builder_no_time_limit
                        else state.config.builder_timeout
                    ),
                    env_extra=builder_env_extra,
                    variant_id=child_id,
                    logs_dir=logs_dir,
                    log_name=f"{child_id}_mutation",
                    builder_bin=state.config.resolve_builder_bin(island_cfg),
                )
            finally:
                teardown_scratch_vm(scratch_handle)

            # Fail-fast on builder crash BEFORE any per-child-id DB writes.
            if result.crashed:
                revert_out_of_scope(clone_dir)
                raise BuilderCrashError(
                    f"Builder for {child_id} crashed "
                    f"(exit_code={result.exit_code}). No variant row inserted."
                )

            record_builder_cost(
                conn,
                variant_id=child_id,
                island=island_name,
                log_path=str(result.log_path),
                model_fallback=builder_model,
                timestamp=now_iso(),
            )
            builder_cost = result.cost_usd
            revert_out_of_scope(clone_dir)

            # 4. Tag child
            tag = f"evol/{island_name}/if-{variant_num:03d}"
            msg = f"{tag}: mutation of {parent['id']}"
            git_tag_variant(clone_dir, tag, msg)

        # 5. Eval child on the canonical train set + val set (single
        # parallel call via evaluator harness). v2 drops tier cascade —
        # every variant is evaluated on the same train_set_file, so
        # scores are directly comparable across islands and generations.
        # VMs are 1:1 with task count, so 100 train + 50 val = 150
        # concurrent VMs.
        train_set_file = state.config.train_set_file
        train_path = os.path.join(base_dir, EVAL_SETS_DIR, train_set_file)
        with open(train_path) as f:
            train_tasks = json.load(f)
        val_tasks = _load_val_tasks(base_dir, state.config.val_set_file)
        all_tasks = train_tasks + val_tasks
        logger.info(
            "Evaluating %s on %s: %d train + %d val = %d tasks",
            child_id,
            train_set_file,
            len(train_tasks),
            len(val_tasks),
            len(all_tasks),
        )

        evaluator_key = state.config.resolve_evaluator_harness_key(island_cfg)
        evaluator = resolve_evaluator_harness(evaluator_key, state.config.benchmark)
        result_dir = os.path.join(
            base_dir,
            state.eval_cache_dir.name,
            child_id,
        )
        eval_out = evaluator.run_tasks(
            clone_dir=state.clone_dir,
            shadow_dir=state.clone_dir / "shadow_interface",
            result_dir=result_dir,
            task_ids=all_tasks,
            model=eval_model,
            region=state.config.region,
            timeout=state.config.eval_timeout,
            max_steps=state.config.eval_max_steps,
            no_time_limit=state.config.eval_no_time_limit,
            harness=evaluator_key,
            benchmark=state.config.benchmark,
        )
        if eval_out.timed_out:
            raise RuntimeError(
                f"Eval timed out for {child_id}; refusing partial zero fitness"
            )

        # 6. Insert child and per-task results.
        # `step` ties the variant row to the dispatch step that created
        # it — the driver's resume protocol uses this to find and delete
        # orphan variants from a crashed step. `eval_level` is retained
        # as a deprecated column (back-compat row width); v2 doesn't use
        # tier promotion, so it's pinned to 0.
        insert_variant(
            conn,
            {
                "id": child_id,
                "island": island_name,
                "tag": tag,
                "parent_id": parent["id"],
                "parent_b_id": None,
                "origin": "mutation",
                "generation": parent["generation"] + 1,
                "score": None,
                "eval_level": 0,
                "step": state.step,
                "builder_cost": builder_cost,
                "created_at": now_iso(),
            },
        )
        per_task = _collect_per_task_records(
            clone_dir,
            child_id,
            all_tasks,
            eval_model,
            benchmark=state.config.benchmark,
        )
        now = now_iso()
        for tid, record in per_task.items():
            insert_eval_result(
                conn,
                child_id,
                tid,
                record["passed"],
                now,
                raw_score=record["raw_score"],
                fitness_score=record["fitness_score"],
                details_json=record["details_json"],
                failure_type=record["failure_type"],
            )
        record_eval_costs(
            conn,
            variant_id=child_id,
            island=island_name,
            clone_dir=clone_dir,
            task_ids=all_tasks,
            event_type="eval_task",
            model_fallback=eval_model,
            timestamp=now,
            benchmark=state.config.benchmark,
        )
        # Train score counts passes restricted to the canonical train
        # task set; val score joins against val_set. Both columns are
        # populated atomically from a single parallel eval call.
        score = recompute_score_from_train_set(conn, child_id, train_tasks)
        val_score = recompute_val_score(conn, child_id)
        update_variant(conn, child_id, score=score, val_score=val_score)
        logger.info(
            "%s scored %.3f/%d (val: %.3f/%d)",
            child_id,
            score,
            len(train_tasks),
            val_score,
            len(val_tasks),
        )

        append_history(
            conn,
            {
                "type": "mutation",
                "island": island_name,
                "parent": parent["id"],
                "child": child_id,
                "score": get_variant(conn, child_id)["score"],
                "cost": builder_cost,
                "details": json.dumps({"train_set_file": train_set_file}),
                "timestamp": now_iso(),
            },
        )
