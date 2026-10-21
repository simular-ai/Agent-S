"""Crossover operator (intra-island, v2).

Both parents come from the same island's population. Recently-arrived
migrants (re-evaluated under this island's evaluator) participate
naturally — that's the cross-environment synthesis channel.

The builder gets a scratch VM in v2 (was missing in v1). Prompt
composition follows the additive pattern: the base crossover prompt
plus the optional scratch_vm_addendum when the VM is enabled.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import ClassVar

from agent_s4.paths import prompt_path
from agent_s4.costs import record_builder_cost, record_eval_costs
from agent_s4.db import (
    append_history,
    get_population,
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
from agent_s4.constants import BUILDER_LOGS_DIR, EVAL_SETS_DIR
from agent_s4.git_utils import git_archive_variant, git_tag_variant
from agent_s4.utils import (
    _load_val_tasks,
    get_next_variant_num,
    now_iso,
    stage_results_for_builder,
)


def _assemble_crossover_prompt(
    scratch_vm_enabled: bool, benchmark: str = "osworld"
) -> str:
    """Compose the crossover CLAUDE.md: base prompt + optional scratch
    VM addendum. Mirrors the additive composition pattern from
    initialize_clone — config-gated sections appended, never forked. The
    scratch VM addendum is benchmark-specific (ScienceBoard launches its apps
    via execute_python_command; OSWorld uses a SetupController)."""
    text = prompt_path("crossover.md").read_text().rstrip() + "\n"
    if scratch_vm_enabled:
        scratch_fname = (
            "scratch_vm_addendum_scienceboard.md"
            if benchmark == "scienceboard"
            else "scratch_vm_addendum.md"
        )
        text += "\n" + prompt_path(scratch_fname).read_text().rstrip() + "\n"
    return text


logger = logging.getLogger("agent_s4.operators.crossover")


class Crossover(Operator):
    kind: ClassVar[str] = "crossover"

    def __init__(self) -> None:
        pass

    def apply(self, state: LoopState) -> None:
        island_name = state.current_island
        island_cfg = state.current_config()
        clone_dir = str(state.clone_dir)
        base_dir = str(state.base_dir)
        conn = state.conn

        signal = state.config.selection_signal
        score_key = "val_score" if signal == "val" else "score"
        pop = [v for v in get_population(conn, island_name) if v[score_key] is not None]
        if len(pop) < 2:
            logger.info(
                "Crossover skipped: %s has %d scored variants (<2)",
                island_name,
                len(pop),
            )
            return

        eval_model = state.config.resolve_eval_model(island_cfg)

        # Two intra-island tournament draws. Excluding parent_a from the
        # second draw guarantees distinct parents — otherwise a small
        # population could pick the same variant twice and the
        # "crossover" degenerates into a noisy mutation.
        pa_list = tournament(
            conn,
            pop,
            k=2,
            t=1,
            clone_dir=clone_dir,
            eval_model=eval_model,
            eval_timeout=state.config.eval_timeout,
            region=state.config.region,
            selection_signal=signal,
            benchmark=state.config.benchmark,
        )
        parent_a = pa_list[0]
        remaining = [v for v in pop if v["id"] != parent_a["id"]]
        pb_list = tournament(
            conn,
            remaining,
            k=2,
            t=1,
            clone_dir=clone_dir,
            eval_model=eval_model,
            eval_timeout=state.config.eval_timeout,
            region=state.config.region,
            selection_signal=signal,
            benchmark=state.config.benchmark,
        )
        parent_b = pb_list[0]

        tmp_a = os.path.join(base_dir, f"_tmp_xover_a_{island_name}")
        tmp_b = os.path.join(base_dir, f"_tmp_xover_b_{island_name}")
        pa_dir = os.path.join(clone_dir, "crossover_parent_a")
        pb_dir = os.path.join(clone_dir, "crossover_parent_b")
        traj_a = os.path.join(clone_dir, "eval_results_parent_a")
        traj_b = os.path.join(clone_dir, "eval_results_parent_b")
        scratch_handle = None

        try:
            git_archive_variant(clone_dir, parent_a["tag"], tmp_a)
            git_archive_variant(clone_dir, parent_b["tag"], tmp_b)
            for d in (pa_dir, pb_dir):
                if os.path.exists(d):
                    shutil.rmtree(d)
            shutil.copytree(os.path.join(tmp_a, "shadow_interface"), pa_dir)
            shutil.copytree(os.path.join(tmp_b, "shadow_interface"), pb_dir)

            # Stage each parent's TRAIN-task trajectories alongside its code.
            # stage_results_for_builder is the single val-safe boundary (it
            # drops val task dirs + latest_summary.json), so only train traces
            # reach the builder. Sibling dirs mirror mutation's eval_results/.
            stage_results_for_builder(
                parent_a["id"],
                base_dir,
                clone_dir,
                conn,
                benchmark=state.config.benchmark,
                dest_name=os.path.basename(traj_a),
            )
            stage_results_for_builder(
                parent_b["id"],
                base_dir,
                clone_dir,
                conn,
                benchmark=state.config.benchmark,
                dest_name=os.path.basename(traj_b),
            )

            # Stage parent_a's code as the working tree starting point so
            # the builder has a coherent shadow_interface/ to diff against
            # while combining the two parents.
            dst = os.path.join(clone_dir, "shadow_interface")
            if os.path.exists(dst):
                for item in os.listdir(dst):
                    if item == ".git":
                        continue
                    p = os.path.join(dst, item)
                    shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            else:
                os.makedirs(dst, exist_ok=True)
            for item in os.listdir(pa_dir):
                s = os.path.join(pa_dir, item)
                d = os.path.join(dst, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)

            # Write crossover prompt (with scratch VM addendum if enabled).
            crossover_text = _assemble_crossover_prompt(
                state.config.scratch_vm_enabled,
                benchmark=state.config.benchmark,
            )
            with open(os.path.join(clone_dir, "CLAUDE.md"), "w") as f:
                f.write(crossover_text)

            # Provision scratch VM (NEW in v2). Soft-degrades: a None
            # handle means the builder runs without a VM (same fallback
            # contract as Mutation.apply).
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
                            "parent": parent_a["id"],
                            "child": None,
                            "timestamp": now_iso(),
                        },
                    )
                    if state.config.benchmark == "osworld_v2":
                        raise RuntimeError(
                            "OSWorld V2 scratch VM is enabled but provisioning "
                            "returned no handle; refusing a silent trajectory-only run"
                        )
            else:
                logger.info(
                    "Scratch VM disabled via config — crossover builder runs without VM"
                )
            builder_env_extra = (
                scratch_vm_env(scratch_handle) if scratch_handle else None
            )

            pop_now = get_population(conn, island_name)
            variant_num = get_next_variant_num({"population": pop_now}, clone_dir)
            child_id = f"{island_name}-if-{variant_num:03d}"
            logs_dir = os.path.join(
                base_dir, ns(BUILDER_LOGS_DIR, state.config.benchmark)
            )

            # Parents are SELECTED by the configured signal (val) above, but
            # the builder is shown TRAIN scores only — val is held out from
            # what the builder sees so its synthesis can't be conditioned on
            # the held-out signal. (Deliberate: do not switch these to val_score.)
            score_a = parent_a["score"] or 0
            score_b = parent_b["score"] or 0
            crossover_prompt = (
                f"Crossover step. Read CLAUDE.md. "
                f"Parent A ({parent_a['id']}, train={score_a}): code in "
                f"crossover_parent_a/, its train-task trajectories in "
                f"eval_results_parent_a/. "
                f"Parent B ({parent_b['id']}, train={score_b}): code in "
                f"crossover_parent_b/, its train-task trajectories in "
                f"eval_results_parent_b/. "
                "These trajectories show how each parent's interface actually "
                "behaved on the training tasks — where each succeeded and "
                "failed. Study both, then combine the best of both into "
                "shadow_interface/."
            )

            builder = resolve_builder_harness(
                state.config.resolve_builder_harness_key(island_cfg)
            )
            builder_model = state.config.resolve_builder_model(island_cfg)
            result = builder.launch(
                clone_dir=state.clone_dir,
                prompt=crossover_prompt,
                model=builder_model,
                timeout=(
                    None
                    if state.config.builder_no_time_limit
                    else state.config.builder_timeout
                ),
                env_extra=builder_env_extra,
                variant_id=child_id,
                logs_dir=logs_dir,
                log_name=f"{child_id}_crossover",
                builder_bin=state.config.resolve_builder_bin(island_cfg),
            )
            if result.crashed or result.exit_code != 0:
                raise RuntimeError(
                    f"Builder for {child_id} failed "
                    f"(exit_code={result.exit_code}); refusing crossover evaluation"
                )
            record_builder_cost(
                conn,
                variant_id=child_id,
                island=island_name,
                log_path=str(result.log_path),
                model_fallback=builder_model,
                timestamp=now_iso(),
            )

            tag = f"evol/{island_name}/if-{variant_num:03d}"
            git_tag_variant(
                clone_dir,
                tag,
                f"{tag}: crossover of {parent_a['id']} x {parent_b['id']}",
            )

            # v2: single canonical train set + val set, no tier cascade.
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
                    f"Crossover eval timed out for {child_id}; refusing partial zero fitness"
                )

            insert_variant(
                conn,
                {
                    "id": child_id,
                    "island": island_name,
                    "tag": tag,
                    "parent_id": parent_a["id"],
                    "parent_b_id": parent_b["id"],
                    "origin": "crossover",
                    "generation": max(parent_a["generation"], parent_b["generation"])
                    + 1,
                    "score": None,
                    "eval_level": 0,
                    "step": state.step,
                    "builder_cost": result.cost_usd,
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
                event_type="crossover_eval_task",
                model_fallback=eval_model,
                timestamp=now,
                benchmark=state.config.benchmark,
            )

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
                    "type": "crossover",
                    "island": island_name,
                    "parent": parent_a["id"],
                    "child": child_id,
                    "score": score,
                    "cost": result.cost_usd,
                    "details": json.dumps({"parent_b": parent_b["id"]}),
                    "timestamp": now_iso(),
                },
            )
            logger.info(
                "Crossover %s x %s -> %s (score=%.3f, val=%.3f)",
                parent_a["id"],
                parent_b["id"],
                child_id,
                score,
                val_score,
            )
        finally:
            # teardown_scratch_vm is None-safe (per scratch_vm.py).
            teardown_scratch_vm(scratch_handle)
            for d in (tmp_a, tmp_b, pa_dir, pb_dir, traj_a, traj_b):
                if os.path.exists(d):
                    shutil.rmtree(d)
            # Restore the global CLAUDE.md from the prompt template. If
            # the prompt file is gone (someone deleted it mid-run) or the
            # write fails, log and continue — the outer exception (if any)
            # is more interesting to surface than a CLAUDE.md restore
            # failure.
            try:
                prompt_text = Path(state.config.prompt_template).read_text()
                tmp_path = os.path.join(clone_dir, "CLAUDE.md.restoring")
                with open(tmp_path, "w") as f:
                    f.write(prompt_text)
                os.replace(tmp_path, os.path.join(clone_dir, "CLAUDE.md"))
            except Exception as e:
                logger.warning(
                    "Crossover finally: failed to restore CLAUDE.md (%s). "
                    "Next builder on this island may see stale crossover prompt.",
                    e,
                )
