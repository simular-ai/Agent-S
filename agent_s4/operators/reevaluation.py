"""Repeat validation for the current leaders without creating interfaces.

Each island pins its top-k selection, evaluates those immutable commits in
parallel, then commits their cumulative validation means together. A retry
uses the same round and cache directories, never the original evaluation.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

from agent_s4.costs import record_eval_costs
from agent_s4.db import (
    complete_reevaluation_round,
    get_population,
    get_reevaluation_round,
    get_val_task_ids,
    start_reevaluation_round,
)
from agent_s4.harness import resolve_evaluator_harness
from agent_s4.operators.base import Operator
from agent_s4.scoring import canonicalize_full_credit
from agent_s4.state import LoopState
from agent_s4.tournament import _collect_per_task_records
from agent_s4.utils import now_iso

logger = logging.getLogger("agent_s4.operators.reevaluation")


def _git(clone_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(clone_dir), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout.strip()


@contextmanager
def _variant_workspace(clone_dir: Path, base_dir: Path, commit: str, result_dir: Path):
    # Evaluator harnesses find *_Base relative to clone_dir.parent. Keep
    # this detached worktree alongside the island clone, with its own logs.
    with tempfile.TemporaryDirectory(prefix=".validation-reeval-", dir=base_dir) as tmp:
        workspace = Path(tmp)
        added = False
        try:
            _git(clone_dir, "worktree", "add", "--detach", str(workspace), commit)
            added = True
            if not (workspace / "shadow_interface").is_dir():
                raise RuntimeError(f"variant commit {commit} has no shadow_interface")
            yield workspace
        finally:
            try:
                if (workspace / "logs").is_dir():
                    shutil.copytree(
                        workspace / "logs",
                        result_dir / "host_logs",
                        dirs_exist_ok=True,
                    )
            finally:
                if added:
                    _git(clone_dir, "worktree", "remove", "--force", str(workspace))


def _evaluate_candidate(
    state: LoopState, candidate: dict, context: dict, result_dir: Path
) -> dict:
    """Run one candidate in a thread; all database access stays in apply()."""
    evaluator = resolve_evaluator_harness(context["harness"], context["benchmark"])
    result_dir.mkdir(parents=True, exist_ok=True)
    with _variant_workspace(
        state.clone_dir,
        state.base_dir,
        candidate["commit"],
        result_dir,
    ) as workspace:
        out = evaluator.run_tasks(
            clone_dir=workspace,
            shadow_dir=workspace / "shadow_interface",
            result_dir=result_dir,
            task_ids=context["task_ids"],
            model=context["model"],
            region=context["region"],
            timeout=context["timeout"],
            max_steps=context["max_steps"],
            no_time_limit=context["no_time_limit"],
            harness=context["harness"],
            benchmark=context["benchmark"],
        )
    task_ids = context["task_ids"]
    if out.timed_out:
        raise RuntimeError(f"Validation reevaluation timed out for {candidate['id']}")
    returned_ids = [r.get("task_id") for r in out.results]
    if (
        out.total != len(task_ids)
        or len(returned_ids) != len(task_ids)
        or set(returned_ids) != set(task_ids)
    ):
        raise RuntimeError(f"Incomplete validation reevaluation for {candidate['id']}")
    # Follow the same fitness policy as initial evaluations: exhausted
    # retries may be imputed zeros when the harness marks them valid.
    if any(not r.get("valid_for_fitness", True) for r in out.results):
        raise RuntimeError(
            f"Validation reevaluation for {candidate['id']} contains infrastructure "
            "failures; leaving representative scores unchanged"
        )
    records = _collect_per_task_records(
        state.clone_dir,
        candidate["id"],
        task_ids,
        context["model"],
        benchmark=context["benchmark"],
        result_dir=result_dir,
    )
    for result in out.results:
        reported = float(result["score"])
        if context["benchmark"] == "osworld_v2":
            reported = canonicalize_full_credit(reported)
        actual = records[result["task_id"]]["raw_score"]
        if not math.isfinite(reported) or not math.isclose(
            reported, actual, abs_tol=1e-9
        ):
            raise RuntimeError(
                f"Validation summary disagrees with task results for {candidate['id']}"
            )
    return {
        "variant_id": candidate["id"],
        "score": sum(record["fitness_score"] for record in records.values()),
        "records": records,
        "evaluated_at": now_iso(),
        "result_dir": str(result_dir),
    }


class Reevaluation(Operator):
    kind: ClassVar[str] = "reevaluation"

    def apply(self, state: LoopState) -> None:
        conn = state.conn
        island = state.current_island
        round_ = get_reevaluation_round(conn, island, state.step)
        if round_ is not None and round_["status"] == "complete":
            return
        task_ids = sorted(get_val_task_ids(conn))
        if not task_ids:
            raise ValueError(
                "validation reevaluation requires a nonempty validation set"
            )
        island_cfg = state.current_config()
        context = {
            "task_ids": task_ids,
            "model": state.config.resolve_eval_model(island_cfg),
            "harness": state.config.resolve_evaluator_harness_key(island_cfg),
            "benchmark": state.config.benchmark,
            "region": state.config.region,
            "timeout": state.config.eval_timeout,
            "max_steps": state.config.eval_max_steps,
            "no_time_limit": state.config.eval_no_time_limit,
            "top_k": state.config.reevaluation_top_k,
        }
        if round_ is None:
            population = [
                v for v in get_population(conn, island) if v["val_score"] is not None
            ]
            if any(
                not math.isfinite(v["val_score"])
                or not 0 <= v["val_score"] <= len(task_ids)
                for v in population
            ):
                raise ValueError("population contains invalid validation scores")
            # Reproducible ties; rank only by the representative val score.
            population.sort(key=lambda v: (-v["val_score"], v["id"]))
            candidates = [
                {
                    "id": v["id"],
                    "tag": v["tag"],
                    "val_score": v["val_score"],
                    "commit": _git(
                        state.clone_dir,
                        "rev-parse",
                        "--verify",
                        f"{v['tag']}^{{commit}}",
                    ),
                }
                for v in population[: state.config.reevaluation_top_k]
            ]
            round_ = start_reevaluation_round(
                conn,
                island,
                state.step,
                candidates,
                context,
                now_iso(),
            )
        if round_["context"] != context:
            raise ValueError(
                "pending validation reevaluation settings changed; restore its saved config"
            )
        candidates = round_["candidates"]
        if not candidates:
            complete_reevaluation_round(conn, island, state.step, [], now_iso())
            return

        logger.info(
            "[after step=%d] reevaluating %s on %d validation tasks each",
            state.step,
            [v["id"] for v in candidates],
            len(task_ids),
        )
        outcomes = []
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            futures = {
                pool.submit(
                    _evaluate_candidate,
                    state,
                    candidate,
                    context,
                    state.eval_cache_dir
                    / candidate["id"]
                    / "validation_reevaluations"
                    / f"step_{state.step:06d}",
                ): candidate["id"]
                for candidate in candidates
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    outcome = future.result()
                    outcomes.append(outcome)
                    # Repeats still incur real cost, but never consume a step.
                    # The round's stable timestamp makes resumed recording idempotent.
                    record_eval_costs(
                        conn,
                        variant_id=outcome["variant_id"],
                        island=island,
                        clone_dir=state.clone_dir,
                        task_ids=task_ids,
                        event_type="validation_reevaluation",
                        model_fallback=context["model"],
                        timestamp=round_["started_at"],
                        benchmark=context["benchmark"],
                        result_dir=outcome["result_dir"],
                    )
                except Exception as exc:
                    errors.append((futures[future], exc))
        if errors:
            failed = ", ".join(variant_id for variant_id, _ in errors)
            raise RuntimeError(
                f"Validation reevaluation failed for {failed}; scores unchanged. "
                "Resume retries the pinned round using its completed result caches."
            ) from errors[0][1]
        means = complete_reevaluation_round(
            conn,
            island,
            state.step,
            outcomes,
            now_iso(),
        )
        logger.info(
            "[after step=%d] representative validation scores: %s", state.step, means
        )
