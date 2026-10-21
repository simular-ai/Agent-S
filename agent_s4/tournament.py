"""Tournament selection for island evolution v2.

Ranks a random pool by stored val_score (default) or train score. All
candidates share a canonical eval set — there is no per-candidate
fill_eval cascade.
"""

import logging
import json
import math
import os
import random

from agent_s4.benchmark_spec import ns
from agent_s4.constants import EVAL_CACHE_DIR
from agent_s4.scoring import canonicalize_full_credit, is_binary_success

logger = logging.getLogger("island")


def _collect_per_task_records(
    clone_dir, variant_id, task_ids, eval_model, *, benchmark="osworld", result_dir=None
):
    """Read raw score plus rich V2 result payloads from an eval cache."""
    base_dir = os.path.dirname(os.path.abspath(clone_dir))
    if result_dir is None:
        result_dir = os.path.join(base_dir, ns(EVAL_CACHE_DIR, benchmark), variant_id)
    cache = os.path.join(result_dir, "shadow_eval", "screenshot", eval_model)
    out = {}
    for tid in task_ids:
        task_dir = os.path.join(cache, tid)
        result_path = os.path.join(task_dir, "result.txt")
        raw_score = 0.0
        failure_type = None
        if os.path.exists(result_path):
            try:
                with open(result_path) as f:
                    raw_score = float(f.read().strip())
                if benchmark == "osworld_v2":
                    raw_score = canonicalize_full_credit(raw_score)
                if not math.isfinite(raw_score) or not 0.0 <= raw_score <= 1.0:
                    failure_type = "invalid_result"
            except (OSError, ValueError):
                failure_type = "invalid_result"
        else:
            failure_type = "missing_result"
        details_json = None
        details_path = os.path.join(task_dir, "result.json")
        if os.path.exists(details_path):
            try:
                with open(details_path) as f:
                    details = json.load(f)
                details_json = json.dumps(details, ensure_ascii=False, default=str)
            except (OSError, ValueError):
                failure_type = failure_type or "invalid_result_details"
        out[tid] = {
            "raw_score": raw_score,
            "passed": int(is_binary_success(raw_score, benchmark)),
            "fitness_score": (
                raw_score
                if benchmark == "osworld_v2"
                else float(is_binary_success(raw_score, benchmark))
            ),
            "details_json": details_json,
            "failure_type": failure_type,
        }
    failures = {
        tid: record["failure_type"]
        for tid, record in out.items()
        if record["failure_type"]
    }
    if failures:
        raise RuntimeError(
            "Evaluation cache is incomplete or invalid; refusing to turn "
            f"infrastructure output into zero fitness: {failures}"
        )
    return out


def _collect_per_task(
    clone_dir, variant_id, task_ids, eval_model, *, benchmark="osworld"
):
    """Read per-task result.txt files after an eval run.

    eval_model must match the --model flag passed to agent_s4.eval.osworld,
    because the eval writer's output path embeds the model name
    (eval/osworld.py:335-338). Passing the wrong name silently returns all
    zeros. Previously hardcoded to 'haiku' here, which was a latent bug for
    any non-haiku eval model.

    Returns dict of {task_id: 0 or 1}. Missing/invalid artifacts raise because
    they are infrastructure failures, not evidence of task failure.
    """
    records = _collect_per_task_records(
        clone_dir,
        variant_id,
        task_ids,
        eval_model,
        benchmark=benchmark,
    )
    return {tid: record["passed"] for tid, record in records.items()}


def _checkout_variant(clone_dir, tag):
    """Checkout the variant's code into the clone's shadow_interface/."""
    from agent_s4.git_utils import git_checkout_variant

    git_checkout_variant(clone_dir, tag)


# ---- Core API ------------------------------------------------------------


def tournament(
    conn,
    candidates,
    k,
    t,
    clone_dir,
    eval_model,
    eval_timeout,
    region,
    selection_signal="val",
    *,
    benchmark="osworld",
):
    """Select the top `t` candidates from a random sample of `k`, ranked
    by val_score or train score depending on selection_signal.

    No fill_eval cascade — all candidates are evaluated on the same
    canonical task set in v2. eval_model/eval_timeout/region/benchmark are
    kept in the signature for parity with the eval helpers but are unused
    here (selection reads only the DB / in-memory candidate rows, no disk
    I/O). benchmark defaults to 'osworld' so existing callers/tests are
    unaffected.
    """
    if not candidates:
        return []

    if k >= len(candidates):
        pool = list(candidates)
    else:
        pool = random.sample(list(candidates), k)

    if selection_signal == "val":
        ranked = sorted(
            pool,
            key=lambda v: (
                v["val_score"] if v["val_score"] is not None else -1,
                v["score"] if v["score"] is not None else -1,
                random.random(),
            ),
            reverse=True,
        )
    elif selection_signal == "train":
        ranked = sorted(
            pool,
            key=lambda v: (
                v["score"] if v["score"] is not None else -1,
                random.random(),
            ),
            reverse=True,
        )
    else:
        raise ValueError(f"Unknown selection_signal: {selection_signal!r}")

    return ranked[:t]
