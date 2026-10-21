"""Cost event recording helpers.

Parses stream-json logs (builder subprocess) and eval task trajectory files,
inserts rows into the cost_events table. Sync-path hooks in run_island.py,
island_orchestrator.py, and island_tournament.py call these helpers right
after each subprocess completes.
"""

import json
import logging
import os

from agent_s4.benchmark_spec import ns
from agent_s4.constants import EVAL_CACHE_DIR
from agent_s4.db import insert_cost_event

logger = logging.getLogger("island")


def _parse_stream_json(path):
    """Read a stream-json log, return (init_model, result_event) pair.

    init_model is the model from the first `{"type": "system", "subtype": "init"}`
    line, or None if no such line.
    result_event is the last `{"type": "result"}` line as a dict, or None if
    no result event was found (indicates a crash before completion).
    """
    init_model = None
    result_event = None
    if not os.path.exists(path):
        return init_model, result_event
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = d.get("type")
            if t == "system" and d.get("subtype") == "init" and init_model is None:
                init_model = d.get("model")
            elif t == "result":
                result_event = d
    return init_model, result_event


def _parse_opencode_trajectory(path):
    """Aggregate cost and token usage across an OpenCode trajectory.

    OpenCode's --format json emits one `{"type":"step_finish"}` event per
    agent step with per-step `cost` and `tokens.{input,output,reasoning,
    cache.{read,write}}` payload. Unlike Prime's stream-json, which has
    a single terminal `result` event with cumulative usage, OpenCode emits
    no cumulative summary — the caller must sum across step_finish events to get
    per-task totals.

    Returns a dict shaped like {cost_usd, input_tokens, output_tokens,
    cache_read_tokens, cache_creation_tokens} on success, or None if the
    trajectory has no step_finish events (which is indistinguishable from
    a non-OpenCode trajectory).

    Reasoning tokens are folded into output_tokens because cost_events
    has no dedicated reasoning column and OpenRouter bills reasoning as
    output for most providers anyway.
    """
    if not os.path.exists(path):
        return None

    saw_step_finish = False
    total_cost = 0.0
    sum_input = 0
    sum_output_plus_reasoning = 0
    sum_cache_read = 0
    sum_cache_write = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "step_finish":
                continue
            saw_step_finish = True
            part = d.get("part") or {}
            total_cost += float(part.get("cost") or 0)
            tokens = part.get("tokens") or {}
            sum_input += int(tokens.get("input") or 0)
            sum_output_plus_reasoning += int(tokens.get("output") or 0)
            sum_output_plus_reasoning += int(tokens.get("reasoning") or 0)
            cache = tokens.get("cache") or {}
            sum_cache_read += int(cache.get("read") or 0)
            sum_cache_write += int(cache.get("write") or 0)

    if not saw_step_finish:
        return None

    return {
        "cost_usd": total_cost,
        "input_tokens": sum_input,
        "output_tokens": sum_output_plus_reasoning,
        "cache_read_tokens": sum_cache_read,
        "cache_creation_tokens": sum_cache_write,
    }


def _extract_usage(result_event):
    """Pull token fields out of the result event's `usage` dict."""
    usage = (result_event or {}).get("usage") or {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cache_read_tokens": usage.get("cache_read_input_tokens"),
        "cache_creation_tokens": usage.get("cache_creation_input_tokens"),
    }


def record_builder_cost(conn, variant_id, island, log_path, model_fallback, timestamp):
    """Parse a builder stream-json log and insert one `build` cost_event row.

    If the log has no result event (crash), nothing is inserted.
    """
    init_model, result = _parse_stream_json(log_path)
    if result is None:
        logger.warning("record_builder_cost: no result event in %s", log_path)
        return
    usage = _extract_usage(result)
    insert_cost_event(
        conn,
        event_type="build",
        island=island,
        variant_id=variant_id,
        task_id=None,
        model=init_model or model_fallback,
        cost_usd=result.get("total_cost_usd") or 0,
        timestamp=timestamp,
        **usage,
    )


def record_eval_costs(
    conn,
    variant_id,
    island,
    clone_dir,
    task_ids,
    event_type,
    model_fallback,
    timestamp,
    *,
    benchmark="osworld",
    result_dir=None
):
    """Walk each task's claude_trajectory.jsonl and insert one cost_event
    row per task that has a result event.

    Tasks without a trajectory, or whose trajectory has no result event,
    are skipped silently — the task's pass/fail is recorded separately
    in eval_results.

    model_fallback is used both as the cost_event.model when the trajectory
    has no init event AND as the path segment for locating trajectories
    (the eval subprocess writes under .../shadow_eval/screenshot/<model>/...
    — previously hardcoded to 'haiku' here, silently dropping cost events
    for any non-haiku eval model).
    """
    base_dir = os.path.dirname(os.path.abspath(clone_dir))
    if result_dir is None:
        result_dir = os.path.join(base_dir, ns(EVAL_CACHE_DIR, benchmark), variant_id)
    traj_root = os.path.join(result_dir, "shadow_eval", "screenshot", model_fallback)
    for tid in task_ids:
        traj_path = os.path.join(traj_root, tid, "claude_trajectory.jsonl")

        # Prime Agent writes stream-json with a single terminal `result`
        # event containing cumulative usage and total_cost_usd.
        init_model, result = _parse_stream_json(traj_path)
        if result is not None:
            usage = _extract_usage(result)
            cost_usd = result.get("total_cost_usd") or 0
            model = init_model or model_fallback
        else:
            # OpenCode path: aggregate per-step `step_finish` events. The
            # trajectory has no init event identifying the model, so use
            # model_fallback (the path-segment slug).
            opencode_summary = _parse_opencode_trajectory(traj_path)
            if opencode_summary is None:
                # Neither schema matched: skip silently (the task's pass/fail
                # is still recorded separately in eval_results; we just have
                # no cost data to attribute to it).
                continue
            usage = {
                k: opencode_summary[k]
                for k in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_tokens",
                    "cache_creation_tokens",
                )
            }
            cost_usd = opencode_summary["cost_usd"]
            model = model_fallback

        insert_cost_event(
            conn,
            event_type=event_type,
            island=island,
            variant_id=variant_id,
            task_id=tid,
            model=model,
            cost_usd=cost_usd,
            timestamp=timestamp,
            **usage,
        )
