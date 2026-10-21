"""Tests for island_costs recording helpers."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _write_stream_json_log(path, model="claude-opus-4-7",
                           cost=4.78, usage=None):
    """Helper: write a minimal stream-json log file with an init + result event."""
    events = [
        {"type": "system", "subtype": "init", "model": model,
         "session_id": "test-session"},
        {"type": "result",
         "total_cost_usd": cost,
         "usage": usage or {
             "input_tokens": 32,
             "output_tokens": 34531,
             "cache_read_input_tokens": 1709482,
             "cache_creation_input_tokens": 79258,
         }},
    ]
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_record_builder_cost_inserts_row(tmp_db, tmp_path):
    """record_builder_cost parses a stream-json log and inserts one row."""
    from agent_s4.costs import record_builder_cost
    log = tmp_path / "gui-if-001_mutation.jsonl"
    _write_stream_json_log(log, cost=4.78)
    record_builder_cost(
        tmp_db, variant_id="gui-if-001", island="gui",
        log_path=str(log),
        model_fallback="opus",
        timestamp="2026-04-17T00:00:00",
    )
    row = tmp_db.execute(
        "SELECT * FROM cost_events WHERE variant_id = 'gui-if-001'"
    ).fetchone()
    assert row is not None
    assert row["event_type"] == "build"
    assert row["model"] == "claude-opus-4-7"  # pulled from init event, not fallback
    assert abs(row["cost_usd"] - 4.78) < 1e-6
    assert row["input_tokens"] == 32
    assert row["output_tokens"] == 34531
    assert row["cache_read_tokens"] == 1709482
    assert row["cache_creation_tokens"] == 79258


def test_record_builder_cost_uses_fallback_model_when_no_init(tmp_db, tmp_path):
    """If the log lacks an init event, use model_fallback."""
    from agent_s4.costs import record_builder_cost
    log = tmp_path / "x.jsonl"
    with open(log, "w") as f:
        f.write(json.dumps({"type": "result", "total_cost_usd": 1.0,
                            "usage": {}}) + "\n")
    record_builder_cost(
        tmp_db, variant_id="gui-if-001", island="gui",
        log_path=str(log), model_fallback="opus",
        timestamp="2026-04-17T00:00:00",
    )
    row = tmp_db.execute(
        "SELECT model FROM cost_events WHERE variant_id = 'gui-if-001'"
    ).fetchone()
    assert row["model"] == "opus"


def test_record_builder_cost_no_result_event_inserts_nothing(tmp_db, tmp_path):
    """A log with no result event does not crash and inserts nothing."""
    from agent_s4.costs import record_builder_cost
    log = tmp_path / "broken.jsonl"
    with open(log, "w") as f:
        f.write(json.dumps({"type": "system", "subtype": "init",
                            "model": "claude-opus-4-7"}) + "\n")
    record_builder_cost(
        tmp_db, variant_id="gui-if-001", island="gui",
        log_path=str(log), model_fallback="opus",
        timestamp="2026-04-17T00:00:00",
    )
    count = tmp_db.execute(
        "SELECT COUNT(*) FROM cost_events WHERE variant_id = 'gui-if-001'"
    ).fetchone()[0]
    assert count == 0


def test_record_builder_cost_idempotent(tmp_db, tmp_path):
    """Running record_builder_cost twice with same timestamp inserts only once."""
    from agent_s4.costs import record_builder_cost
    log = tmp_path / "x.jsonl"
    _write_stream_json_log(log)
    for _ in range(2):
        record_builder_cost(
            tmp_db, variant_id="gui-if-001", island="gui",
            log_path=str(log), model_fallback="opus",
            timestamp="2026-04-17T00:00:00",
        )
    count = tmp_db.execute(
        "SELECT COUNT(*) FROM cost_events WHERE variant_id = 'gui-if-001'"
    ).fetchone()[0]
    assert count == 1


def _write_task_trajectory(dir_path, task_id, cost=0.03,
                            model="claude-haiku-4-5-20251001"):
    """Helper: write a minimal task trajectory with init + result events."""
    task_dir = dir_path / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    events = [
        {"type": "system", "subtype": "init", "model": model,
         "session_id": f"task-{task_id}"},
        {"type": "result",
         "total_cost_usd": cost,
         "usage": {
             "input_tokens": 100,
             "output_tokens": 500,
             "cache_read_input_tokens": 0,
             "cache_creation_input_tokens": 0,
         }},
    ]
    with open(task_dir / "claude_trajectory.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_record_eval_costs_inserts_one_row_per_task(tmp_db, tmp_path):
    """record_eval_costs walks the trajectory dir and inserts one row per task."""
    from agent_s4.costs import record_eval_costs
    clone_dir = tmp_path / "OSWorld_Evol_gui"
    traj_root = tmp_path / "eval_cache" / "gui-seed" / \
        "shadow_eval" / "screenshot" / "haiku"
    for tid, cost in [("task_a", 0.03), ("task_b", 0.05), ("task_c", 0.04)]:
        _write_task_trajectory(traj_root, tid, cost=cost)
    clone_dir.mkdir()

    record_eval_costs(
        tmp_db, variant_id="gui-seed", island="gui",
        clone_dir=str(clone_dir),
        task_ids=["task_a", "task_b", "task_c"],
        event_type="seed_eval_task",
        model_fallback="haiku",
        timestamp="2026-04-17T00:00:00",
    )
    rows = tmp_db.execute(
        "SELECT task_id, cost_usd, model, event_type FROM cost_events "
        "WHERE variant_id = 'gui-seed' ORDER BY task_id"
    ).fetchall()
    assert len(rows) == 3
    assert rows[0]["task_id"] == "task_a"
    assert abs(rows[0]["cost_usd"] - 0.03) < 1e-6
    assert rows[0]["model"] == "claude-haiku-4-5-20251001"
    assert rows[0]["event_type"] == "seed_eval_task"


def test_record_eval_costs_skips_missing_trajectory(tmp_db, tmp_path):
    """A task without a trajectory file is skipped silently."""
    from agent_s4.costs import record_eval_costs
    clone_dir = tmp_path / "OSWorld_Evol_gui"
    clone_dir.mkdir()
    traj_root = tmp_path / "eval_cache" / "gui-seed" / \
        "shadow_eval" / "screenshot" / "haiku"
    _write_task_trajectory(traj_root, "task_a", cost=0.02)
    # task_b has no trajectory

    record_eval_costs(
        tmp_db, variant_id="gui-seed", island="gui",
        clone_dir=str(clone_dir),
        task_ids=["task_a", "task_b"],
        event_type="eval_task", model_fallback="haiku",
        timestamp="2026-04-17T00:00:00",
    )
    rows = tmp_db.execute(
        "SELECT task_id FROM cost_events WHERE variant_id = 'gui-seed' "
        "ORDER BY task_id"
    ).fetchall()
    assert [r["task_id"] for r in rows] == ["task_a"]


def test_record_eval_costs_idempotent(tmp_db, tmp_path):
    """Running twice with same timestamp does not duplicate rows."""
    from agent_s4.costs import record_eval_costs
    clone_dir = tmp_path / "OSWorld_Evol_gui"
    clone_dir.mkdir()
    traj_root = tmp_path / "eval_cache" / "gui-seed" / \
        "shadow_eval" / "screenshot" / "haiku"
    _write_task_trajectory(traj_root, "task_a", cost=0.02)

    for _ in range(2):
        record_eval_costs(
            tmp_db, variant_id="gui-seed", island="gui",
            clone_dir=str(clone_dir),
            task_ids=["task_a"],
            event_type="eval_task", model_fallback="haiku",
            timestamp="2026-04-17T00:00:00",
        )
    count = tmp_db.execute(
        "SELECT COUNT(*) FROM cost_events WHERE variant_id = 'gui-seed'"
    ).fetchone()[0]
    assert count == 1


# ---------------------------------------------------------------------------
# OpenCode trajectory format (per-step `step_finish` events, no terminal
# `result` event).
# ---------------------------------------------------------------------------

def _write_opencode_task_trajectory(dir_path, task_id, steps):
    """Helper: write an OpenCode-format trajectory with `step_finish` events.

    Each step in `steps` is a tuple
        (cost, input, output, reasoning, cache_read, cache_write).
    """
    task_dir = dir_path / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    events = []
    for cost, inp, out, reasoning, cr, cw in steps:
        events.append({
            "type": "step_finish",
            "timestamp": 1778793976642,
            "sessionID": "ses_test",
            "part": {
                "id": "prt_test",
                "reason": "tool-calls",
                "type": "step-finish",
                "tokens": {
                    "total": inp + out + reasoning,
                    "input": inp,
                    "output": out,
                    "reasoning": reasoning,
                    "cache": {"read": cr, "write": cw},
                },
                "cost": cost,
            },
        })
    with open(task_dir / "claude_trajectory.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_record_eval_costs_opencode_aggregates_step_finish_events(tmp_db, tmp_path):
    """An OpenCode trajectory (no `result` event, only per-step `step_finish`)
    is aggregated: per-step costs sum, per-step token counts sum, and
    reasoning is folded into output_tokens."""
    from agent_s4.costs import record_eval_costs
    clone_dir = tmp_path / "OSWorld_Evol_gui"
    clone_dir.mkdir()
    traj_root = tmp_path / "eval_cache" / "gui-opencode" / \
        "shadow_eval" / "screenshot" / "openrouter" / "qwen" / "qwen3.5-397b-a17b"
    # Three steps with distinct numbers so sums catch arithmetic mistakes.
    _write_opencode_task_trajectory(
        traj_root,
        "task_a",
        steps=[
            # (cost,    input, output, reasoning, cache_read, cache_write)
            (0.001,     100,   10,     5,         0,          0),
            (0.002,     200,   20,     10,        50,         0),
            (0.004,     300,   30,     15,        100,        0),
        ],
    )

    record_eval_costs(
        tmp_db, variant_id="gui-opencode", island="gui",
        clone_dir=str(clone_dir),
        task_ids=["task_a"],
        event_type="eval_task",
        model_fallback="openrouter/qwen/qwen3.5-397b-a17b",
        timestamp="2026-05-14T00:00:00",
    )
    row = tmp_db.execute(
        "SELECT * FROM cost_events WHERE variant_id = 'gui-opencode'"
    ).fetchone()
    assert row is not None
    # cost_usd: 0.001 + 0.002 + 0.004 = 0.007
    assert abs(row["cost_usd"] - 0.007) < 1e-9
    # input_tokens: 100 + 200 + 300 = 600
    assert row["input_tokens"] == 600
    # output_tokens: (10+5) + (20+10) + (30+15) = 90 — reasoning folded in
    assert row["output_tokens"] == 90
    # cache_read_tokens: 0 + 50 + 100 = 150
    assert row["cache_read_tokens"] == 150
    # cache_creation_tokens: all zero in this trajectory
    assert row["cache_creation_tokens"] == 0
    # Model falls back to the slug (OpenCode trajectories have no init event)
    assert row["model"] == "openrouter/qwen/qwen3.5-397b-a17b"


def test_record_eval_costs_opencode_empty_step_finish_skipped(tmp_db, tmp_path):
    """A trajectory file with no step_finish events AND no result event is
    silently skipped (no exception, no row inserted)."""
    from agent_s4.costs import record_eval_costs
    clone_dir = tmp_path / "OSWorld_Evol_gui"
    clone_dir.mkdir()
    traj_root = tmp_path / "eval_cache" / "gui-opencode" / \
        "shadow_eval" / "screenshot" / "x"
    task_dir = traj_root / "task_a"
    task_dir.mkdir(parents=True, exist_ok=True)
    # Trajectory with only step_start events — no costs to record.
    with open(task_dir / "claude_trajectory.jsonl", "w") as f:
        f.write(json.dumps({"type": "step_start", "timestamp": 1, "part": {}}) + "\n")

    record_eval_costs(
        tmp_db, variant_id="gui-opencode", island="gui",
        clone_dir=str(clone_dir),
        task_ids=["task_a"],
        event_type="eval_task", model_fallback="x",
        timestamp="2026-05-14T00:00:00",
    )
    count = tmp_db.execute(
        "SELECT COUNT(*) FROM cost_events WHERE variant_id = 'gui-opencode'"
    ).fetchone()[0]
    assert count == 0


def test_parse_opencode_trajectory_real_smoke_artifact(tmp_path):
    """Sanity-check the parser produces the same totals as our manual
    inspection of the actual Phase 6 smoke trajectory: 7 step_finish
    events, $0.0182778 total cost, 17161/2217/77088/0 token columns.

    Uses a fixture file that mirrors the smoke trajectory's shape (same
    7 step costs and per-step token counts). If the parser ever drifts
    from this, the smoke verification on the spec is stale."""
    from agent_s4.costs import _parse_opencode_trajectory

    smoke_steps = [
        # Exact per-step values dumped from the real Phase 6 smoke trajectory.
        # (cost,      input, output, reasoning, cache_read, cache_write)
        (0.0074922, 11107,  139,   91,    0,     0),
        (0.0011208,   788,  149,   31,    11088, 0),
        (0.0011016,   762,  133,   46,    11616, 0),
        (0.0030210,  1477,  593,    0,    12144, 0),
        (0.0016506,  1485,  211,    0,    13200, 0),
        (0.0028464,   658,  681,    0,    14256, 0),
        (0.0010452,   884,  143,    0,    14784, 0),
    ]
    task_dir = tmp_path / "task_smoke"
    task_dir.mkdir()
    events = []
    for cost, inp, out, reasoning, cr, cw in smoke_steps:
        events.append({
            "type": "step_finish",
            "timestamp": 1, "sessionID": "ses_smoke",
            "part": {
                "id": "prt_smoke", "reason": "tool-calls",
                "type": "step-finish",
                "tokens": {
                    "total": inp + out + reasoning,
                    "input": inp, "output": out, "reasoning": reasoning,
                    "cache": {"read": cr, "write": cw},
                },
                "cost": cost,
            },
        })
    traj_path = task_dir / "claude_trajectory.jsonl"
    with open(traj_path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    summary = _parse_opencode_trajectory(str(traj_path))
    assert summary is not None
    assert abs(summary["cost_usd"] - 0.0182778) < 1e-9
    assert summary["input_tokens"] == 17161
    # output (2049) + reasoning (168) = 2217 — reasoning folded into output
    assert summary["output_tokens"] == 2217
    assert summary["cache_read_tokens"] == 77088
    assert summary["cache_creation_tokens"] == 0
