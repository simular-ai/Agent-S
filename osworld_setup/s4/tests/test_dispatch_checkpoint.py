"""Completed evolutionary operators must survive a driver restart exactly once."""

import sqlite3

import pytest

from agent_s4 import driver as driver_module
from agent_s4.config.experiment import ExperimentConfig
from agent_s4.config.island import IslandConfig
from agent_s4.config.scheduler import DeterministicScheduler
from agent_s4.db import (
    find_pending_dispatch,
    get_current_step,
    get_population,
    insert_variant,
    log_dispatch_complete,
    log_dispatch_start,
    set_current_step,
)
from agent_s4.driver import Driver


class CheckpointInterruption(BaseException):
    """Stop at the durable checkpoint boundary without changing operator logic."""


# ---- Test helpers ---------------------------------------------------------


def _elimination_config():
    return ExperimentConfig(
        islands=[
            IslandConfig(name="a", evaluator_harness="prime_agent", eval_model="m")
        ],
        scheduler=DeterministicScheduler(0, 0, 10),
        elimination_count=1,
        max_steps=10,
    )


# ---- End test helpers -----------------------------------------------------


def test_resume_after_completed_elimination_does_not_cull_again(tmp_path, monkeypatch):
    """Replay the real driver and elimination operator across a checkpoint stop.

    No builder, model, desktop, or VM is needed: elimination consumes only a
    local SQLite population. The failure hook delegates to the real checkpoint
    writer, then interrupts before the driver can execute its next statement.
    """
    driver = Driver(_elimination_config(), tmp_path)
    set_current_step(driver.conn, 9)
    for score in range(4):
        insert_variant(
            driver.conn,
            {
                "id": f"a-if-{score:03d}",
                "island": "a",
                "tag": f"evol/a/if-{score:03d}",
                "origin": "mutation",
                "step": score + 1,
                "score": score,
                "val_score": score,
            },
        )

    original = driver_module.log_dispatch_complete

    def interrupt_after_checkpoint(*args, **kwargs):
        original(*args, **kwargs)
        raise CheckpointInterruption()

    try:
        with monkeypatch.context() as checkpoint_hook:
            checkpoint_hook.setattr(
                driver_module, "log_dispatch_complete", interrupt_after_checkpoint
            )
            with pytest.raises(CheckpointInterruption):
                driver.run()
        assert [v["id"] for v in get_population(driver.conn, "a")] == [
            "a-if-001",
            "a-if-002",
            "a-if-003",
        ]
    finally:
        driver.conn.close()

    resumed = Driver(_elimination_config(), tmp_path)
    try:
        resumed.run()
        assert [v["id"] for v in get_population(resumed.conn, "a")] == [
            "a-if-001",
            "a-if-002",
            "a-if-003",
        ]
        assert get_current_step(resumed.conn) == 10
        assert find_pending_dispatch(resumed.conn) is None
        assert (
            resumed.conn.execute("SELECT COUNT(*) FROM dispatch_log").fetchone()[0] == 1
        )
        assert (
            resumed.conn.execute(
                "SELECT COUNT(*) FROM history WHERE type = 'elimination'"
            ).fetchone()[0]
            == 1
        )
    finally:
        resumed.conn.close()


def test_failed_step_advance_keeps_dispatch_pending(tmp_db):
    """A failed cursor write must leave recovery evidence for the operator."""
    set_current_step(tmp_db, 9)
    dispatch_id = log_dispatch_start(tmp_db, 10, "elimination", "started")
    tmp_db.execute(
        "CREATE TRIGGER reject_step_advance BEFORE UPDATE ON driver_state "
        "BEGIN SELECT RAISE(ABORT, 'injected cursor failure'); END"
    )

    with pytest.raises(sqlite3.IntegrityError, match="injected cursor failure"):
        log_dispatch_complete(tmp_db, dispatch_id, "completed")

    pending = find_pending_dispatch(tmp_db)
    assert pending is not None and pending["id"] == dispatch_id
    assert get_current_step(tmp_db) == 9
    assert (
        tmp_db.execute(
            "SELECT completed_at FROM dispatch_log WHERE id = ?", (dispatch_id,)
        ).fetchone()[0]
        is None
    )
    assert not tmp_db.in_transaction

    tmp_db.execute("DROP TRIGGER reject_step_advance")
    log_dispatch_complete(tmp_db, dispatch_id, "retried")
    assert find_pending_dispatch(tmp_db) is None
    assert get_current_step(tmp_db) == 10
