"""Elimination must snapshot the database before it deletes from it.

``delete_variant`` cascades to ``eval_results``, so a cull is the one operator
that discards per-task history. Every held-out comparison is computed from
those rows, so losing them for a culled variant is unrecoverable.
"""
from __future__ import annotations

import sqlite3

import pytest

from agent_s4.db import connect, init_schema
from agent_s4.operators.elimination import BACKUP_DIR, backup_before_cull


def _seeded_db(path):
    conn = connect(str(path))
    init_schema(conn)
    conn.execute(
        "INSERT INTO variants (id, island, tag, origin, score, val_score, step) "
        "VALUES ('v1', 'isl', 'evol/isl/v1', 'seed', 1.0, 1.0, 0)"
    )
    conn.execute(
        "INSERT INTO eval_results (variant_id, task_id, passed, raw_score) "
        "VALUES ('v1', '001', 1, 1.0)"
    )
    return conn


def test_backup_captures_rows_that_the_cull_will_cascade_away(tmp_path):
    db = tmp_path / "shadow_evolution_osworld_v2_ns.db"
    conn = _seeded_db(db)

    target = backup_before_cull(conn, tmp_path, 10)

    assert target == tmp_path / BACKUP_DIR / "shadow_evolution_osworld_v2_ns.pre_elimination_step10.db"
    assert target.is_file()

    # The cull now happens; the backup must still hold the per-task row.
    conn.execute("DELETE FROM variants WHERE id = 'v1'")
    assert conn.execute("SELECT count(*) FROM eval_results").fetchone()[0] == 0

    restored = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
    assert restored.execute(
        "SELECT raw_score FROM eval_results WHERE variant_id = 'v1'"
    ).fetchone()[0] == 1.0
    assert restored.execute("SELECT count(*) FROM variants").fetchone()[0] == 1


def test_backup_refuses_when_the_database_path_is_unknown(tmp_path):
    """An in-memory connection has no file to snapshot, so deleting is refused."""
    conn = sqlite3.connect(":memory:")
    with pytest.raises(RuntimeError, match="could not resolve the database path"):
        backup_before_cull(conn, tmp_path, 10)
