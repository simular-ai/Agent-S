from datetime import datetime, timezone
from unittest.mock import MagicMock
from agent_s4.driver import Driver
from agent_s4.db import (
    connect, init_schema, insert_variant, log_dispatch_start,
)


def test_resume_cleanup_deletes_orphan_variants(tmp_path):
    """Crash mid-step left orphan variants; resume must delete them."""
    db_path = tmp_path / "shadow_evolution.db"
    conn = connect(str(db_path))
    init_schema(conn)

    # Simulate crashed dispatch at step=5
    log_dispatch_start(conn, 5, "mutation", datetime.now(timezone.utc).isoformat())
    insert_variant(conn, {
        "id": "haiku-if-001", "island": "haiku", "tag": "evol/haiku/if-001",
        "parent_id": "haiku-seed", "parent_b_id": None, "origin": "mutation",
        "generation": 1, "score": 50, "eval_level": 0, "step": 5,
        "builder_cost": 0.0, "created_at": "2026-05-20T00:00:00Z",
    })
    conn.close()

    config = MagicMock()
    config.benchmark = "osworld"
    config.islands = [MagicMock(name="haiku")]
    config.islands[0].name = "haiku"

    driver = Driver(config=config, base_dir=tmp_path)
    cleaned = driver._resume_check()
    assert cleaned is True

    # Variant deleted
    row = driver.conn.execute(
        "SELECT id FROM variants WHERE id = 'haiku-if-001'"
    ).fetchone()
    assert row is None


def test_resume_no_pending_dispatches_is_noop(tmp_path):
    db_path = tmp_path / "shadow_evolution.db"
    conn = connect(str(db_path))
    init_schema(conn)
    conn.close()

    config = MagicMock()
    config.benchmark = "osworld"
    config.islands = [MagicMock(name="haiku")]
    config.islands[0].name = "haiku"

    driver = Driver(config=config, base_dir=tmp_path)
    cleaned = driver._resume_check()
    assert cleaned is False
