from agent_s4.db import connect, init_schema


def test_driver_state_table_exists(tmp_path):
    conn = connect(str(tmp_path / "test.db"))
    init_schema(conn)
    rows = list(conn.execute("PRAGMA table_info(driver_state)"))
    cols = {r[1] for r in rows}
    assert cols == {"current_step"}


def test_dispatch_log_table_exists(tmp_path):
    conn = connect(str(tmp_path / "test.db"))
    init_schema(conn)
    rows = list(conn.execute("PRAGMA table_info(dispatch_log)"))
    cols = {r[1] for r in rows}
    assert {"step", "op_kind", "started_at", "completed_at", "status"} <= cols


def test_variants_has_step_column(tmp_path):
    conn = connect(str(tmp_path / "test.db"))
    init_schema(conn)
    rows = list(conn.execute("PRAGMA table_info(variants)"))
    cols = {r[1] for r in rows}
    assert "step" in cols


def test_init_schema_idempotent(tmp_path):
    conn = connect(str(tmp_path / "test.db"))
    init_schema(conn)
    init_schema(conn)  # second call must not raise
    rows = list(conn.execute("SELECT current_step FROM driver_state"))
    # connect() sets row_factory=sqlite3.Row; cast to tuples for value compare.
    assert [tuple(r) for r in rows] == [(0,)]


def test_current_step_roundtrip(tmp_path):
    from agent_s4.db import get_current_step, set_current_step
    conn = connect(str(tmp_path / "t.db"))
    init_schema(conn)
    assert get_current_step(conn) == 0
    set_current_step(conn, 5)
    assert get_current_step(conn) == 5
    set_current_step(conn, 12)
    assert get_current_step(conn) == 12


def test_dispatch_log_lifecycle(tmp_path):
    from agent_s4.db import (
        log_dispatch_start, log_dispatch_complete, find_pending_dispatch,
    )
    conn = connect(str(tmp_path / "t.db"))
    init_schema(conn)
    assert find_pending_dispatch(conn) is None
    did = log_dispatch_start(conn, step=1, op_kind="mutation",
                             timestamp="2026-05-20T00:00:00Z")
    pending = find_pending_dispatch(conn)
    assert pending is not None
    assert pending["id"] == did
    assert pending["op_kind"] == "mutation"
    log_dispatch_complete(conn, did, "2026-05-20T00:00:05Z")
    assert find_pending_dispatch(conn) is None
