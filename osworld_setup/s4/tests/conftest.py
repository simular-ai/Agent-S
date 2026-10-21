"""Shared pytest fixtures for V3 tests."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite DB with the V3 schema applied.

    Uses `island_db.connect()` so the connection has WAL mode, FK enforcement,
    and Row factory configured — matching production setup.
    """
    db_path = tmp_path / "test.db"
    from agent_s4.db import connect, init_schema
    conn = connect(str(db_path))
    init_schema(conn)
    yield conn
    conn.close()
