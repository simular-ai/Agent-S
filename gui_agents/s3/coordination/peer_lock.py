"""Peer coordination: lock SQLite compartilhado + log de eventos entre o
CubeFlow autopilot (Node) e o Agent-S3 (Python). O schema abaixo precisa
ficar em sincronia com CubeFlow/src/coordination/schema.sql — os dois
processos abrem o MESMO arquivo .db, então nome de tabela/coluna aqui são
um contrato entre repositórios, não um detalhe interno.
"""
import json
import os
import sqlite3
import time

HEARTBEAT_STALE_MS = 30_000
DEFAULT_TIMEOUT_MS = 10_000
POLL_INTERVAL_S = 0.1
DB_PATH_ENV = "CUBEFLOW_COORDINATION_DB"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS driver_lock (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  holder TEXT,
  task_id TEXT,
  acquired_at INTEGER,
  heartbeat_at INTEGER
);

INSERT OR IGNORE INTO driver_lock (id, holder, task_id, acquired_at, heartbeat_at)
VALUES (1, NULL, NULL, NULL, NULL);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp INTEGER NOT NULL,
  peer TEXT NOT NULL,
  task_id TEXT,
  action TEXT NOT NULL,
  payload TEXT,
  success INTEGER
);
"""


def _now_ms():
    return int(time.time() * 1000)


def default_db_path():
    path = os.environ.get(DB_PATH_ENV)
    if not path:
        raise RuntimeError(
            f"{DB_PATH_ENV} not set — point it at CubeFlow's "
            "src/database/peer-coordination.db"
        )
    return path


class PeerLock:
    def __init__(self, db_path=None):
        self.db_path = db_path or default_db_path()
        self.conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA_SQL)

    def _insert_event(self, peer, task_id, action, payload, success):
        success_val = None if success is None else (1 if success else 0)
        cur = self.conn.execute(
            "INSERT INTO events (timestamp, peer, task_id, action, payload, success) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (_now_ms(), peer, task_id, action, json.dumps(payload or {}), success_val),
        )
        return cur.lastrowid

    def _try_acquire_once(self, peer, task_id):
        now = _now_ms()
        stale_before = now - HEARTBEAT_STALE_MS
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                "SELECT holder, heartbeat_at FROM driver_lock WHERE id = 1"
            ).fetchone()
            holder, heartbeat_at = row
            free = holder is None or (
                heartbeat_at is not None and heartbeat_at < stale_before
            )
            if not free:
                self.conn.execute("COMMIT")
                return False
            stolen = holder is not None
            self.conn.execute(
                "UPDATE driver_lock SET holder=?, task_id=?, acquired_at=?, heartbeat_at=? "
                "WHERE id=1",
                (peer, task_id, now, now),
            )
            self._insert_event(
                peer, task_id, "lock_stolen" if stolen else "lock_acquired",
                {"from": holder if stolen else None}, True,
            )
            self.conn.execute("COMMIT")
            return True
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def acquire(self, peer, task_id, timeout_ms=DEFAULT_TIMEOUT_MS):
        deadline = _now_ms() + timeout_ms
        while True:
            if self._try_acquire_once(peer, task_id):
                return {"ok": True}
            if _now_ms() >= deadline:
                return {"ok": False, "reason": "timeout"}
            time.sleep(POLL_INTERVAL_S)

    def is_held_by_other(self, peer):
        """True se o lock está num peer diferente, não-stale. False se livre ou held por `peer`."""
        row = self.conn.execute(
            "SELECT holder, heartbeat_at FROM driver_lock WHERE id = 1"
        ).fetchone()
        if row is None:
            return False
        holder, heartbeat_at = row
        if holder is None:
            return False
        stale = heartbeat_at is not None and heartbeat_at < (_now_ms() - HEARTBEAT_STALE_MS)
        if stale:
            return False
        return holder != peer

    def release(self, peer):
        cur = self.conn.execute(
            "UPDATE driver_lock SET holder=NULL, task_id=NULL WHERE id=1 AND holder=?",
            (peer,),
        )
        released = cur.rowcount == 1
        if released:
            self._insert_event(peer, None, "lock_released", {}, True)
        return released

    def heartbeat(self, peer):
        cur = self.conn.execute(
            "UPDATE driver_lock SET heartbeat_at=? WHERE id=1 AND holder=?",
            (_now_ms(), peer),
        )
        return cur.rowcount == 1

    def log_event(self, peer, task_id, action, payload=None, success=None):
        return self._insert_event(peer, task_id, action, payload, success)

    def events_since(self, cursor_id=0):
        rows = self.conn.execute(
            "SELECT id, timestamp, peer, task_id, action, payload, success "
            "FROM events WHERE id > ? ORDER BY id",
            (cursor_id,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "timestamp": r[1],
                "peer": r[2],
                "task_id": r[3],
                "action": r[4],
                "payload": json.loads(r[5]),
                "success": None if r[6] is None else bool(r[6]),
            }
            for r in rows
        ]

    def close(self):
        self.conn.close()
