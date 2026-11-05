# gui_agents/s3/persistence/task_store.py
"""TaskStore — persistência de estado de tarefas (SQLite, stdlib + pydantic).

Camada genérica de lifecycle de tarefa: id, status, instrução, resultado,
erro, tentativas, timestamps. Complementar ao ``gui_agents.s3.taskstore``
(que faz replay de memória procedural de sequências vencedoras) — este
módulo só rastreia estado, não decide execução.

Schema idempotente, conexão por chamada (mesmo padrão do taskstore existente),
row_factory=sqlite3.Row. Path default ``~/Agent-S/data/tasks.db``.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

DEFAULT_DB = os.path.expanduser("~/Agent-S/data/tasks.db")

_SCHEMA = """CREATE TABLE IF NOT EXISTS tasks (
    id            TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    instruction   TEXT NOT NULL,
    result_json   TEXT,
    error         TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    started_at    REAL,
    completed_at  REAL,
    meta_json     TEXT,
    idempotency_key TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Adiciona colunas/indexes novos em DBs existentes (idempotency_key)."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    if "idempotency_key" not in cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN idempotency_key TEXT")
        conn.commit()
    # Index idempotente — criado sempre (novo DB ou migrado). WHERE parcial
    # deixa index compacto e útil só p/ lookups de keys não-NULL.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_idem "
        "ON tasks(idempotency_key) WHERE idempotency_key IS NOT NULL"
    )
    conn.commit()


class TaskStatus(str, Enum):
    """Ciclo de vida de uma tarefa."""

    PENDING = "pending"
    RUNNING = "running"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Statuses terminais — usados por DAG executor e API p/ saber se acabou.
TERMINAL_STATES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)


class TaskRecord(BaseModel):
    """Snapshot imutável do estado de uma tarefa."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    instruction: str
    result: Optional[Any] = None
    error: Optional[str] = None
    attempts: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATES


def _connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    """Conexão por chamada, configurada p/ concorrência segura.

    - WAL: leitores não bloqueiam escritor; writes concorrentes serializam
      via file-lock mas sem bloquear reads.
    - busy_timeout=30s: threads que contendem pelo lock ESPERAM em vez de
      estourar ``OperationalError: database is locked``.
    - synchronous=NORMAL: seguro em WAL, mais rápido que FULL.
    Conexão é por-chamada (abre/fecha) — sem conexão compartilhada entre
    threads, então ``check_same_thread`` default não é problema.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    conn.commit()
    _migrate(conn)
    return conn


def _row_to_record(row: sqlite3.Row) -> TaskRecord:
    return TaskRecord(
        id=row["id"],
        status=TaskStatus(row["status"]),
        instruction=row["instruction"],
        result=json.loads(row["result_json"]) if row["result_json"] else None,
        error=row["error"],
        attempts=row["attempts"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        metadata=json.loads(row["meta_json"]) if row["meta_json"] else {},
    )


class TaskStore:
    """Store de lifecycle de tarefas, thread-safe por chamada.

    Cada método abre/fecha a conexão (sqlite3 serializa writes no nível do
    arquivo), espelhando o padrão do ``s3.taskstore`` — sem lock em Python,
    sem state compartilhado.
    """

    def __init__(self, db_path: str = DEFAULT_DB) -> None:
        self.db_path = db_path
        # Cria schema na construção pra falhar cedo se path ruim.
        conn = _connect(db_path)
        conn.close()

    def create(
        self,
        instruction: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
        task_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> TaskRecord:
        """Cria tarefa PENDING e devolve o record inicial.

        Idempotência: se ``idempotency_key`` já existe, devolve o record
        existente SEM criar novo (INSERT or IGNORE + lookup). Permite
        retries seguros de ``POST /tasks`` (mesma key → mesma task).
        """
        if idempotency_key:
            existing = self.get_by_idempotency(idempotency_key)
            if existing is not None:
                return existing
        rec = TaskRecord(
            id=task_id or str(uuid.uuid4()),
            instruction=instruction,
            metadata=metadata or {},
        )
        conn = _connect(self.db_path)
        try:
            # INSERT or IGNORE: race-safe se 2 threads entram com mesma key
            # antes do primeiro commit. Fallback p/ lookup após.
            conn.execute(
                "INSERT OR IGNORE INTO tasks "
                "(id,status,instruction,result_json,error,attempts,"
                "created_at,updated_at,started_at,completed_at,meta_json,"
                "idempotency_key) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rec.id,
                    rec.status.value,
                    rec.instruction,
                    None,
                    None,
                    0,
                    rec.created_at,
                    rec.updated_at,
                    None,
                    None,
                    json.dumps(rec.metadata, default=str),
                    idempotency_key,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        if idempotency_key:
            # em caso de race (outra thread inseriu primeiro), devolve o dela
            existing = self.get_by_idempotency(idempotency_key)
            if existing is not None and existing.id != rec.id:
                return existing
        return rec

    def get_by_idempotency(self, key: str) -> Optional[TaskRecord]:
        conn = _connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT * FROM tasks WHERE idempotency_key=?", (key,)
            ).fetchone()
            return _row_to_record(row) if row else None
        finally:
            conn.close()

    def get(self, task_id: str) -> Optional[TaskRecord]:
        conn = _connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            return _row_to_record(row) if row else None
        finally:
            conn.close()

    def _update(self, task_id: str, fields: dict[str, Any]) -> Optional[TaskRecord]:
        """Aplica um patch de colunas. ``fields`` já vem com nomes de coluna."""
        fields["updated_at"] = time.time()
        set_clause = ", ".join(f"{k}=?" for k in fields)
        values = list(fields.values()) + [task_id]
        conn = _connect(self.db_path)
        try:
            conn.execute(f"UPDATE tasks SET {set_clause} WHERE id=?", values)
            conn.commit()
        finally:
            conn.close()
        return self.get(task_id)

    def mark_running(self, task_id: str) -> Optional[TaskRecord]:
        now = time.time()
        return self._update(
            task_id,
            {
                "status": TaskStatus.RUNNING.value,
                "started_at": now,
                "updated_at": now,
            },
        )

    def mark_retrying(self, task_id: str) -> Optional[TaskRecord]:
        return self._update(task_id, {"status": TaskStatus.RETRYING.value})

    def increment_attempts(self, task_id: str) -> Optional[TaskRecord]:
        conn = _connect(self.db_path)
        try:
            conn.execute(
                "UPDATE tasks SET attempts=attempts+1, updated_at=? WHERE id=?",
                (time.time(), task_id),
            )
            conn.commit()
        finally:
            conn.close()
        return self.get(task_id)

    def set_result(self, task_id: str, result: Any) -> Optional[TaskRecord]:
        now = time.time()
        return self._update(
            task_id,
            {
                "status": TaskStatus.COMPLETED.value,
                "result_json": json.dumps(result, default=str),
                "completed_at": now,
                "updated_at": now,
            },
        )

    def set_error(self, task_id: str, error: str) -> Optional[TaskRecord]:
        now = time.time()
        return self._update(
            task_id,
            {
                "status": TaskStatus.FAILED.value,
                "error": error,
                "completed_at": now,
                "updated_at": now,
            },
        )

    def cancel(self, task_id: str) -> Optional[TaskRecord]:
        now = time.time()
        return self._update(
            task_id,
            {
                "status": TaskStatus.CANCELLED.value,
                "completed_at": now,
                "updated_at": now,
            },
        )

    def cancel_pending(self, reason: str = "shutdown") -> int:
        """Shutdown gracioso: marca todas PENDING (ainda na fila, não iniciadas)
        como CANCELLED. Devolve qtd afetada. RUNNING não é tocado (vai terminar)."""
        now = time.time()
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                "UPDATE tasks SET status=?, error=COALESCE(error, ?), "
                "completed_at=?, updated_at=? WHERE status=?",
                (TaskStatus.CANCELLED.value, reason, now, now,
                 TaskStatus.PENDING.value),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def list(
        self,
        *,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
    ) -> list[TaskRecord]:
        conn = _connect(self.db_path)
        try:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE status=? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (status.value, limit),
                ).fetchall()
            return [_row_to_record(r) for r in rows]
        finally:
            conn.close()

    def count(self, status: Optional[TaskStatus] = None) -> int:
        conn = _connect(self.db_path)
        try:
            if status is None:
                row = conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM tasks WHERE status=?",
                    (status.value,),
                ).fetchone()
            return int(row["n"])
        finally:
            conn.close()

    def recover_orphans(self, *, requeue: bool = False) -> int:
        """Reaper de startup — trata tarefas RUNNING/RETRYING órfãs.

        Após um crash/restart do processo, tasks que estavam RUNNING ficam
        travadas nesse estado (ninguém vai completá-las). Chame no startup.

        Args:
            requeue: True → volta p/ PENDING (re-executar). False (default)
                → marca FAILED com erro "process restarted".
        Devolve qtd de tasks afetadas.
        """
        now = time.time()
        new_status = TaskStatus.PENDING.value if requeue else TaskStatus.FAILED.value
        error = None if requeue else "orphaned: process restarted while RUNNING"
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                "UPDATE tasks SET status=?, error=COALESCE(error, ?), "
                "updated_at=?, completed_at=? WHERE status IN (?,?)",
                (
                    new_status,
                    error,
                    now,
                    now if not requeue else None,
                    TaskStatus.RUNNING.value,
                    TaskStatus.RETRYING.value,
                ),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()