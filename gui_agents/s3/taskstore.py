"""TaskStore — memória procedural do Agent-S3 (sqlite3 stdlib, sem deps).

Grava cada run (telemetria + sequência de tools executada) e replaya a
sequência vencedora (perfect=true) de runs passadas com a mesma signature,
pular o PLAN Anthropic no ciclo 1 e só VERIFY. Se verify não for perfect,
cai no loop ReAct normal — nunca confia cegamente na memória.

Signature = f"{bpm}:{grid}:{ntracks}:{sorted(track_names)}" — mesma tarefa
mesma assinatura. Replay SEMPRE roteia tools destrutivas pelo destructiveGate
/PeerLock existente no orchestrator (não é aqui).

Gate: AGENT_S_USE_MEMORY=0 desabilita lookup/record. Data em
~/Agent-S/data/taskstore.db (cria se não existir).
"""
import json
import os
import sqlite3
import time
from pathlib import Path

DEFAULT_DB = os.path.expanduser("~/Agent-S/data/taskstore.db")

_SCHEMA = """CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    task            TEXT,
    signature       TEXT,
    t_start         REAL,
    t_end           REAL,
    cycles          INTEGER,
    api_calls       INTEGER,
    tool_calls      INTEGER,
    cost_usd        REAL,
    perfect         INTEGER,
    before_json     TEXT,
    after_json      TEXT,
    tool_sequence_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_sig_perfect ON runs(signature, perfect, t_end);
"""


def _connect(db_path=DEFAULT_DB):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def enabled():
    """Memória procedural habilitada? (AGENT_S_USE_MEMORY=0 desliga)."""
    return os.environ.get("AGENT_S_USE_MEMORY", "1") != "0"


def signature(bpm, grid, tracks, project=None):
    """Assinatura de tarefa: bpm:grid:n:sorted(names)[:project].

    `project` (caminho .cpr ou id do projeto Cubase) disambigua projetos
    distintos com mesmas inputs — sem ele, bpm=None + 0 tracks colide em
    "None:1/16:0:[]" entre projetos e a memória procedural replaya a sequência
    vencedora de um projeto no outro (cross-project replay). Default None
    preserva callers legados (3 args)."""
    names = sorted(t.get("name", "") for t in (tracks or []))
    base = f"{bpm}:{grid}:{len(names)}:{names}"
    if project is None:
        return base
    return f"{base}:{project}"


def record_run(run_id, task, sig, t_start, t_end, cycles, api_calls,
               tool_calls, cost_usd, perfect, before, after, tool_sequence,
               db_path=DEFAULT_DB):
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO runs "
            "(run_id,task,signature,t_start,t_end,cycles,api_calls,tool_calls,"
            "cost_usd,perfect,before_json,after_json,tool_sequence_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, task, sig, t_start, t_end, cycles, api_calls, tool_calls,
             cost_usd, 1 if perfect else 0,
             json.dumps(before, default=str),
             json.dumps(after, default=str),
             json.dumps(tool_sequence, default=str)),
        )
        conn.commit()
    finally:
        conn.close()


def lookup_winning_sequence(sig, db_path=DEFAULT_DB):
    """Sequência da run perfect=true MAIS RECENTE com essa signature.
    Retorna list[{name,args}] ou None."""
    if not enabled():
        return None
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT tool_sequence_json FROM runs WHERE signature=? AND perfect=1 "
            "ORDER BY t_end DESC LIMIT 1", (sig,)
        ).fetchone()
        if not row or not row["tool_sequence_json"]:
            return None
        seq = json.loads(row["tool_sequence_json"])
        return seq if isinstance(seq, list) and seq else None
    finally:
        conn.close()


def list_recent(limit=20, db_path=DEFAULT_DB):
    """Runs recentes pra diagnóstico."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT run_id,task,signature,t_end,cycles,api_calls,tool_calls,"
            "perfect,cost_usd FROM runs ORDER BY t_end DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def init(db_path=DEFAULT_DB):
    """Cria/schema do banco. Idempotente."""
    conn = _connect(db_path)
    conn.close()