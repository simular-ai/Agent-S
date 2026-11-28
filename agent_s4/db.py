"""SQLite DB layer for island evolution state.

Schema:
    variants — one row per variant in any island's population
    eval_results — per-variant, per-task pass/fail
    eval_sets — task_id → eval level membership (nested subsets)
    islands — per-island counters (mutation_count, last_X_at)
    history — append-only log of mutations, migrations, crossovers, eliminations
"""

import json
import math
import os
import sqlite3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS variants (
    id TEXT PRIMARY KEY,
    island TEXT NOT NULL,
    tag TEXT NOT NULL,
    parent_id TEXT,
    parent_b_id TEXT,
    origin TEXT NOT NULL,
    generation INTEGER DEFAULT 0,
    score REAL,
    val_score REAL,
    eval_level INTEGER DEFAULT 25,
    builder_cost REAL DEFAULT 0,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS val_set (
    task_id TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS eval_results (
    variant_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    passed INTEGER NOT NULL,
    raw_score REAL,
    fitness_score REAL,
    details_json TEXT,
    failure_type TEXT,
    evaluated_at TEXT,
    PRIMARY KEY (variant_id, task_id),
    FOREIGN KEY (variant_id) REFERENCES variants(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS eval_sets (
    task_id TEXT NOT NULL,
    level INTEGER NOT NULL,
    PRIMARY KEY (task_id, level)
);

CREATE TABLE IF NOT EXISTS islands (
    name TEXT PRIMARY KEY,
    mutation_count INTEGER DEFAULT 0,
    last_migration_at INTEGER DEFAULT 0,
    last_crossover_at INTEGER DEFAULT 0,
    last_elimination_at INTEGER DEFAULT 0,
    last_updated TEXT
);

CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    island TEXT,
    parent TEXT,
    child TEXT,
    score REAL,
    cost REAL,
    details TEXT,
    timestamp TEXT
);

CREATE INDEX IF NOT EXISTS idx_variants_island ON variants(island);
CREATE INDEX IF NOT EXISTS idx_eval_results_variant ON eval_results(variant_id);
CREATE INDEX IF NOT EXISTS idx_eval_sets_level ON eval_sets(level);

CREATE TABLE IF NOT EXISTS cost_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    island TEXT NOT NULL,
    variant_id TEXT,
    task_id TEXT,
    model TEXT,
    cost_usd REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_creation_tokens INTEGER,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cost_events_island ON cost_events(island);
CREATE INDEX IF NOT EXISTS idx_cost_events_variant ON cost_events(variant_id);
CREATE INDEX IF NOT EXISTS idx_cost_events_type ON cost_events(event_type);
-- COALESCE the nullable columns so SQLite's "each NULL is distinct in a UNIQUE
-- index" rule doesn't defeat idempotency for build rows (where task_id is NULL).
CREATE UNIQUE INDEX IF NOT EXISTS uniq_cost_events ON cost_events(
    event_type, COALESCE(variant_id, ''), COALESCE(task_id, ''), timestamp
);

CREATE TABLE IF NOT EXISTS driver_state (
    current_step INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS dispatch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step INTEGER NOT NULL,
    op_kind TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('pending', 'complete'))
);

CREATE INDEX IF NOT EXISTS idx_dispatch_log_step ON dispatch_log(step);
CREATE INDEX IF NOT EXISTS idx_dispatch_log_status ON dispatch_log(status);

-- Maintenance has its own journal: an unfinished reevaluation must never
-- make the normal dispatch recovery delete the interfaces it evaluated.
CREATE TABLE IF NOT EXISTS validation_reevaluation_rounds (
    island TEXT NOT NULL,
    after_step INTEGER NOT NULL,
    candidates_json TEXT NOT NULL,
    context_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'complete')),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    PRIMARY KEY (island, after_step)
);

-- after_step=0 is the original evaluation. Later rows are independent
-- validation repeats. Keep the original per-task eval_results untouched.
CREATE TABLE IF NOT EXISTS validation_score_samples (
    variant_id TEXT NOT NULL,
    after_step INTEGER NOT NULL,
    score REAL NOT NULL,
    task_results_json TEXT,
    evaluated_at TEXT NOT NULL,
    PRIMARY KEY (variant_id, after_step),
    FOREIGN KEY (variant_id) REFERENCES variants(id) ON DELETE CASCADE
);
"""


def connect(db_path):
    """Open a SQLite connection with WAL mode and FK enforcement enabled."""
    conn = sqlite3.connect(db_path, isolation_level=None)  # autocommit
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn):
    """Apply the full schema. Idempotent (CREATE IF NOT EXISTS)."""
    conn.executescript(SCHEMA_SQL)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    # Backfill: add step column to variants if missing (SQLite has no
    # ADD COLUMN IF NOT EXISTS, so check first)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(variants)")}
    if "step" not in cols:
        conn.execute("ALTER TABLE variants ADD COLUMN step INTEGER NOT NULL DEFAULT 0")
    eval_cols = {r[1] for r in conn.execute("PRAGMA table_info(eval_results)")}
    for name, sql_type in (
        ("raw_score", "REAL"),
        ("fitness_score", "REAL"),
        ("details_json", "TEXT"),
        ("failure_type", "TEXT"),
    ):
        if name not in eval_cols:
            conn.execute(f"ALTER TABLE eval_results ADD COLUMN {name} {sql_type}")
    # Old rows predate raw_score; binary passed is their lossless fallback.
    conn.execute("UPDATE eval_results SET raw_score = passed WHERE raw_score IS NULL")
    conn.execute(
        "UPDATE eval_results SET fitness_score = passed WHERE fitness_score IS NULL"
    )
    # Ensure driver_state has exactly one row
    row = conn.execute("SELECT COUNT(*) FROM driver_state").fetchone()
    if row[0] == 0:
        conn.execute("INSERT INTO driver_state (current_step) VALUES (0)")


VARIANT_COLUMNS = [
    "id",
    "island",
    "tag",
    "parent_id",
    "parent_b_id",
    "origin",
    "generation",
    "score",
    "val_score",
    "eval_level",
    "step",
    "builder_cost",
    "created_at",
]


def insert_variant(conn, variant):
    """Insert a variant dict.

    Only columns present in `variant` are written; absent columns fall
    back to the schema's DEFAULT (or stay NULL for nullable columns).
    This matters for NOT NULL columns with a DEFAULT — e.g. `step` is
    `NOT NULL DEFAULT 0`, so callers like the seed-row inserter that
    pre-date the column can omit it without violating the constraint.
    Unknown keys raise ValueError to surface typos early.
    """
    unknown = set(variant) - set(VARIANT_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown variant columns: {sorted(unknown)}")
    cols = [c for c in VARIANT_COLUMNS if c in variant]
    placeholders = ", ".join("?" for _ in cols)
    values = [variant[c] for c in cols]
    conn.execute(
        f"INSERT INTO variants ({', '.join(cols)}) VALUES ({placeholders})",
        values,
    )


def get_variant(conn, variant_id):
    """Return the variant row as a dict, or None if not found."""
    row = conn.execute("SELECT * FROM variants WHERE id = ?", (variant_id,)).fetchone()
    return dict(row) if row else None


def get_population(conn, island_name):
    """Return all variants for an island as a list of dicts."""
    rows = conn.execute(
        "SELECT * FROM variants WHERE island = ? ORDER BY id",
        (island_name,),
    ).fetchall()
    return [dict(r) for r in rows]


def update_variant(conn, variant_id, **fields):
    """Update arbitrary fields on a variant. Only known columns allowed."""
    allowed = set(VARIANT_COLUMNS) - {"id"}
    for k in fields:
        if k not in allowed:
            raise ValueError(f"Not a variant column: {k!r}")
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [variant_id]
    conn.execute(f"UPDATE variants SET {set_clause} WHERE id = ?", values)


def delete_variant(conn, variant_id):
    """Remove a variant. Cascades to eval_results."""
    conn.execute("DELETE FROM variants WHERE id = ?", (variant_id,))


def insert_eval_result(
    conn,
    variant_id,
    task_id,
    passed,
    evaluated_at,
    *,
    raw_score=None,
    fitness_score=None,
    details_json=None,
    failure_type=None,
):
    """Insert or replace one task result for a variant."""
    if raw_score is None:
        raw_score = float(passed)
    if fitness_score is None:
        fitness_score = raw_score
    conn.execute(
        "INSERT OR REPLACE INTO eval_results "
        "(variant_id, task_id, passed, raw_score, fitness_score, details_json, failure_type, evaluated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            variant_id,
            task_id,
            passed,
            raw_score,
            fitness_score,
            details_json,
            failure_type,
            evaluated_at,
        ),
    )


def load_eval_sets(conn, base_dir, eval_sets_map):
    """Populate eval_sets table from JSON files.

    eval_sets_map: dict of {level: relative_path}, e.g. in production
    {25: 'eval_sets/eval_island_25.json', ...} or, in tests with their
    own layout, {25: 'eval_island_25.json'}. The path is joined with
    base_dir verbatim — the caller is responsible for the convention.
    Files contain JSON arrays of task_ids.
    """
    rows = []
    for level, filename in eval_sets_map.items():
        path = os.path.join(base_dir, filename)
        with open(path) as f:
            task_ids = json.load(f)
        for tid in task_ids:
            rows.append((tid, level))
    conn.executemany(
        "INSERT OR IGNORE INTO eval_sets (task_id, level) VALUES (?, ?)",
        rows,
    )


def recompute_score_from_train_set(conn, variant_id, train_task_ids):
    """Count this variant's passes restricted to an explicit train task set.

    v2 replacement for the retired tier-joined `recompute_score`: rather
    than joining eval_results against the eval_sets/level row (which
    encoded the old tier-cascade promotion model), the caller passes the
    canonical train task IDs directly. Returns 0 when train_task_ids is
    empty.
    """
    if not train_task_ids:
        return 0
    placeholders = ",".join("?" * len(train_task_ids))
    row = conn.execute(
        # COALESCE preserves compatibility with legacy/direct SQL writers
        # that only populate the historical binary ``passed`` column.
        f"SELECT COALESCE(SUM(COALESCE(fitness_score, passed)), 0) "
        f"FROM eval_results "
        f"WHERE variant_id = ? AND task_id IN ({placeholders})",
        (variant_id, *train_task_ids),
    ).fetchone()
    return float(row[0])


ISLAND_UPDATABLE_COLUMNS = {
    "mutation_count",
    "last_migration_at",
    "last_crossover_at",
    "last_elimination_at",
    "last_updated",
}

HISTORY_COLUMNS = [
    "type",
    "island",
    "parent",
    "child",
    "score",
    "cost",
    "details",
    "timestamp",
]


def init_island(conn, name):
    """Ensure an island row exists. Idempotent."""
    conn.execute(
        "INSERT OR IGNORE INTO islands (name) VALUES (?)",
        (name,),
    )


def get_island(conn, name):
    row = conn.execute("SELECT * FROM islands WHERE name = ?", (name,)).fetchone()
    return dict(row) if row else None


def update_island(conn, name, **fields):
    for k in fields:
        if k not in ISLAND_UPDATABLE_COLUMNS:
            raise ValueError(f"Not an island column: {k!r}")
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [name]
    conn.execute(f"UPDATE islands SET {set_clause} WHERE name = ?", values)


def append_history(conn, entry):
    cols = ", ".join(HISTORY_COLUMNS)
    placeholders = ", ".join("?" for _ in HISTORY_COLUMNS)
    values = [entry.get(c) for c in HISTORY_COLUMNS]
    conn.execute(f"INSERT INTO history ({cols}) VALUES ({placeholders})", values)


COST_EVENT_COLUMNS = [
    "event_type",
    "island",
    "variant_id",
    "task_id",
    "model",
    "cost_usd",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "timestamp",
]


def insert_cost_event(conn, **fields):
    """Insert a cost event row. Required keys: all columns in COST_EVENT_COLUMNS.

    Uses INSERT OR IGNORE against a unique index on
    (event_type, COALESCE(variant_id, ''), COALESCE(task_id, ''), timestamp)
    so re-runs of the backfill (or accidental duplicate calls) are idempotent.
    """
    for k in fields:
        if k not in COST_EVENT_COLUMNS:
            raise ValueError(f"Not a cost_events column: {k!r}")
    cols = ", ".join(COST_EVENT_COLUMNS)
    placeholders = ", ".join("?" for _ in COST_EVENT_COLUMNS)
    values = [fields.get(c) for c in COST_EVENT_COLUMNS]
    conn.execute(
        f"INSERT OR IGNORE INTO cost_events ({cols}) VALUES ({placeholders})",
        values,
    )


# ---- Validation set --------------------------------------------------------


def load_val_set(conn, task_ids):
    """Replace val_set table contents with task_ids.

    Single point at which the val_set_file → DB transition happens. After
    this is called, only the val_set table is consulted; the file is not
    re-read by any code path. Idempotent on equal inputs.
    """
    conn.execute("DELETE FROM val_set")
    conn.executemany(
        "INSERT INTO val_set(task_id) VALUES (?)",
        [(tid,) for tid in task_ids],
    )


def get_val_task_ids(conn):
    """Return all task_ids in val_set as a list."""
    rows = conn.execute("SELECT task_id FROM val_set").fetchall()
    return [r["task_id"] for r in rows]


def recompute_val_score(conn, variant_id):
    """Return the representative validation score for this variant.

    Repeated evaluations contribute equally, including the original score.
    Without repeats, join eval_results against the canonical validation set.
    """
    mean = conn.execute(
        "SELECT AVG(score) FROM validation_score_samples WHERE variant_id = ?",
        (variant_id,),
    ).fetchone()[0]
    if mean is not None:
        return float(mean)
    row = conn.execute(
        """
        SELECT COALESCE(SUM(COALESCE(er.fitness_score, er.passed)), 0) AS n
        FROM eval_results er
        JOIN val_set vs ON er.task_id = vs.task_id
        WHERE er.variant_id = ?
        """,
        (variant_id,),
    ).fetchone()
    return float(row["n"])


def get_reevaluation_round(conn, island, after_step):
    row = conn.execute(
        "SELECT * FROM validation_reevaluation_rounds "
        "WHERE island = ? AND after_step = ?",
        (island, after_step),
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["candidates"] = json.loads(result.pop("candidates_json"))
    result["context"] = json.loads(result.pop("context_json"))
    return result


def start_reevaluation_round(conn, island, after_step, candidates, context, timestamp):
    """Pin the candidates and evaluation settings before any remote work."""
    conn.execute(
        "INSERT OR IGNORE INTO validation_reevaluation_rounds "
        "(island, after_step, candidates_json, context_json, status, started_at) "
        "VALUES (?, ?, ?, ?, 'pending', ?)",
        (island, after_step, json.dumps(candidates), json.dumps(context), timestamp),
    )
    return get_reevaluation_round(conn, island, after_step)


def complete_reevaluation_round(conn, island, after_step, outcomes, timestamp):
    """Commit all new samples, means, history, and completion atomically.

    The round's pinned selection survives failed evaluations. Replaying a
    completed round does not add samples or average a result a second time.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        round_ = get_reevaluation_round(conn, island, after_step)
        if round_ is None:
            raise ValueError("reevaluation round has not been started")
        candidates = round_["candidates"]
        if round_["status"] == "complete":
            means = {v["id"]: recompute_val_score(conn, v["id"]) for v in candidates}
            conn.commit()
            return means
        by_id = {o["variant_id"]: o for o in outcomes}
        if len(by_id) != len(outcomes) or set(by_id) != {v["id"] for v in candidates}:
            raise ValueError("reevaluation outcomes do not match the pinned candidates")
        n_tasks = len(round_["context"]["task_ids"])
        means = {}
        for candidate in candidates:
            variant_id = candidate["id"]
            outcome = by_id[variant_id]
            new_score = float(outcome["score"])
            original_score = float(candidate["val_score"])
            if any(
                not math.isfinite(s) or not 0 <= s <= n_tasks
                for s in (new_score, original_score)
            ):
                raise ValueError(
                    "validation score is outside the validation task count"
                )
            variant = get_variant(conn, variant_id)
            if variant is None or variant["island"] != island:
                raise ValueError(
                    f"reevaluated variant is no longer in its island: {variant_id}"
                )
            conn.execute(
                "INSERT OR IGNORE INTO validation_score_samples "
                "(variant_id, after_step, score, evaluated_at) VALUES (?, 0, ?, ?)",
                (variant_id, original_score, round_["started_at"]),
            )
            conn.execute(
                "INSERT INTO validation_score_samples "
                "(variant_id, after_step, score, task_results_json, evaluated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    variant_id,
                    after_step,
                    new_score,
                    json.dumps(outcome["records"]),
                    outcome["evaluated_at"],
                ),
            )
            mean = recompute_val_score(conn, variant_id)
            count = conn.execute(
                "SELECT COUNT(*) FROM validation_score_samples WHERE variant_id = ?",
                (variant_id,),
            ).fetchone()[0]
            update_variant(conn, variant_id, val_score=mean)
            append_history(
                conn,
                {
                    "type": "reevaluation",
                    "island": island,
                    "child": variant_id,
                    "score": mean,
                    "timestamp": timestamp,
                    "details": json.dumps(
                        {
                            "after_step": after_step,
                            "previous_val_score": variant["val_score"],
                            "new_val_score": new_score,
                            "representative_val_score": mean,
                            "evaluation_count": count,
                            "result_dir": outcome["result_dir"],
                        }
                    ),
                },
            )
            means[variant_id] = mean
        conn.execute(
            "UPDATE validation_reevaluation_rounds SET status = 'complete', completed_at = ? "
            "WHERE island = ? AND after_step = ?",
            (timestamp, island, after_step),
        )
        conn.commit()
        return means
    except BaseException:
        conn.rollback()
        raise


def verify_val_set_unchanged(conn, file_task_ids):
    """Raise ValueError if the val_set table is non-empty AND its content
    differs from file_task_ids (as sets). No-op if val_set is empty
    (first-run / not-yet-populated state).

    Used at orchestrator init to catch resumes where someone changed the
    val_set_file content between runs — silently changing val task IDs
    would invalidate already-recorded val_score values.
    """
    db_ids = set(get_val_task_ids(conn))
    if not db_ids:
        return  # Fresh table; safe to populate.
    file_ids = set(file_task_ids)
    if db_ids == file_ids:
        return
    added = sorted(file_ids - db_ids)
    removed = sorted(db_ids - file_ids)
    raise ValueError(
        "val_set has changed since this run was started. "
        f"Added in file but not in DB: {added}. "
        f"In DB but missing from file: {removed}. "
        "Either restore the original val_set_file content or start a fresh run."
    )


# ---- Driver state / dispatch log (v2 lockstep loop) -----------------------


def get_current_step(conn):
    row = conn.execute("SELECT current_step FROM driver_state").fetchone()
    return int(row[0]) if row else 0


def set_current_step(conn, step):
    conn.execute("UPDATE driver_state SET current_step = ?", (step,))


def log_dispatch_start(conn, step, op_kind, timestamp):
    cur = conn.execute(
        "INSERT INTO dispatch_log (step, op_kind, started_at, status) "
        "VALUES (?, ?, ?, 'pending')",
        (step, op_kind, timestamp),
    )
    return cur.lastrowid


def log_dispatch_complete(conn, dispatch_id, timestamp):
    """Checkpoint completion and the step cursor together to prevent replay.

    Connections use autocommit, so separate writes can leave a completed
    dispatch behind a stale cursor after interruption. Recovery only looks
    for pending dispatches and would otherwise repeat the completed operator.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT step FROM dispatch_log WHERE id = ?", (dispatch_id,)
        ).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE dispatch_log SET completed_at = ?, status = 'complete' WHERE id = ?",
                (timestamp, dispatch_id),
            )
            set_current_step(conn, row["step"])
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def find_pending_dispatch(conn):
    row = conn.execute(
        "SELECT id, step, op_kind, started_at FROM dispatch_log "
        "WHERE status = 'pending' ORDER BY step ASC, id ASC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def delete_orphan_variants_at_step(conn, step):
    """Delete variants inserted by a crashed step. Cascades to eval_results."""
    rows = list(
        conn.execute("SELECT id, island, tag FROM variants WHERE step = ?", (step,))
    )
    for row in rows:
        conn.execute("DELETE FROM variants WHERE id = ?", (row["id"],))
    return rows  # caller uses tag to clean up git tags on disk
