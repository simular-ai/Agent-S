"""Tests for island_db module."""


def test_init_schema_creates_all_tables(tmp_db):
    """After init_schema, the expected tables exist."""
    cursor = tmp_db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cursor.fetchall()]
    assert "variants" in tables
    assert "eval_results" in tables
    assert "eval_sets" in tables
    assert "islands" in tables
    assert "history" in tables


def test_init_schema_creates_indexes(tmp_db):
    """Indexes are created for common queries."""
    cursor = tmp_db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
    indexes = [r[0] for r in cursor.fetchall()]
    assert "idx_variants_island" in indexes
    assert "idx_eval_results_variant" in indexes
    assert "idx_eval_sets_level" in indexes


def test_init_schema_enables_wal_mode(tmp_db):
    """WAL journal mode is enabled."""
    cursor = tmp_db.cursor()
    cursor.execute("PRAGMA journal_mode")
    mode = cursor.fetchone()[0]
    assert mode.lower() == "wal"


def test_insert_and_get_variant(tmp_db):
    """insert_variant stores a variant; get_variant retrieves it."""
    from agent_s4.db import insert_variant, get_variant
    variant = {
        "id": "gui-if-001",
        "island": "gui",
        "tag": "evol/gui/if-001",
        "parent_id": "gui-seed",
        "parent_b_id": None,
        "origin": "mutation",
        "generation": 1,
        "score": 13,
        "eval_level": 25,
        "builder_cost": 4.78,
        "created_at": "2026-04-17T00:00:00",
    }
    insert_variant(tmp_db, variant)
    got = get_variant(tmp_db, "gui-if-001")
    assert got["id"] == "gui-if-001"
    assert got["score"] == 13
    assert got["eval_level"] == 25
    assert got["origin"] == "mutation"


def test_get_variant_missing_returns_none(tmp_db):
    from agent_s4.db import get_variant
    assert get_variant(tmp_db, "does-not-exist") is None


def test_get_population_filters_by_island(tmp_db):
    """get_population returns only variants for the given island."""
    from agent_s4.db import insert_variant, get_population
    for vid, isl in [("gui-a", "gui"), ("gui-b", "gui"), ("cli-a", "cli")]:
        insert_variant(tmp_db, {
            "id": vid, "island": isl, "tag": f"tag/{vid}",
            "parent_id": None, "parent_b_id": None, "origin": "seed",
            "generation": 0, "score": 10, "eval_level": 25,
            "builder_cost": 0, "created_at": "2026-04-17T00:00:00",
        })
    gui_pop = get_population(tmp_db, "gui")
    assert {v["id"] for v in gui_pop} == {"gui-a", "gui-b"}


def test_update_variant_score_and_level(tmp_db):
    """update_variant changes score and eval_level."""
    from agent_s4.db import insert_variant, update_variant, get_variant
    insert_variant(tmp_db, {
        "id": "gui-a", "island": "gui", "tag": "t",
        "parent_id": None, "parent_b_id": None, "origin": "seed",
        "generation": 0, "score": 10, "eval_level": 25,
        "builder_cost": 0, "created_at": "2026-04-17T00:00:00",
    })
    update_variant(tmp_db, "gui-a", score=18, eval_level=50)
    got = get_variant(tmp_db, "gui-a")
    assert got["score"] == 18
    assert got["eval_level"] == 50


def test_delete_variant_cascades_to_eval_results(tmp_db):
    """Deleting a variant removes its eval_results via ON DELETE CASCADE."""
    from agent_s4.db import insert_variant, delete_variant, insert_eval_result
    insert_variant(tmp_db, {
        "id": "gui-a", "island": "gui", "tag": "t",
        "parent_id": None, "parent_b_id": None, "origin": "seed",
        "generation": 0, "score": 10, "eval_level": 25,
        "builder_cost": 0, "created_at": "2026-04-17T00:00:00",
    })
    insert_eval_result(tmp_db, "gui-a", "task-x", 1, "2026-04-17T00:00:00")
    delete_variant(tmp_db, "gui-a")
    cursor = tmp_db.execute("SELECT COUNT(*) FROM eval_results WHERE variant_id = 'gui-a'")
    assert cursor.fetchone()[0] == 0


def test_fractional_scores_and_details_are_preserved(tmp_db):
    from agent_s4.db import (
        insert_eval_result, insert_variant, recompute_score_from_train_set,
    )
    insert_variant(tmp_db, {
        "id": "raw-v2", "island": "gui", "tag": "x", "origin": "seed",
    })
    details = '{"score": 0.49, "partial_scores": {"a": 1}}'
    insert_eval_result(
        tmp_db, "raw-v2", "task-a", 0, "now",
        raw_score=0.49, details_json=details,
    )
    insert_eval_result(
        tmp_db, "raw-v2", "task-b", 1, "now", raw_score=0.51,
    )
    row = tmp_db.execute(
        "SELECT passed, raw_score, fitness_score, details_json FROM eval_results "
        "WHERE variant_id='raw-v2' AND task_id='task-a'"
    ).fetchone()
    assert row["passed"] == 0
    assert row["raw_score"] == 0.49
    assert row["fitness_score"] == 0.49
    assert row["details_json"] == details
    assert recompute_score_from_train_set(
        tmp_db, "raw-v2", ["task-a", "task-b"],
    ) == 1.0


def test_load_eval_sets_from_json(tmp_db, tmp_path):
    """load_eval_sets reads eval_island_*.json files and inserts rows."""
    import json
    from agent_s4.db import load_eval_sets
    (tmp_path / "eval_island_25.json").write_text(json.dumps(["task_a", "task_b"]))
    (tmp_path / "eval_island_50.json").write_text(json.dumps(["task_a", "task_b", "task_c"]))
    load_eval_sets(tmp_db, str(tmp_path), {25: "eval_island_25.json", 50: "eval_island_50.json"})
    rows = tmp_db.execute(
        "SELECT task_id, level FROM eval_sets ORDER BY level, task_id"
    ).fetchall()
    assert [(r["task_id"], r["level"]) for r in rows] == [
        ("task_a", 25), ("task_b", 25),
        ("task_a", 50), ("task_b", 50), ("task_c", 50),
    ]


def test_recompute_score_from_train_set_counts_passes(tmp_db):
    """recompute_score_from_train_set sums passes for the given task list."""
    from agent_s4.db import (
        insert_variant, insert_eval_result, recompute_score_from_train_set,
    )
    insert_variant(tmp_db, {
        "id": "v1", "island": "gui", "tag": "t",
        "parent_id": None, "parent_b_id": None, "origin": "seed",
        "generation": 0, "score": 0, "eval_level": 0,
        "step": 0,
        "builder_cost": 0, "created_at": "2026-04-17T00:00:00",
    })
    insert_eval_result(tmp_db, "v1", "task_a", 1, "t")
    insert_eval_result(tmp_db, "v1", "task_b", 0, "t")
    insert_eval_result(tmp_db, "v1", "task_c", 1, "t")
    # Restrict to 2 train tasks → only counts those.
    assert recompute_score_from_train_set(tmp_db, "v1", ["task_a", "task_b"]) == 1
    # Wider train set picks up task_c too.
    assert recompute_score_from_train_set(
        tmp_db, "v1", ["task_a", "task_b", "task_c"],
    ) == 2
    # Empty train set returns 0 by spec.
    assert recompute_score_from_train_set(tmp_db, "v1", []) == 0


def test_init_island_row(tmp_db):
    """init_island creates a row with defaults if none exists; idempotent."""
    from agent_s4.db import init_island, get_island
    init_island(tmp_db, "gui")
    got = get_island(tmp_db, "gui")
    assert got["name"] == "gui"
    assert got["mutation_count"] == 0
    assert got["last_migration_at"] == 0
    # Idempotent: second call doesn't error
    init_island(tmp_db, "gui")
    got2 = get_island(tmp_db, "gui")
    assert got2["mutation_count"] == 0


def test_update_island_fields(tmp_db):
    from agent_s4.db import init_island, update_island, get_island
    init_island(tmp_db, "gui")
    update_island(tmp_db, "gui", mutation_count=5, last_migration_at=5)
    got = get_island(tmp_db, "gui")
    assert got["mutation_count"] == 5
    assert got["last_migration_at"] == 5


def test_append_history_entry(tmp_db):
    from agent_s4.db import append_history
    append_history(tmp_db, {
        "type": "mutation",
        "island": "gui",
        "parent": "gui-seed",
        "child": "gui-if-001",
        "score": 13,
        "cost": 4.78,
        "details": None,
        "timestamp": "2026-04-17T00:00:00",
    })
    row = tmp_db.execute("SELECT * FROM history WHERE child = ?", ("gui-if-001",)).fetchone()
    assert row["type"] == "mutation"
    assert row["score"] == 13


def test_cost_events_table_exists(tmp_db):
    """After init_schema, cost_events table exists."""
    cursor = tmp_db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cursor.fetchall()]
    assert "cost_events" in tables


def test_insert_cost_event_basic(tmp_db):
    """insert_cost_event stores a row with all fields."""
    from agent_s4.db import insert_cost_event
    insert_cost_event(
        tmp_db,
        event_type="build",
        island="gui",
        variant_id="gui-if-001",
        task_id=None,
        model="claude-opus-4-7",
        cost_usd=4.78,
        input_tokens=32,
        output_tokens=34531,
        cache_read_tokens=1709482,
        cache_creation_tokens=79258,
        timestamp="2026-04-17T00:00:00",
    )
    row = tmp_db.execute("SELECT * FROM cost_events WHERE variant_id = 'gui-if-001'").fetchone()
    assert row["event_type"] == "build"
    assert row["island"] == "gui"
    assert row["model"] == "claude-opus-4-7"
    assert abs(row["cost_usd"] - 4.78) < 1e-6
    assert row["input_tokens"] == 32
    assert row["output_tokens"] == 34531


def test_insert_cost_event_idempotent_on_build(tmp_db):
    """Re-inserting same build row is a no-op (task_id=NULL handled by unique index)."""
    from agent_s4.db import insert_cost_event
    for _ in range(2):
        insert_cost_event(
            tmp_db,
            event_type="build", island="gui", variant_id="gui-if-001",
            task_id=None, model="opus", cost_usd=1.0,
            input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_creation_tokens=None,
            timestamp="2026-04-17T00:00:00",
        )
    count = tmp_db.execute(
        "SELECT COUNT(*) FROM cost_events WHERE variant_id = 'gui-if-001'"
    ).fetchone()[0]
    assert count == 1


def test_insert_cost_event_different_timestamps_not_collapsed(tmp_db):
    """Same variant with different timestamps produces two rows."""
    from agent_s4.db import insert_cost_event
    for ts in ["2026-04-17T00:00:00", "2026-04-17T00:00:01"]:
        insert_cost_event(
            tmp_db,
            event_type="build", island="gui", variant_id="gui-if-001",
            task_id=None, model="opus", cost_usd=1.0,
            input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_creation_tokens=None,
            timestamp=ts,
        )
    count = tmp_db.execute(
        "SELECT COUNT(*) FROM cost_events WHERE variant_id = 'gui-if-001'"
    ).fetchone()[0]
    assert count == 2


def test_cost_event_survives_variant_delete(tmp_db):
    """cost_events rows are NOT cascaded when their variant is deleted."""
    from agent_s4.db import insert_variant, delete_variant, insert_cost_event
    insert_variant(tmp_db, {
        "id": "gui-if-001", "island": "gui", "tag": "t",
        "parent_id": None, "parent_b_id": None, "origin": "seed",
        "generation": 0, "score": 10, "eval_level": 25,
        "builder_cost": 0, "created_at": "2026-04-17T00:00:00",
    })
    insert_cost_event(
        tmp_db,
        event_type="build", island="gui", variant_id="gui-if-001",
        task_id=None, model="opus", cost_usd=4.78,
        input_tokens=None, output_tokens=None,
        cache_read_tokens=None, cache_creation_tokens=None,
        timestamp="2026-04-17T00:00:00",
    )
    delete_variant(tmp_db, "gui-if-001")
    count = tmp_db.execute(
        "SELECT COUNT(*) FROM cost_events WHERE variant_id = 'gui-if-001'"
    ).fetchone()[0]
    assert count == 1, "cost_events must survive variant deletion"


def test_insert_cost_event_rejects_unknown_field(tmp_db):
    """insert_cost_event raises ValueError when given a column that doesn't exist."""
    import pytest
    from agent_s4.db import insert_cost_event
    with pytest.raises(ValueError, match="bogus_column"):
        insert_cost_event(
            tmp_db,
            event_type="build", island="gui", variant_id="v1",
            task_id=None, model="opus", cost_usd=1.0,
            input_tokens=None, output_tokens=None,
            cache_read_tokens=None, cache_creation_tokens=None,
            timestamp="2026-04-17T00:00:00",
            bogus_column="bad",
        )
