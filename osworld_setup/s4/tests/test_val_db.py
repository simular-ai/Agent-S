"""Tests for val_score column, val_set table, and val helpers in db.py."""

from agent_s4.db import (
    insert_variant, get_variant, update_variant,
    insert_eval_result,
    recompute_val_score, get_val_task_ids, load_val_set,
)


def _now():
    return "2026-05-04T00:00:00Z"


def _seed_variant(conn, vid="v1"):
    insert_variant(conn, {
        "id": vid, "island": "t", "tag": "evol/t/seed",
        "parent_id": None, "parent_b_id": None, "origin": "seed",
        "generation": 0, "score": None, "eval_level": 0,
        "builder_cost": 0.0, "created_at": _now(),
    })


def test_val_score_column_exists_and_defaults_null(tmp_db):
    _seed_variant(tmp_db)
    v = get_variant(tmp_db, "v1")
    assert "val_score" in v
    assert v["val_score"] is None


def test_val_score_can_be_updated(tmp_db):
    _seed_variant(tmp_db)
    update_variant(tmp_db, "v1", val_score=37)
    v = get_variant(tmp_db, "v1")
    assert v["val_score"] == 37


def test_load_val_set_populates_table(tmp_db):
    load_val_set(tmp_db, ["t1", "t2", "t3"])
    rows = tmp_db.execute("SELECT task_id FROM val_set ORDER BY task_id").fetchall()
    assert [r["task_id"] for r in rows] == ["t1", "t2", "t3"]


def test_load_val_set_replaces_prior_content(tmp_db):
    load_val_set(tmp_db, ["a", "b"])
    load_val_set(tmp_db, ["c"])
    rows = tmp_db.execute("SELECT task_id FROM val_set").fetchall()
    assert [r["task_id"] for r in rows] == ["c"]


def test_get_val_task_ids_returns_list(tmp_db):
    load_val_set(tmp_db, ["x", "y", "z"])
    ids = get_val_task_ids(tmp_db)
    assert set(ids) == {"x", "y", "z"}


def test_get_val_task_ids_empty_when_unloaded(tmp_db):
    assert list(get_val_task_ids(tmp_db)) == []


def test_recompute_val_score_zero_when_no_results(tmp_db):
    _seed_variant(tmp_db)
    load_val_set(tmp_db, ["t1", "t2"])
    assert recompute_val_score(tmp_db, "v1") == 0


def test_recompute_val_score_counts_only_passed_val_tasks(tmp_db):
    _seed_variant(tmp_db)
    load_val_set(tmp_db, ["v1_task", "v2_task"])
    insert_eval_result(tmp_db, "v1", "v1_task", 1, _now())
    insert_eval_result(tmp_db, "v1", "v2_task", 0, _now())
    insert_eval_result(tmp_db, "v1", "non_val_task", 1, _now())
    assert recompute_val_score(tmp_db, "v1") == 1


def test_recompute_val_score_all_pass(tmp_db):
    _seed_variant(tmp_db)
    load_val_set(tmp_db, ["a", "b", "c"])
    for tid in ["a", "b", "c"]:
        insert_eval_result(tmp_db, "v1", tid, 1, _now())
    assert recompute_val_score(tmp_db, "v1") == 3


def test_recompute_val_score_recomputes_after_update(tmp_db):
    _seed_variant(tmp_db)
    load_val_set(tmp_db, ["a"])
    insert_eval_result(tmp_db, "v1", "a", 0, _now())
    assert recompute_val_score(tmp_db, "v1") == 0
    tmp_db.execute(
        "INSERT OR REPLACE INTO eval_results(variant_id, task_id, passed, evaluated_at) "
        "VALUES (?, ?, ?, ?)", ("v1", "a", 1, _now()),
    )
    assert recompute_val_score(tmp_db, "v1") == 1
