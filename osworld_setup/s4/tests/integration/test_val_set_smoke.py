"""End-to-end smoke for val-set wiring with synthetic eval data.

Exercises the path: val_set populated; per-task eval_results inserted
covering both train and val task IDs; recompute_score_from_train_set
and recompute_val_score correctly partition the two.
"""
from agent_s4.db import (
    connect, init_schema, insert_variant, insert_eval_result,
    load_val_set, recompute_score_from_train_set, recompute_val_score,
)


def test_seed_eval_populates_both_scores(tmp_path):
    """With val_set populated and per-task results covering both train and val,
    score and val_score correctly partition by membership."""
    db_path = tmp_path / "test.db"
    conn = connect(str(db_path))
    init_schema(conn)

    # Pre-populate val_set table (would normally happen at orchestrator init).
    val_ids = ["val_a", "val_b"]
    load_val_set(conn, val_ids)
    train_ids = ["train_x", "train_y", "train_z"]

    insert_variant(conn, {
        "id": "t-seed", "island": "t", "tag": "evol/t/seed",
        "parent_id": None, "parent_b_id": None, "origin": "seed",
        "generation": 0, "score": None, "val_score": None, "eval_level": 0,
        "step": 0,
        "builder_cost": 0.0, "created_at": "2026-05-04T00:00:00Z",
    })

    # Synthesize eval_results: train_x and train_y pass; val_a passes.
    now = "2026-05-04T00:00:00Z"
    insert_eval_result(conn, "t-seed", "train_x", 1, now)
    insert_eval_result(conn, "t-seed", "train_y", 1, now)
    insert_eval_result(conn, "t-seed", "train_z", 0, now)
    insert_eval_result(conn, "t-seed", "val_a", 1, now)
    insert_eval_result(conn, "t-seed", "val_b", 0, now)

    # Both score and val_score should compute correctly.
    assert recompute_score_from_train_set(conn, "t-seed", train_ids) == 2
    assert recompute_val_score(conn, "t-seed") == 1   # 1 of 2 val passed

    conn.close()


def test_val_set_disabled_keeps_val_score_zero(tmp_path):
    """When val_set is empty (--val-set none), val_score returns 0."""
    db_path = tmp_path / "test.db"
    conn = connect(str(db_path))
    init_schema(conn)

    insert_variant(conn, {
        "id": "t-seed", "island": "t", "tag": "evol/t/seed",
        "parent_id": None, "parent_b_id": None, "origin": "seed",
        "generation": 0, "score": None, "val_score": None, "eval_level": 0,
        "step": 0,
        "builder_cost": 0.0, "created_at": "2026-05-04T00:00:00Z",
    })

    insert_eval_result(conn, "t-seed", "train_x", 1, "2026-05-04T00:00:00Z")
    # val_set deliberately empty → no rows match the JOIN → val_score = 0.
    assert recompute_val_score(conn, "t-seed") == 0

    conn.close()
