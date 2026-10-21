"""Tournament ranks parents by val_score under selection_signal=val."""
import random


from agent_s4.db import insert_variant, update_variant
from agent_s4.tournament import tournament


def _v(tmp_db, vid, score, val_score):
    insert_variant(tmp_db, {
        "id": vid, "island": "t", "tag": f"evol/t/{vid}",
        "parent_id": None, "parent_b_id": None, "origin": "seed",
        "generation": 0, "score": score, "eval_level": 100,
        "builder_cost": 0.0, "created_at": "2026-05-04T00:00:00Z",
    })
    update_variant(tmp_db, vid, val_score=val_score)
    return {"id": vid, "score": score, "val_score": val_score,
            "eval_level": 100, "tag": f"evol/t/{vid}"}


def test_val_signal_ranks_by_val_score(tmp_db, tmp_path):
    """Tournament under selection_signal=val picks the highest val_score."""
    random.seed(0)
    candidates = [
        _v(tmp_db, "v_high_train", score=80, val_score=10),
        _v(tmp_db, "v_balanced", score=60, val_score=30),
        _v(tmp_db, "v_high_val", score=40, val_score=45),
    ]
    winners = tournament(
        tmp_db, candidates, k=3, t=1, clone_dir=str(tmp_path),
        eval_model="haiku", eval_timeout=60, region="us-east-1",
        selection_signal="val",
    )
    assert winners[0]["id"] == "v_high_val"


def test_train_signal_ranks_by_score_regression(tmp_db, tmp_path):
    """Backward compat: selection_signal=train preserves today's behavior."""
    random.seed(0)
    candidates = [
        _v(tmp_db, "v_high_train", score=80, val_score=10),
        _v(tmp_db, "v_balanced", score=60, val_score=30),
        _v(tmp_db, "v_high_val", score=40, val_score=45),
    ]
    winners = tournament(
        tmp_db, candidates, k=3, t=1, clone_dir=str(tmp_path),
        eval_model="haiku", eval_timeout=60, region="us-east-1",
        selection_signal="train",
    )
    assert winners[0]["id"] == "v_high_train"


def test_val_ties_broken_by_train_score(tmp_db, tmp_path):
    """When val_score ties, secondary key is train score."""
    random.seed(0)
    candidates = [
        _v(tmp_db, "tie_low_train", score=20, val_score=25),
        _v(tmp_db, "tie_high_train", score=70, val_score=25),
        _v(tmp_db, "loser", score=50, val_score=10),
    ]
    winners = tournament(
        tmp_db, candidates, k=3, t=2, clone_dir=str(tmp_path),
        eval_model="haiku", eval_timeout=60, region="us-east-1",
        selection_signal="val",
    )
    assert winners[0]["id"] == "tie_high_train"
    assert winners[1]["id"] == "tie_low_train"
