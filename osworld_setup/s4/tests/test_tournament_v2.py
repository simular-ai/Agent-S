from agent_s4.tournament import tournament


def _v(id, score, val_score):
    return {"id": id, "score": score, "val_score": val_score, "eval_level": 100,
            "island": "test"}


def test_tournament_ranks_by_val_score(monkeypatch):
    monkeypatch.setattr("random.sample", lambda pop, k: pop[:k])
    monkeypatch.setattr("random.random", lambda: 0.5)
    candidates = [
        _v("a", 80, 60), _v("b", 70, 80), _v("c", 90, 50),
    ]
    winners = tournament(
        conn=None, candidates=candidates, k=3, t=1,
        clone_dir=".", eval_model="haiku",
        eval_timeout=600, region="us-east-1",
        selection_signal="val",
    )
    assert winners[0]["id"] == "b"  # highest val_score


def test_tournament_ranks_by_train_score(monkeypatch):
    monkeypatch.setattr("random.sample", lambda pop, k: pop[:k])
    monkeypatch.setattr("random.random", lambda: 0.5)
    candidates = [
        _v("a", 80, 60), _v("b", 70, 80), _v("c", 90, 50),
    ]
    winners = tournament(
        conn=None, candidates=candidates, k=3, t=1,
        clone_dir=".", eval_model="haiku",
        eval_timeout=600, region="us-east-1",
        selection_signal="train",
    )
    assert winners[0]["id"] == "c"  # highest train score
