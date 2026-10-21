"""Validation repeats use real SQLite state and isolated tagged Git trees."""
import json
import random
import subprocess
import threading
from pathlib import Path

import pytest

from agent_s4.config.experiment import ExperimentConfig
from agent_s4.config.island import IslandConfig
from agent_s4.db import (
    complete_reevaluation_round, connect, get_current_step, get_population,
    get_reevaluation_round, get_variant, init_schema, insert_eval_result,
    insert_variant, load_val_set, recompute_val_score, set_current_step,
    start_reevaluation_round, update_variant,
)
from agent_s4.git_utils import next_variant_num_from_population_and_tags
from agent_s4.harness.base import EvalResults
from agent_s4.operators import reevaluation as module
from agent_s4.operators.reevaluation import Reevaluation
from agent_s4.state import LoopState


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def state(tmp_path):
    clone = tmp_path / "OSWorld_Evol_a"
    (clone / "shadow_interface").mkdir(parents=True)
    git(clone, "init")
    git(clone, "config", "user.name", "Test")
    git(clone, "config", "user.email", "test@example.invalid")
    conn = connect(str(tmp_path / "shadow_evolution.db"))
    init_schema(conn)
    load_val_set(conn, ["val1", "val2", "val3"])
    for i, score in enumerate([2, 3, 1, 0], 1):
        vid = f"a-if-{i:03d}"
        tag = f"evol/a/if-{i:03d}"
        (clone / "shadow_interface" / "identity.txt").write_text(vid)
        git(clone, "add", ".")
        git(clone, "commit", "-m", vid)
        git(clone, "tag", tag)
        insert_variant(conn, {
            "id": vid, "island": "a", "tag": tag, "origin": "mutation",
            "score": 20 - i, "val_score": score, "step": i, "generation": i,
        })
        for j in range(1, 4):
            insert_eval_result(conn, vid, f"val{j}", int(j <= score), "original")
        insert_eval_result(conn, vid, "train1", 1, "original")
    # Dirty live code must not be evaluated or changed by a tagged repeat.
    (clone / "shadow_interface" / "identity.txt").write_text("uncommitted")
    (clone / "keep.txt").write_text("leave this alone")
    config = ExperimentConfig(
        islands=[IslandConfig(name="a", evaluator_harness="claude_code", eval_model="m")],
        reevaluation_interval=5, reevaluation_top_k=3,
        eval_timeout=47, eval_max_steps=None, seed=1,
    )
    set_current_step(conn, 5)
    yield LoopState(conn, "a", 5, random.Random(1), config, clone, tmp_path,
                    tmp_path / "eval_cache")
    conn.close()


class FakeEvaluator:
    def __init__(self, *, scores=None, barrier=None, fail=None, corrupt=None):
        self.scores = scores or {"a-if-001": 0, "a-if-002": 2, "a-if-003": 3}
        self.barrier = barrier
        self.fail = fail
        self.corrupt = corrupt
        self.calls = []

    def run_tasks(self, **kw):
        vid = (kw["shadow_dir"] / "identity.txt").read_text()
        self.calls.append((vid, kw))
        if self.barrier:
            self.barrier.wait(timeout=10)
        if self.fail == vid:
            raise RuntimeError("simulated evaluator failure")
        results = []
        for i, task in enumerate(kw["task_ids"]):
            score = float(i < self.scores.get(vid, 0))
            directory = kw["result_dir"] / "shadow_eval/screenshot" / kw["model"] / task
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "result.txt").write_text(str(score))
            (directory / "result.json").write_text(json.dumps({"score": score}))
            (directory / "claude_trajectory.jsonl").write_text(json.dumps({
                "type": "result", "total_cost_usd": 0.01, "usage": {"input_tokens": 10},
            }) + "\n")
            results.append({"task_id": task, "score": score, "valid_for_fitness": True})
        result = EvalResults(
            pass_rate=sum(r["score"] for r in results) / len(results),
            total=len(results), passed=sum(r["score"] for r in results),
            failed=len(results) - sum(r["score"] for r in results), results=results,
        )
        if self.corrupt:
            self.corrupt(result, kw)
        return result


def install(monkeypatch, evaluator):
    monkeypatch.setattr(module, "resolve_evaluator_harness", lambda *_: evaluator)


def test_top_three_run_in_parallel_from_tags_and_update_selection_scores(state, monkeypatch):
    fake = FakeEvaluator(barrier=threading.Barrier(3))
    install(monkeypatch, fake)
    before = {v["id"]: v for v in get_population(state.conn, "a")}
    original_results = [tuple(r) for r in state.conn.execute("SELECT * FROM eval_results")]
    head = git(state.clone_dir, "rev-parse", "HEAD")
    dirty = git(state.clone_dir, "status", "--porcelain")

    Reevaluation().apply(state)

    assert {vid for vid, _ in fake.calls} == {"a-if-001", "a-if-002", "a-if-003"}
    assert len({kw["clone_dir"] for _, kw in fake.calls}) == 3
    for vid, kw in fake.calls:
        assert kw["task_ids"] == ["val1", "val2", "val3"]
        assert kw["clone_dir"].parent == state.base_dir
        assert kw["clone_dir"] != state.clone_dir
        assert kw["timeout"] == 47
        assert kw["result_dir"] == (
            state.eval_cache_dir / vid / "validation_reevaluations/step_000005"
        )
        assert not kw["clone_dir"].exists()
    after = {v["id"]: v for v in get_population(state.conn, "a")}
    assert {v: row["val_score"] for v, row in after.items()} == {
        "a-if-001": 1, "a-if-002": 2.5, "a-if-003": 2, "a-if-004": 0,
    }
    for vid in before:
        assert {k: v for k, v in before[vid].items() if k != "val_score"} == {
            k: v for k, v in after[vid].items() if k != "val_score"
        }
    assert [tuple(r) for r in state.conn.execute("SELECT * FROM eval_results")] == original_results
    assert get_current_step(state.conn) == 5
    assert next_variant_num_from_population_and_tags(list(after.values()), str(state.clone_dir)) == 5
    assert git(state.clone_dir, "rev-parse", "HEAD") == head
    assert git(state.clone_dir, "status", "--porcelain") == dirty
    assert git(state.clone_dir, "worktree", "list", "--porcelain").count("worktree ") == 1
    assert state.conn.execute("SELECT COUNT(*) FROM validation_score_samples").fetchone()[0] == 6
    assert state.conn.execute("SELECT COUNT(*) FROM cost_events").fetchone()[0] == 9
    assert get_reevaluation_round(state.conn, "a", 5)["status"] == "complete"
    from agent_s4.tournament import tournament
    ranked = tournament(state.conn, list(after.values()), 4, 4, str(state.clone_dir),
                        "m", 47, "us-east-1")
    assert [v["id"] for v in ranked] == ["a-if-002", "a-if-003", "a-if-001", "a-if-004"]

    Reevaluation().apply(state)
    assert len(fake.calls) == 3
    assert state.conn.execute("SELECT COUNT(*) FROM validation_score_samples").fetchone()[0] == 6
    assert state.conn.execute("SELECT COUNT(*) FROM cost_events").fetchone()[0] == 9


def test_repeated_scores_are_an_equal_weight_mean_and_commit_is_idempotent(state):
    vid = "a-if-001"
    for step, score in [(5, 3), (10, 0)]:
        candidate = get_variant(state.conn, vid)
        start_reevaluation_round(state.conn, "a", step, [candidate],
                                 {"task_ids": ["val1", "val2", "val3"]}, str(step))
        outcomes = [{"variant_id": vid, "score": score, "records": {},
                     "evaluated_at": str(step), "result_dir": f"round-{step}"}]
        complete_reevaluation_round(state.conn, "a", step, outcomes, str(step))
        complete_reevaluation_round(state.conn, "a", step, outcomes, str(step))
    assert get_variant(state.conn, vid)["val_score"] == pytest.approx((2 + 3 + 0) / 3)
    assert recompute_val_score(state.conn, vid) == pytest.approx(5 / 3)
    assert state.conn.execute("SELECT COUNT(*) FROM validation_score_samples").fetchone()[0] == 3
    assert state.conn.execute("SELECT COUNT(*) FROM history").fetchone()[0] == 2


def test_failed_round_keeps_scores_and_selection_then_resumes_once(state, monkeypatch):
    fake = FakeEvaluator(fail="a-if-003")
    install(monkeypatch, fake)
    before = get_population(state.conn, "a")
    with pytest.raises(RuntimeError, match="scores unchanged"):
        Reevaluation().apply(state)
    assert get_population(state.conn, "a") == before
    assert get_reevaluation_round(state.conn, "a", 5)["status"] == "pending"
    assert state.conn.execute("SELECT COUNT(*) FROM validation_score_samples").fetchone()[0] == 0
    assert state.conn.execute("SELECT COUNT(*) FROM cost_events").fetchone()[0] == 6
    first_paths = {vid: kw["result_dir"] for vid, kw in fake.calls}
    # Even if the ranking changes between attempts, finish the pinned round.
    update_variant(state.conn, "a-if-004", val_score=3)
    fake.fail = None
    fake.calls.clear()
    Reevaluation().apply(state)
    assert {vid: kw["result_dir"] for vid, kw in fake.calls} == first_paths
    assert get_variant(state.conn, "a-if-004")["val_score"] == 3
    assert state.conn.execute("SELECT COUNT(*) FROM cost_events").fetchone()[0] == 9
    assert state.conn.execute("SELECT COUNT(*) FROM validation_score_samples").fetchone()[0] == 6


@pytest.mark.parametrize("fault", ["timeout", "missing", "duplicate", "infra", "mismatch", "nan"])
def test_incomplete_or_invalid_evaluations_do_not_change_means(state, monkeypatch, fault):
    state.config.reevaluation_top_k = 1
    def corrupt(out, kw):
        if fault == "timeout":
            out.timed_out = True
        elif fault == "missing":
            out.results.pop()
        elif fault == "duplicate":
            out.results[1]["task_id"] = out.results[0]["task_id"]
        elif fault == "infra":
            out.results[0]["valid_for_fitness"] = False
        elif fault == "mismatch":
            out.results[0]["score"] = 0.5
        elif fault == "nan":
            p = kw["result_dir"] / "shadow_eval/screenshot/m/val1/result.txt"
            p.write_text("nan")
    install(monkeypatch, FakeEvaluator(corrupt=corrupt))
    before = get_population(state.conn, "a")
    with pytest.raises(RuntimeError, match="scores unchanged"):
        Reevaluation().apply(state)
    assert get_population(state.conn, "a") == before
    assert state.conn.execute("SELECT COUNT(*) FROM validation_score_samples").fetchone()[0] == 0


def test_next_round_uses_fresh_cache_and_reranks_by_updated_means(state, monkeypatch):
    fake = FakeEvaluator()
    install(monkeypatch, fake)
    Reevaluation().apply(state)
    first_paths = {kw["result_dir"] for _, kw in fake.calls}
    # The next round ranks current means, including newly competitive variants.
    update_variant(state.conn, "a-if-004", val_score=3)
    state.step = 10
    fake.calls.clear()
    Reevaluation().apply(state)
    assert {vid for vid, _ in fake.calls} == {"a-if-002", "a-if-003", "a-if-004"}
    assert not first_paths & {kw["result_dir"] for _, kw in fake.calls}
    assert all(kw["result_dir"].name == "step_000010" for _, kw in fake.calls)
    assert get_variant(state.conn, "a-if-002")["val_score"] == pytest.approx((3 + 2 + 2) / 3)
    assert get_variant(state.conn, "a-if-003")["val_score"] == pytest.approx((1 + 3 + 3) / 3)
    assert get_variant(state.conn, "a-if-004")["val_score"] == 1.5
    assert get_variant(state.conn, "a-if-001")["val_score"] == 1


def test_fitness_valid_imputed_zero_uses_existing_harness_policy(state, monkeypatch):
    state.config.reevaluation_top_k = 1
    def impute(out, kw):
        out.results[-1]["imputed_zero"] = True
    install(monkeypatch, FakeEvaluator(corrupt=impute))
    Reevaluation().apply(state)
    assert get_variant(state.conn, "a-if-002")["val_score"] == 2.5


def test_v2_partial_credit_and_full_credit_rounding_are_preserved(state, monkeypatch):
    state.config.reevaluation_top_k = 1
    state.config.benchmark = "osworld_v2"
    def fractional(out, kw):
        for result, score in zip(out.results, [0.25, 0.5, 0.9999999999999999]):
            result["score"] = score
            path = kw["result_dir"] / "shadow_eval/screenshot/m" / result["task_id"]
            (path / "result.txt").write_text(str(score))
            (path / "result.json").write_text(json.dumps({"score": score}))
    install(monkeypatch, FakeEvaluator(corrupt=fractional))
    Reevaluation().apply(state)
    assert get_variant(state.conn, "a-if-002")["val_score"] == pytest.approx((3 + 1.75) / 2)
    sample = state.conn.execute(
        "SELECT task_results_json FROM validation_score_samples WHERE after_step = 5",
    ).fetchone()[0]
    assert json.loads(sample)["val1"]["fitness_score"] == 0.25
    assert json.loads(sample)["val3"]["fitness_score"] == 1


def test_pending_round_rejects_changed_settings(state, monkeypatch):
    fake = FakeEvaluator(fail="a-if-003")
    install(monkeypatch, fake)
    with pytest.raises(RuntimeError):
        Reevaluation().apply(state)
    state.config.eval_timeout += 1
    fake.calls.clear()
    with pytest.raises(ValueError, match="settings changed"):
        Reevaluation().apply(state)
    assert not fake.calls


def test_fewer_than_k_scored_variants_and_empty_population(state, monkeypatch):
    for vid in ["a-if-001", "a-if-003", "a-if-004"]:
        update_variant(state.conn, vid, val_score=None)
    fake = FakeEvaluator()
    install(monkeypatch, fake)
    Reevaluation().apply(state)
    assert [vid for vid, _ in fake.calls] == ["a-if-002"]
    state.step = 10
    update_variant(state.conn, "a-if-002", val_score=None)
    Reevaluation().apply(state)
    assert get_reevaluation_round(state.conn, "a", 10)["status"] == "complete"
    assert len(fake.calls) == 1


def test_round_commit_rolls_back_all_scores_when_any_outcome_is_invalid(state):
    candidates = get_population(state.conn, "a")[:2]
    start_reevaluation_round(state.conn, "a", 5, candidates,
                             {"task_ids": ["val1", "val2", "val3"]}, "start")
    outcomes = [{"variant_id": candidate["id"], "score": score, "records": {},
                 "evaluated_at": "end", "result_dir": "cache"}
                for candidate, score in zip(candidates, [1, 99])]
    with pytest.raises(ValueError, match="outside"):
        complete_reevaluation_round(state.conn, "a", 5, outcomes, "end")
    assert get_population(state.conn, "a")[:2] == candidates
    assert state.conn.execute("SELECT COUNT(*) FROM validation_score_samples").fetchone()[0] == 0
    assert state.conn.execute("SELECT COUNT(*) FROM history").fetchone()[0] == 0
    assert get_reevaluation_round(state.conn, "a", 5)["status"] == "pending"
