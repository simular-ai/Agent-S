from agent_s4.orchestrator import _seed_tasks_needing_eval


def _write_result(root, model, task_id, value):
    path = root / "shadow_eval" / "screenshot" / model / task_id / "result.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def test_seed_resume_runs_only_missing_or_invalid_results(tmp_path):
    model = "openrouter/qwen/qwen3.8-27b"
    _write_result(tmp_path, model, "complete_zero", "0.0\n")
    _write_result(tmp_path, model, "complete_one", "1\n")
    _write_result(tmp_path, model, "malformed", "not-a-number\n")
    _write_result(tmp_path, model, "nan", "nan\n")
    _write_result(tmp_path, model, "out_of_range", "1.01\n")

    assert _seed_tasks_needing_eval(
        tmp_path,
        model,
        [
            "complete_zero",
            "complete_one",
            "missing",
            "malformed",
            "nan",
            "out_of_range",
        ],
    ) == ["missing", "malformed", "nan", "out_of_range"]
