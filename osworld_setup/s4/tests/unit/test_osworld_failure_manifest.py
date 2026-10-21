from __future__ import annotations

import json

from agent_s4.eval.osworld import clear_task_failure, persist_task_failure


def test_task_failure_manifest_is_independent_and_clearable(tmp_path):
    result = {
        "task_id": "048",
        "score": 0.0,
        "failure_type": "task_setup_failure",
        "error": "setup failed",
        "valid_for_fitness": False,
    }
    persist_task_failure(str(tmp_path), result, stage="task_setup")

    manifest = json.loads((tmp_path / "failure.json").read_text())
    assert manifest["task_id"] == "048"
    assert manifest["status"] == "infra_failure"
    assert manifest["stage"] == "task_setup"
    assert manifest["created_at"].endswith("+00:00")
    assert list(tmp_path.glob(".failure.*.json.tmp")) == []

    clear_task_failure(str(tmp_path))
    assert not (tmp_path / "failure.json").exists()
