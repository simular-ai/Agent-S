"""Exercise buffered compaction with native and legacy Prime event names."""
from __future__ import annotations

import json
import sys

import pytest

from agent_s4.harness.prime.step_runner import run


@pytest.mark.parametrize("prefix", ["", "auto_"])
def test_compaction_uses_model_budget_then_resumes(prefix, capsys, tmp_path):
    # Compaction is a separate model call after agent_end. It has no ordinary
    # turn_start and can legitimately exceed the tool/startup idle budget.
    start = json.dumps({"type": f"{prefix}compaction_start"}) + "\n"
    end = json.dumps({"type": f"{prefix}compaction_end"}) + "\n"
    child = (
        "import os,time; "
        "os.write(1,b'{\"type\":\"agent_end\"}\\n'); "
        f"os.write(1,{start.encode()!r}); time.sleep(1.3); "
        f"os.write(1,{end.encode()!r}); "
        "os.write(1,b'{\"type\":\"turn_start\"}\\n"
        "{\"type\":\"message_end\",\"message\":{\"role\":\"assistant\"}}\\n'); "
        "time.sleep(30)"
    )
    state = tmp_path / "state.json"
    assert run([sys.executable, "-c", child], 1, 1, 4, str(state)) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["type"] == "prime_step_limit"
    assert not any("timeout" in event["type"] for event in events)
    assert json.loads(state.read_text())["phase"] == "between_events"


@pytest.mark.parametrize("prefix", ["", "auto_"])
def test_compaction_end_restores_idle_budget(prefix, capsys):
    events = "".join(json.dumps({"type": f"{prefix}compaction_{suffix}"}) + "\n"
                     for suffix in ("start", "end"))
    child = f"import os,time; os.write(1,{events.encode()!r}); time.sleep(30)"
    assert run([sys.executable, "-c", child], 10, 1, 4) == 125
    timeout = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert timeout["type"] == "prime_agent_idle_timeout"
    assert timeout["phase"] == "between_events"
    assert timeout["budget_seconds"] == 1


def test_repeated_compaction_start_does_not_extend_model_deadline(capsys):
    child = (
        "import os,time\n"
        "while True:\n"
        " os.write(1,b'{\"type\":\"compaction_start\"}\\n')\n"
        " time.sleep(0.1)\n"
    )
    assert run([sys.executable, "-c", child], 10, 1, 2) == 125
    timeout = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert timeout["type"] == "prime_agent_model_timeout"
    assert timeout["phase"] == "model"
    assert timeout["budget_seconds"] == 2
    assert 2 <= timeout["elapsed_seconds"] < 4


def test_silent_compaction_after_agent_end_uses_model_budget(capsys, tmp_path):
    # Prime can start a threshold compaction without emitting
    # compaction_start; the first event after agent_end is compaction_end.
    child = (
        "import os,time; "
        "os.write(1,b'{\"type\":\"agent_end\"}\\n'); time.sleep(1.3); "
        "os.write(1,b'{\"type\":\"compaction_end\"}\\n"
        "{\"type\":\"turn_start\"}\\n"
        "{\"type\":\"message_end\",\"message\":{\"role\":\"assistant\"}}\\n'); "
        "time.sleep(30)"
    )
    state = tmp_path / "state.json"
    assert run([sys.executable, "-c", child], 1, 1, 4, str(state)) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["type"] == "prime_step_limit"
    assert not any("timeout" in event["type"] for event in events)


def test_agent_end_without_follow_up_hits_model_budget(capsys):
    child = "import os,time; os.write(1,b'{\"type\":\"agent_end\"}\\n'); time.sleep(30)"
    assert run([sys.executable, "-c", child], 10, 1, 2) == 125
    timeout = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert timeout["type"] == "prime_agent_model_timeout"
    assert timeout["last_event"] == "agent_end"
    assert timeout["budget_seconds"] == 2
