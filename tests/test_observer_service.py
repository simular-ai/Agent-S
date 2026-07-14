import asyncio
import hashlib
import json
import re
from pathlib import Path

import pytest

from gui_agents.s3.observer.capture import CapturedObservation
from gui_agents.s3.observer.mcp_server import mcp
from gui_agents.s3.observer.planner import AgentSObserverPlanner
from gui_agents.s3.observer.service import ObserverService


PNG = b"synthetic-png-with-stable-screen-state"


class FakeCapture:
    def __init__(self):
        self.calls = 0

    def capture(self):
        self.calls += 1
        return CapturedObservation(
            png=PNG,
            width=1920,
            height=1080,
            captured_at="2026-07-13T00:00:00+00:00",
            sha256=hashlib.sha256(PNG).hexdigest(),
        )


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, messages, *, max_tokens):
        self.calls.append((messages, max_tokens))
        return self.response


def test_proposal_does_not_change_observation_or_expose_executor():
    capture = FakeCapture()
    main = FakeClient(
        """(Screenshot Analysis) A synthetic page is visible.
(Next Action) Select the blue Continue button.
(Grounded Action)
```python
agent.click("The blue Continue button near the center of the page")
```"""
    )
    grounding = FakeClient("(820, 604)")
    planner = AgentSObserverPlanner(main, grounding)
    service = ObserverService(capture=capture, planner_factory=lambda: planner)

    task = service.start_task("Identify the next step on the synthetic page")
    before = service.observe()
    proposal = service.propose_next(task["task_id"])
    after = service.observe()

    assert before.sha256 == after.sha256 == proposal["screenshot_sha256"]
    assert proposal["action"]["kind"] == "click"
    assert proposal["target"] == {"x": 820, "y": 604}
    assert proposal["risk_class"] == "proposal_only"
    assert capture.calls == 3


def test_task_limit_and_reset():
    capture = FakeCapture()
    main = FakeClient(
        "(Screenshot Analysis) Stable.\n(Next Action) Wait.\n"
        "(Grounded Action)\n```python\nagent.wait(1)\n```"
    )
    planner = AgentSObserverPlanner(main, FakeClient("(1, 1)"), max_steps=1)
    service = ObserverService(capture=capture, planner_factory=lambda: planner)
    task_id = service.start_task("Observe the stable page")["task_id"]
    service.propose_next(task_id)
    with pytest.raises(RuntimeError, match="proposal limit"):
        service.propose_next(task_id)
    assert service.reset_task(task_id)["reset"] is True
    assert service.status()["active_task_id"] is None


def test_status_reports_sealed_build_identity(tmp_path, monkeypatch):
    metadata_path = tmp_path / "observer-build.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_commit": "a" * 40,
                "source_dirty": False,
                "built_at": "2026-07-14T00:00:00Z",
                "source_archive_sha256": "b" * 64,
                "requirements_lock_sha256": "c" * 64,
            }
        )
    )
    monkeypatch.setenv("AGENT_S_BUILD_METADATA", str(metadata_path))

    build = ObserverService(capture=FakeCapture()).status()["build"]

    assert build == {
        "status": "sealed",
        "source_commit": "a" * 40,
        "source_dirty": False,
        "built_at": "2026-07-14T00:00:00Z",
        "source_archive_sha256": "b" * 64,
        "requirements_lock_sha256": "c" * 64,
    }


def test_status_reports_development_identity_when_metadata_is_absent(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(
        "AGENT_S_BUILD_METADATA", str(tmp_path / "missing-observer-build.json")
    )

    build = ObserverService(capture=FakeCapture()).status()["build"]

    assert build["status"] == "development"
    assert build["source_commit"] == "unknown"


def test_mcp_exposes_observation_tools_only():
    tools = asyncio.run(mcp.list_tools())
    assert {tool.name for tool in tools} == {
        "status",
        "observe",
        "start_task",
        "propose_next",
        "reset_task",
    }
    assert all("execute" not in tool.name for tool in tools)


def test_hardened_runtime_has_no_desktop_executor_imports():
    observer_dir = Path(__file__).parents[1] / "gui_agents" / "s3" / "observer"
    source = "\n".join(path.read_text() for path in observer_dir.glob("*.py"))
    assert "pyautogui" not in source
    assert "subprocess" not in source
    assert "osworld-public-evaluation" not in source
    assert "sudo -S" not in source


def test_bridge_disables_xtrace_before_loading_credentials():
    bridge = (
        Path(__file__).parents[1] / "scripts" / "agent_s_vm" / "codex-mcp-bridge.sh"
    ).read_text()
    assert bridge.index("set +x") < bridge.index("HF_TOKEN=")


def test_vm_lifecycle_detects_orphaned_observer_processes():
    scripts = Path(__file__).parents[1] / "scripts" / "agent_s_vm"
    assert "observer_is_running" in (scripts / "start.sh").read_text()
    assert "observer_is_running" in (scripts / "build_base.sh").read_text()
    assert "observer_pids" in (scripts / "status.sh").read_text()
    assert "observer_pids" in (scripts / "stop.sh").read_text()


def test_s3_has_no_dynamic_execution_or_benchmark_password():
    s3_dir = Path(__file__).parents[1] / "gui_agents" / "s3"
    source = "\n".join(path.read_text() for path in s3_dir.rglob("*.py"))
    assert re.search(r"\b(?:eval|exec)\s*\(", source) is None
    assert "osworld-public-evaluation" not in source
    assert "sudo -S" not in source
