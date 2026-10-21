from __future__ import annotations

import logging

import pytest

from agent_s4.vm import utils


class Controller:
    def __init__(self, marker):
        self.marker = marker

    def get_file(self, path):
        return self.marker


@pytest.mark.parametrize("status,expected", [(200, b"0\n"), (404, None), (503, None)])
def test_exit_marker_http_poll_treats_only_404_as_expected(status, expected, monkeypatch, caplog):
    from types import SimpleNamespace

    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=status, content=b"0\n")
    monkeypatch.setattr(utils.requests, "post", post)
    controller = SimpleNamespace(http_server="http://vm:5000")
    with caplog.at_level(logging.WARNING):
        assert utils._get_agent_exit_marker(controller) == expected
    assert calls == [("http://vm:5000/file", {
        "data": {"file_path": "/tmp/claude_task/exit_code.txt"}, "timeout": 15,
    })]
    assert bool(caplog.records) is (status == 503)


@pytest.mark.parametrize("marker,expected", [
    (b"0", True), (b"1", False), (b"124", False), (b"garbage", False),
])
def test_agent_driver_honors_exit_code(marker, expected, monkeypatch):
    monkeypatch.setattr(utils, "_launch_on_vm", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        utils, "_execute_shell_on_vm",
        lambda *args, **kwargs: {"returncode": 0},
    )
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)
    assert utils.run_agent_on_vm(Controller(marker), "command", 1) is expected


def test_agent_driver_timeout_is_failure_and_attempts_cleanup(monkeypatch):
    cleanup = []
    monkeypatch.setattr(utils, "_launch_on_vm", lambda *args, **kwargs: True)
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        utils, "_execute_shell_on_vm",
        lambda controller, command, timeout: cleanup.append(command) or {"returncode": 0},
    )
    assert utils.run_agent_on_vm(Controller(None), "command", 0) is False
    assert cleanup[0] == "rm -f /tmp/claude_task/exit_code.txt"
    assert "pkill" in cleanup[-1]
    assert "[q]wen -p" in cleanup[-1]
    assert "[p]rime-agent" in cleanup[-1]
    assert "approval-mode" not in cleanup[-1]


def test_agent_driver_stages_secrets_separately_from_launch(monkeypatch):
    launched = []
    staged = []
    monkeypatch.setattr(
        utils, "_execute_shell_on_vm", lambda *args, **kwargs: {"returncode": 0},
    )
    monkeypatch.setattr(
        utils, "_stage_agent_secret_env",
        lambda controller, env: staged.append(env) or True,
    )
    monkeypatch.setattr(
        utils, "_launch_on_vm",
        lambda controller, command, shell=True: launched.append(command) or True,
    )
    monkeypatch.setattr(utils.time, "sleep", lambda *_: None)

    class Controller:
        def get_file(self, _path):
            return b"0"

    command = utils.AgentCommand(
        "source-env-file && prime-agent", {"API_KEY": "secret"},
    )
    assert utils.run_agent_on_vm(Controller(), command, 1) is True
    assert staged == [{"API_KEY": "secret"}]
    assert launched == ["source-env-file && prime-agent"]
    assert "secret" not in launched[0]


def test_agent_driver_honors_command_specific_accepted_exit_codes(monkeypatch):
    monkeypatch.setattr(utils, "_launch_on_vm", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        utils, "_execute_shell_on_vm",
        lambda *args, **kwargs: {"returncode": 0},
    )
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)
    command = utils.AgentCommand("prime-agent", accepted_exit_codes=(0, 124))
    assert utils.run_agent_on_vm(Controller(b"124"), command, 1) is True


def test_agent_driver_accepts_no_wall_clock_limit(monkeypatch):
    monkeypatch.setattr(utils, "_launch_on_vm", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        utils, "_execute_shell_on_vm", lambda *args, **kwargs: {"returncode": 0},
    )
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)
    assert utils.run_agent_on_vm(Controller(b"0"), "command", None) is True


@pytest.mark.parametrize("phase,age,expected", [("model", 500, True), ("model", 1900, False), ("tool", 500, False)])
def test_host_respects_runner_model_phase_deadline(phase, age, expected, monkeypatch):
    monkeypatch.setattr(utils, "_launch_on_vm", lambda *args, **kwargs: True)
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)
    monkeypatch.setattr(utils, "_execute_shell_on_vm", lambda *args, **kwargs: {"returncode": 0})
    monkeypatch.setattr(utils, "_probe_agent_liveness", lambda *args: {
        "exists": True, "runner_alive": True, "last_event": "message_end",
        "age_s": age, "phase": phase, "phase_elapsed_s": age,
    })
    markers = iter([None, b"0"])
    controller = Controller(None)
    controller.get_file = lambda _: next(markers)
    command = utils.AgentCommand("run", activity_path="trajectory", idle_timeout=420, model_request_timeout=1860)
    assert utils.run_agent_on_vm(controller, command, None) is expected


def test_agent_driver_recovers_terminal_trajectory_without_exit_marker(monkeypatch):
    writes = []
    monkeypatch.setattr(utils, "_launch_on_vm", lambda *args, **kwargs: True)
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        utils,
        "_probe_agent_liveness",
        lambda *args: {
            "exists": True,
            "age_s": 1,
            "last_event": "agent_end",
            "runner_alive": False,
        },
    )
    monkeypatch.setattr(
        utils,
        "_execute_shell_on_vm",
        lambda controller, command, timeout: writes.append(command)
        or {"returncode": 0},
    )
    command = utils.AgentCommand(
        "prime-agent",
        activity_path="/tmp/claude_task/trajectory.jsonl",
        idle_timeout=420,
        process_pattern="[p]rime_step_runner.py",
        successful_terminal_events=("agent_end", "prime_step_limit"),
    )

    assert utils.run_agent_on_vm(Controller(None), command, None) is True
    assert any("exit_code.txt" in item and "printf '0" in item for item in writes)


@pytest.mark.parametrize("runner_alive", [True, None])
def test_agent_driver_waits_for_real_exit_after_compaction_agent_end(monkeypatch, runner_alive):
    writes = []
    markers = iter([None, b"1"])
    monkeypatch.setattr(utils, "_launch_on_vm", lambda *args, **kwargs: True)
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)
    monkeypatch.setattr(utils, "_probe_agent_liveness", lambda *args: {
        "exists": True, "age_s": 1, "last_event": "agent_end",
        "runner_alive": runner_alive,
    })
    monkeypatch.setattr(utils, "_execute_shell_on_vm",
                        lambda controller, command, timeout:
                        writes.append(command) or {"returncode": 0})
    controller = Controller(None)
    controller.get_file = lambda path: next(markers)
    command = utils.AgentCommand(
        "prime-agent", activity_path="/tmp/claude_task/trajectory.jsonl",
        successful_terminal_events=("agent_end",),
    )
    assert utils.run_agent_on_vm(controller, command, None) is False
    assert not any("printf '0" in item for item in writes)


def test_agent_driver_recovers_terminal_trajectory_with_invalid_exit_marker(monkeypatch):
    writes = []
    monkeypatch.setattr(utils, "_launch_on_vm", lambda *args, **kwargs: True)
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        utils,
        "_probe_agent_liveness",
        lambda *args: {
            "exists": True,
            "age_s": 1,
            "last_event": "prime_step_limit",
            "runner_alive": False,
        },
    )
    monkeypatch.setattr(
        utils,
        "_execute_shell_on_vm",
        lambda controller, command, timeout: writes.append(command)
        or {"returncode": 0},
    )
    command = utils.AgentCommand(
        "prime-agent",
        activity_path="/tmp/claude_task/trajectory.jsonl",
        idle_timeout=420,
        process_pattern="[p]rime_step_runner.py",
        successful_terminal_events=("agent_end", "prime_step_limit"),
    )

    assert utils.run_agent_on_vm(Controller(b'""0"\n"'), command, None) is True
    assert any("exit_code.txt" in item and "printf '0" in item for item in writes)


def test_agent_driver_rejects_stale_trajectory_when_marker_is_missing(monkeypatch):
    cleanup = []
    monkeypatch.setattr(utils, "_launch_on_vm", lambda *args, **kwargs: True)
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        utils,
        "_probe_agent_liveness",
        lambda *args: {
            "exists": True,
            "age_s": 421,
            "last_event": "tool_execution_end",
            "runner_alive": False,
        },
    )
    monkeypatch.setattr(
        utils,
        "_execute_shell_on_vm",
        lambda controller, command, timeout: cleanup.append(command)
        or {"returncode": 0},
    )
    command = utils.AgentCommand(
        "prime-agent",
        activity_path="/tmp/claude_task/trajectory.jsonl",
        idle_timeout=420,
        process_pattern="[p]rime_step_runner.py",
    )

    assert utils.run_agent_on_vm(Controller(None), command, None) is False
    assert "[p]rime_step_runner.py" in cleanup[-1]


def test_agent_driver_rejects_missing_trajectory_after_runner_disappears(monkeypatch):
    monkeypatch.setattr(utils, "_launch_on_vm", lambda *args, **kwargs: True)
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        utils,
        "_probe_agent_liveness",
        lambda *args: {
            "exists": False,
            "age_s": None,
            "last_event": None,
            "runner_alive": False,
        },
    )
    monkeypatch.setattr(
        utils,
        "_execute_shell_on_vm",
        lambda *args, **kwargs: {"returncode": 0},
    )
    command = utils.AgentCommand(
        "prime-agent",
        activity_path="/tmp/claude_task/trajectory.jsonl",
        idle_timeout=10,
        process_pattern="[p]rime_step_runner.py",
    )

    assert utils.run_agent_on_vm(Controller(None), command, None) is False


def test_agent_driver_salvages_long_episode_after_runner_grace(monkeypatch):
    salvaged = []
    monkeypatch.setattr(utils, "_launch_on_vm", lambda *args, **kwargs: True)
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        utils,
        "_probe_agent_liveness",
        lambda *args: {
            "exists": True,
            "age_s": 5,
            "last_event": "turn_start",
            "runner_alive": False,
        },
    )
    monkeypatch.setattr(
        utils, "_count_completed_assistant_steps", lambda *args: 176,
    )
    monkeypatch.setattr(
        utils,
        "_mark_agent_episode_salvaged",
        lambda controller, command, steps: salvaged.append(steps) or True,
    )
    monkeypatch.setattr(
        utils,
        "_execute_shell_on_vm",
        lambda *args, **kwargs: {"returncode": 0},
    )
    command = utils.AgentCommand(
        "prime-agent",
        activity_path="/tmp/claude_task/trajectory.jsonl",
        idle_timeout=420,
        process_pattern="[p]rime_step_runner.py",
        runner_exit_grace=60,
        salvage_min_steps=50,
        salvage_terminal_event="prime_agent_salvaged",
    )

    assert utils.run_agent_on_vm(Controller(None), command, None) is True
    assert salvaged == [176]


def test_agent_driver_retries_short_episode_after_runner_grace(monkeypatch):
    cleanup = []
    monkeypatch.setattr(utils, "_launch_on_vm", lambda *args, **kwargs: True)
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        utils,
        "_probe_agent_liveness",
        lambda *args: {
            "exists": True,
            "age_s": 5,
            "last_event": "turn_start",
            "runner_alive": False,
        },
    )
    monkeypatch.setattr(
        utils, "_count_completed_assistant_steps", lambda *args: 12,
    )
    monkeypatch.setattr(
        utils,
        "_execute_shell_on_vm",
        lambda controller, command, timeout: cleanup.append(command)
        or {"returncode": 0},
    )
    command = utils.AgentCommand(
        "prime-agent",
        activity_path="/tmp/claude_task/trajectory.jsonl",
        idle_timeout=420,
        process_pattern="[p]rime_step_runner.py",
        runner_exit_grace=60,
        salvage_min_steps=50,
    )

    assert utils.run_agent_on_vm(Controller(None), command, None) is False
    assert "[p]rime_step_runner.py" in cleanup[-1]


def test_agent_driver_refuses_to_launch_if_stale_marker_cannot_be_cleared(monkeypatch):
    launched = []
    monkeypatch.setattr(
        utils, "_execute_shell_on_vm",
        lambda *args, **kwargs: {"returncode": 1},
    )
    monkeypatch.setattr(
        utils, "_launch_on_vm", lambda *args, **kwargs: launched.append(True),
    )
    assert utils.run_agent_on_vm(Controller(b"0"), "command", 1) is False
    assert launched == []


def test_launch_response_does_not_log_echoed_command(monkeypatch, caplog):
    class Response:
        status_code = 200
        text = "export ANTHROPIC_API_KEY=super-secret"

    monkeypatch.setattr(utils.requests, "post", lambda *args, **kwargs: Response())
    controller = type("Controller", (), {"http_server": "http://vm"})()
    with caplog.at_level(logging.INFO, logger="agent_s4.vm.utils"):
        assert utils._launch_on_vm(controller, "secret command") is True
    assert "super-secret" not in caplog.text
    assert "VM command launched successfully" in caplog.text


def test_launch_error_does_not_log_echoed_command(monkeypatch, caplog):
    class Response:
        status_code = 500
        text = "export ANTHROPIC_API_KEY=super-secret"

    monkeypatch.setattr(utils.requests, "post", lambda *args, **kwargs: Response())
    controller = type("Controller", (), {"http_server": "http://vm"})()
    with caplog.at_level(logging.ERROR, logger="agent_s4.vm.utils"):
        assert utils._launch_on_vm(controller, "secret command") is False
    assert "super-secret" not in caplog.text
    assert "VM command launch failed (status 500)" in caplog.text
