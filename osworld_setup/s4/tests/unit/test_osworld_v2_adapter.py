from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from agent_s4.eval import osworld, osworld_v2


def _args(tmp_path, **overrides):
    values = dict(
        screen_width=1920, screen_height=1080, provider_name="docker",
        region="us-east-1", snapshot_name="init_state", enable_proxy=None,
        volume_size=None, cache_dir=None, result_dir=str(tmp_path),
        path_to_vm=None, headless=True, client_password="pw",
        api_key=None, openrouter_api_key=None, parasail_api_key=None,
        timeout=1, model="fake", or_provider=None, max_user_turns=3,
        no_tool=True, prompt_addendum="", action_space="shadow_eval",
        observation_type="screenshot", benchmark="osworld_v2",
        test_config_base_dir=str(tmp_path),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_persist_float_and_dict_results(tmp_path):
    score, details = osworld.persist_evaluation_result(0.25, str(tmp_path))
    assert score == 0.25 and details is None
    assert (tmp_path / "result.txt").read_text() == "0.25\n"

    payload = {"score": 0.75, "partial_scores": {"a": 0.5}, "safety": {"ok": True}}
    score, details = osworld.persist_evaluation_result(payload, str(tmp_path))
    assert score == 0.75 and details == payload
    assert (tmp_path / "result.txt").read_text() == "0.75\n"
    assert json.loads((tmp_path / "result.json").read_text()) == payload


def test_persist_normalizes_only_full_credit_float_dust(tmp_path):
    payload = {"score": 0.9999999999999999, "partial_scores": {"a": 1.0}}
    score, details = osworld.persist_evaluation_result(payload, str(tmp_path))
    assert score == 1.0
    assert details == {
        "score": 1.0,
        "partial_scores": {"a": 1.0},
        "evaluator_raw_score": 0.9999999999999999,
    }
    assert (tmp_path / "result.txt").read_text() == "1.0\n"
    assert json.loads((tmp_path / "result.json").read_text()) == details

    for partial in (0.9979, 0.9969, 0.993, 0.9900000000000001, 0.99):
        score, details = osworld.persist_evaluation_result(partial, str(tmp_path))
        assert score == partial
        assert details is None
        assert float((tmp_path / "result.txt").read_text()) == partial


@pytest.mark.parametrize("payload", [
    {}, {"score": "not-a-number"}, {"score": -0.01}, {"score": 1.0001},
    {"score": float("nan")}, {"score": float("inf")},
])
def test_persist_rejects_missing_invalid_or_out_of_range_scores(tmp_path, payload):
    with pytest.raises(ValueError):
        osworld.persist_evaluation_result(payload, str(tmp_path))


def test_env_spec_preserves_v2_task_resources_and_proxy_default(tmp_path):
    class Task(dict):
        platform = "windows"
        image = "custom-image"
        instance_type = "t3.2xlarge"
        volume_size = 100
        disable_vnc = True
        disable_recording = True

    example = Task(id="x")
    spec = osworld_v2.build_env_spec(example, _args(tmp_path))
    assert spec.os_type == "Windows"
    assert spec.snapshot_name == "init_state"  # non-AWS keeps CLI snapshot
    assert spec.instance_type == "t3.2xlarge"
    assert spec.volume_size == 100
    assert spec.enable_proxy is True
    assert spec.force_disable_vnc is True
    assert spec.force_disable_recording is True
    assert spec.cache_dir.endswith("_task_cache/x")
    kwargs = osworld_v2.build_env_kwargs(spec, _args(tmp_path))
    assert kwargs["force_disable_vnc"] is True
    assert kwargs["force_disable_recording"] is True


def test_env_spec_allows_runtime_instance_type_override(tmp_path):
    class Task(dict):
        instance_type = "t3.large"

    spec = osworld_v2.build_env_spec(
        Task(id="048"),
        _args(tmp_path, instance_type_override="t3.xlarge"),
    )
    assert spec.instance_type == "t3.xlarge"


def test_env_spec_defaults_root_volume_when_nothing_declares_one(tmp_path):
    """90 of the 108 V2 tasks declare no volume_size, and the V2 AMI's stock
    29GB root is already ~26GB full, so falling through to the provider default
    fills the disk and corrupts the agent's exit-code marker."""
    spec = osworld_v2.build_env_spec({}, _args(tmp_path))
    assert spec.volume_size == osworld_v2.DEFAULT_VOLUME_SIZE_GB
    assert osworld_v2.build_env_kwargs(spec, _args(tmp_path))["volume_size"] == (
        osworld_v2.DEFAULT_VOLUME_SIZE_GB
    )


def test_env_spec_prefers_cli_volume_over_the_default(tmp_path):
    spec = osworld_v2.build_env_spec({}, _args(tmp_path, volume_size=120))
    assert spec.volume_size == 120


def test_env_spec_lets_a_task_keep_a_volume_smaller_than_the_default(tmp_path):
    """Upstream sized the task, so its declaration wins over both the CLI and
    the harness default -- but a value this small gets a warning."""
    class Task(dict):
        volume_size = 32

    # Patch the logger rather than use caplog: other suites reconfigure root
    # logging, which makes a caplog assertion pass alone and fail in a full run.
    with mock.patch.object(osworld_v2.logger, "warning") as warn:
        spec = osworld_v2.build_env_spec(Task(id="x"), _args(tmp_path, volume_size=120))
    assert spec.volume_size == 32
    assert warn.call_count == 1
    assert "run out of disk" in warn.call_args[0][0]


def test_env_spec_does_not_warn_for_a_comfortable_volume(tmp_path):
    with mock.patch.object(osworld_v2.logger, "warning") as warn:
        osworld_v2.build_env_spec({}, _args(tmp_path))
    warn.assert_not_called()


def test_v1_env_spec_does_not_get_the_v2_volume_default(tmp_path):
    """The default is a V2-AMI workaround; v1 keeps deferring to the provider."""
    assert osworld.build_env_spec({}, _args(tmp_path)).volume_size is None


def test_env_spec_explicit_proxy_disable_overrides_v2_default(tmp_path):
    spec = osworld_v2.build_env_spec({}, _args(tmp_path, enable_proxy=False))
    assert spec.enable_proxy is False


def test_hitl_instruction_drops_no_question_rule_and_adds_task_date():
    class TaskWithClassDate(dict):
        task_current_date = "2026-03-01"

    instruction = osworld_v2._instruction_for_task(
        TaskWithClassDate(user_simulator={"type": "fixed"}), "Do it",
    )
    assert "Do not ask user" not in instruction
    assert "2026-03-01" in instruction


def test_extract_ask_user_from_stream_json():
    payload = json.dumps({
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": "ASK_USER: Which account?"}]},
    })
    assert osworld_v2._extract_ask_user(payload) == "Which account?"


def test_eval_constraints_preserve_official_task_package_managers(tmp_path):
    command = osworld._build_eval_constraint_command("p'ass word")
    assert "CLAUDE.md" in command
    assert "pip uninstall" not in command
    assert "/usr/local/bin/pip" not in command
    assert "shadow-disabled" not in command
    assert "deny-package-manager" not in command

    agent_command = osworld._build_agent_command(
        _args(tmp_path, no_tool=False), _model_cfg(), "do it", "prompt",
        api_key=None, openrouter_key=None, parasail_key=None,
    )
    assert agent_command == "do it"


def test_shutdown_workers_allows_close_before_escalating():
    class Process:
        def __init__(self, alive_after_join):
            self.alive_after_join = alive_after_join
            self.actions = []

        def join(self, timeout=None):
            self.actions.append(("join", timeout))

        def is_alive(self):
            return self.alive_after_join

        def terminate(self):
            self.actions.append(("terminate", None))
            self.alive_after_join = False

        def kill(self):
            self.actions.append(("kill", None))
            self.alive_after_join = False

    graceful = Process(False)
    escalated = Process(True)
    osworld._shutdown_workers(
        [graceful, escalated], graceful_timeout=0, terminate_timeout=0,
    )
    assert not any(action == "terminate" for action, _ in graceful.actions)
    assert any(action == "terminate" for action, _ in escalated.actions)
    assert not any(action == "kill" for action, _ in escalated.actions)


def test_shutdown_workers_kills_process_that_ignores_term():
    class StubbornProcess:
        def __init__(self):
            self.alive = True
            self.actions = []

        def join(self, timeout=None):
            self.actions.append("join")

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.actions.append("terminate")

        def kill(self):
            self.actions.append("kill")
            self.alive = False

    process = StubbornProcess()
    osworld._shutdown_workers(
        [process], graceful_timeout=0, terminate_timeout=0,
    )
    assert process.actions == ["join", "terminate", "join", "kill", "join"]


class _Controller:
    def __init__(self, trajectories=None):
        self.trajectories = list(trajectories or [b"{}\n"])
        self.requested_paths = []

    def get_file(self, path):
        self.requested_paths.append(path)
        if path.endswith("trajectory.jsonl"):
            if len(self.trajectories) > 1:
                return self.trajectories.pop(0)
            return self.trajectories[0]
        if path.endswith("exit_code.txt"):
            return b"0"
        if path.endswith("stderr.log"):
            return b""
        return None

    def start_recording(self):
        pass

    def end_recording(self, path):
        pass


class _Env:
    def __init__(self, controller=None):
        self.controller = controller or _Controller()
        self.setup_controller = object()
        self.client_password = "pw"
        self.enable_proxy = True
        self.user_simulator = None
        self.action_history = []
        self.is_environment_used = False

    def reset(self, task_config):
        self.user_simulator = None
        setup = getattr(task_config, "setup", None)
        if callable(setup):
            setup(self.setup_controller, use_proxy=False)

    def close(self):
        pass


def _build_command(**kwargs):
    return kwargs["instruction"]


def _model_cfg():
    return {
        "binary": "fake", "model_flag": "fake",
        "installer": lambda controller, password: True,
        "build_command": _build_command,
    }


class _MultiTask(dict):
    def __init__(self, events):
        super().__init__(id="069", instruction="compat", disable_recording=True)
        self.events = events

    def setup(self, controller, use_proxy=False):
        self.events.append(("setup", 1, use_proxy))

    def get_phases(self):
        phases = []
        for index, score in enumerate((0.35, 0.30, 0.10, 0.25), start=1):
            def setup(controller, use_proxy=False, i=index):
                self.events.append(("setup", i, use_proxy))

            def evaluate(env, i=index, value=score):
                self.events.append(("evaluate", i, env))
                return value

            phases.append({
                "name": f"p{index}", "setup": setup,
                "instruction": f"phase {index}", "evaluate": evaluate,
                "gate_min_score": score,
            })
        return phases


def test_multiphase_runs_all_phases_on_same_env(tmp_path, monkeypatch):
    events = []
    env = _Env()
    monkeypatch.setattr(osworld.time, "sleep", lambda _: None)
    monkeypatch.setattr(osworld, "run_agent_on_vm", lambda *args: True)
    result = osworld_v2.run_single_task(
        env, _MultiTask(events), "compat", str(tmp_path),
        _args(tmp_path), _model_cfg(), str(tmp_path),
    )
    assert result["score"] == 1.0
    assert [e[:2] for e in events if e[0] == "evaluate"] == [
        ("evaluate", 1), ("evaluate", 2), ("evaluate", 3), ("evaluate", 4),
    ]
    assert all(e[2] is env for e in events if e[0] == "evaluate")
    # reset performs phase 1 setup; the harness injects phases 2-4.
    assert [e[:2] for e in events if e[0] == "setup"] == [
        ("setup", 1), ("setup", 2), ("setup", 3), ("setup", 4),
    ]
    assert len(json.loads((tmp_path / "phase_results.json").read_text())) == 4


def test_multiphase_gate_stops_later_phases_and_preserves_dict_details(
        tmp_path, monkeypatch):
    events = []

    class GatedTask(dict):
        def __init__(self):
            super().__init__(id="069", instruction="phase", disable_recording=True)

        def get_phases(self):
            return [
                {
                    "name": "gate", "instruction": "phase 1",
                    "setup": lambda *args, **kwargs: None,
                    "evaluate": lambda env: {
                        "score": 0.2, "partial_scores": {"gate": 0.2},
                    },
                    "gate_min_score": 0.5,
                },
                {
                    "name": "must-not-run", "instruction": "phase 2",
                    "setup": lambda *args, **kwargs: events.append("setup-2"),
                    "evaluate": lambda env: events.append("evaluate-2") or 1.0,
                },
            ]

    monkeypatch.setattr(osworld.time, "sleep", lambda _: None)
    monkeypatch.setattr(osworld, "run_agent_on_vm", lambda *args: True)
    result = osworld_v2.run_single_task(
        _Env(), GatedTask(), "phase", str(tmp_path), _args(tmp_path),
        _model_cfg(), str(tmp_path),
    )
    assert result["score"] == 0.2
    assert events == []
    phase_results = json.loads((tmp_path / "phase_results.json").read_text())
    assert phase_results[0]["details"]["partial_scores"] == {"gate": 0.2}


class _Simulator:
    def __init__(self):
        self.questions = []

    def respond(self, question):
        self.questions.append(question)
        return "Use the savings account"


def test_hitl_sentinel_calls_host_simulator_then_continues(tmp_path, monkeypatch):
    ask = json.dumps({"text": "ASK_USER: Which account should I use?"}).encode() + b"\n"
    done = json.dumps({"text": "done"}).encode() + b"\n"
    env = _Env(_Controller([ask, done]))
    env.user_simulator = _Simulator()
    commands = []
    monkeypatch.setattr(osworld, "run_agent_on_vm", lambda controller, command, timeout: commands.append(command) or True)
    history = []
    args = _args(tmp_path)
    osworld._run_agent_episode(
        env, args, _model_cfg(), "do it",
        osworld.TOOL_SYSTEM_PROMPT + osworld_v2.HITL_SYSTEM_PROMPT_SUFFIX,
        str(tmp_path),
        ask_user=osworld_v2._make_ask_user_bridge(env, args, str(tmp_path), history),
    )
    assert env.user_simulator.questions == ["Which account should I use?"]
    assert "Use the savings account" in commands[1]
    assert history[0]["answer"] == "Use the savings account"


def test_step_limit_is_agent_outcome_not_retryable_infra_failure(tmp_path, monkeypatch):
    trajectory = (
        b'{"type":"message_end","message":{"role":"assistant"}}\n'
        b'{"type":"prime_step_limit","limit":300,"steps":300}\n'
    )
    env = _Env(_Controller([trajectory]))
    monkeypatch.setattr(osworld, "run_agent_on_vm", lambda *args: False)

    # A broken/missing exit marker must not turn a completed 300-step episode
    # into agent_execution_failure and therefore an infrastructure retry.
    osworld._run_agent_episode(
        env, _args(tmp_path), _model_cfg(), "do it", "prompt", str(tmp_path),
    )
    assert (tmp_path / "claude_trajectory.jsonl").read_bytes() == trajectory


def test_prime_episode_compacts_stream_updates_and_skips_duplicate_stdout(
        tmp_path, monkeypatch):
    trajectory = (
        b'{"type":"agent_start"}\n'
        b'{"type":"message_update","message":{"text":"partial"}}\n'
        b'{"type":"message_end","message":{"text":"complete"}}\n'
        b'{"type":"agent_end"}\n'
    )
    controller = _Controller([trajectory])
    env = _Env(controller)
    model_cfg = {**_model_cfg(), "binary": "prime-agent"}
    monkeypatch.setattr(osworld, "run_agent_on_vm", lambda *args: True)

    osworld._run_agent_episode(
        env, _args(tmp_path), model_cfg, "do it", "prompt", str(tmp_path),
    )

    assert (tmp_path / "claude_trajectory.jsonl").read_bytes() == (
        b'{"type":"agent_start"}\n'
        b'{"type":"message_end","message":{"text":"complete"}}\n'
        b'{"type":"agent_end"}\n'
    )
    assert not (tmp_path / "mini_stdout.log").exists()
    assert not any(path.endswith("stdout.log") for path in controller.requested_paths)


def test_hitl_simulator_exception_is_classified(tmp_path, monkeypatch):
    ask = json.dumps({"text": "ASK_USER: Which account?"}).encode() + b"\n"
    env = _Env(_Controller([ask]))

    class BrokenSimulator:
        def respond(self, question):
            raise RuntimeError("provider unavailable")

    env.user_simulator = BrokenSimulator()
    monkeypatch.setattr(osworld, "run_agent_on_vm", lambda *args: True)
    args = _args(tmp_path)
    with pytest.raises(osworld.HarnessFailure) as exc:
        osworld._run_agent_episode(
            env, args, _model_cfg(), "do it", "prompt", str(tmp_path),
            ask_user=osworld_v2._make_ask_user_bridge(env, args, str(tmp_path), []),
        )
    assert exc.value.failure_type == "user_simulator_failure"


def test_run_single_task_removes_non_hitl_prompt_rule_for_simulator(
        tmp_path, monkeypatch):
    class HitlEnv(_Env):
        def reset(self, task_config):
            self.user_simulator = _Simulator()

        def evaluate(self):
            return 1.0

    captured_prompts = []

    def build_command(**kwargs):
        captured_prompts.append(kwargs["system_prompt"])
        return "agent"

    model_cfg = {
        "binary": "fake", "model_flag": "fake",
        "installer": lambda controller, password: True,
        "build_command": build_command,
    }
    monkeypatch.setattr(osworld.time, "sleep", lambda _: None)
    monkeypatch.setattr(osworld, "upload_shadow_interface", lambda *args: True)
    monkeypatch.setattr(osworld, "run_setup_script", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        osworld, "_execute_shell_on_vm",
        lambda *args, **kwargs: {"returncode": 0},
    )
    monkeypatch.setattr(osworld, "run_agent_on_vm", lambda *args: True)
    result = osworld_v2.run_single_task(
        HitlEnv(), {"id": "hitl", "instruction": "do it"}, "do it",
        str(tmp_path), _args(tmp_path, no_tool=False), model_cfg, str(tmp_path),
    )
    assert result["score"] == 1.0
    assert "NEVER ask the user" not in captured_prompts[0]
    assert "ASK_USER:" in captured_prompts[0]


def test_run_single_task_classifies_setup_script_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(osworld.time, "sleep", lambda _: None)
    monkeypatch.setattr(osworld, "upload_shadow_interface", lambda *args: True)
    monkeypatch.setattr(osworld, "run_setup_script", lambda *args, **kwargs: False)
    result = osworld_v2.run_single_task(
        _Env(), {"id": "x", "instruction": "do it"}, "do it",
        str(tmp_path), _args(tmp_path, no_tool=False), _model_cfg(), str(tmp_path),
    )
    assert result["failure_type"] == "task_setup_failure"
    assert result["valid_for_fitness"] is False


def test_prime_runtime_is_installed_before_task_can_revoke_sudo(tmp_path, monkeypatch):
    events = []

    class LockdownEnv(_Env):
        def reset(self, task_config):
            if task_config is None:
                events.append("clean-reset")
                return
            events.append("task-setup-lockdown")

    def installer(controller, password):
        events.append("install-or-probe")
        return True

    model_cfg = {
        **_model_cfg(),
        "binary": "prime-agent",
        "installer": installer,
        "install_before_task_setup": True,
    }
    monkeypatch.setattr(osworld.time, "sleep", lambda _: None)
    result = osworld.prepare_task_vm(
        LockdownEnv(), {"id": "048", "disable_recording": True},
        _args(tmp_path), model_cfg, str(tmp_path),
    )

    assert not isinstance(result, dict)
    assert events == [
        "clean-reset",
        "install-or-probe",
        "task-setup-lockdown",
        "install-or-probe",
    ]


def test_task_setup_failure_keeps_nested_broken_pipe_cause(tmp_path, monkeypatch):
    class BrokenSetupEnv(_Env):
        def reset(self, task_config):
            try:
                raise BrokenPipeError(32, "Broken pipe")
            except BrokenPipeError as exc:
                raise RuntimeError("custom setup failed") from exc

    monkeypatch.setattr(osworld.time, "sleep", lambda _: None)
    result = osworld.prepare_task_vm(
        BrokenSetupEnv(), {"id": "064"}, _args(tmp_path),
        _model_cfg(), str(tmp_path),
    )

    assert result["failure_type"] == "task_setup_failure"
    assert "RuntimeError: custom setup failed" in result["error"]
    assert "BrokenPipeError: [Errno 32] Broken pipe" in result["error"]


def test_summary_uses_one_canonical_threshold():
    summary = osworld.print_results_summary([
        {"task_id": "a", "score": 0.49},
        {"task_id": "b", "score": 0.50},
        {"task_id": "c", "score": 1.00},
    ])
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    assert summary["mean_score"] == pytest.approx(1.99 / 3)


def test_v2_summary_uses_strict_normalized_full_credit():
    summary = osworld.print_results_summary([
        {"task_id": "dust", "score": 0.9999999999999999},
        {"task_id": "partial", "score": 0.9979},
        {"task_id": "half", "score": 0.50},
        {"task_id": "exact", "score": 1.00},
    ], benchmark="osworld_v2")
    assert summary["passed"] == 2
    assert summary["failed"] == 2
    assert summary["mean_score"] == pytest.approx(
        (0.9999999999999999 + 0.9979 + 0.5 + 1.0) / 4
    )


def test_worker_uses_instruction_from_same_dynamic_task_instance(tmp_path, monkeypatch):
    class Queue:
        def __init__(self):
            self.done = False

        def get(self, timeout):
            if self.done:
                raise RuntimeError("empty")
            self.done = True
            return ("026", str(tmp_path / "task_026.py"))

    captured = []
    worker_example = {
        "id": "026", "instruction": "worker-generated-password=B",
        "platform": "linux",
    }
    monkeypatch.setattr(osworld_v2, "load_task", lambda *args: worker_example)
    monkeypatch.setattr(osworld_v2, "DesktopEnv", lambda **kwargs: _Env())
    monkeypatch.setattr(
        osworld_v2, "run_single_task",
        lambda env, example, instruction, *args: captured.append(instruction) or {
            "task_id": "026", "score": 1.0, "valid_for_fitness": True,
        },
    )
    results = []
    osworld_v2.worker(Queue(), results, _args(tmp_path), str(tmp_path), _model_cfg())
    assert captured == ["worker-generated-password=B"]
    assert results[0]["score"] == 1.0


def test_worker_preserves_typed_harness_failure(tmp_path, monkeypatch):
    class Queue:
        done = False

        def get(self, timeout):
            if self.done:
                raise RuntimeError("empty")
            self.done = True
            return ("hitl", str(tmp_path / "task_hitl.py"))

    monkeypatch.setattr(osworld_v2, "load_task", lambda *args: {
        "id": "hitl", "instruction": "ask", "platform": "linux",
    })
    monkeypatch.setattr(osworld_v2, "DesktopEnv", lambda **kwargs: _Env())

    def fail(*args, **kwargs):
        raise osworld.HarnessFailure("simulator down", "user_simulator_failure")

    monkeypatch.setattr(osworld_v2, "run_single_task", fail)
    results = []
    osworld_v2.worker(Queue(), results, _args(tmp_path), str(tmp_path), _model_cfg())
    assert results[0]["failure_type"] == "user_simulator_failure"
    assert results[0]["valid_for_fitness"] is True
    assert results[0]["imputed_zero"] is True
    assert results[0]["score"] == 0.0


def test_worker_retries_infra_failure_on_fresh_vm(tmp_path, monkeypatch):
    class Queue:
        def __init__(self):
            self.items = [("048", str(tmp_path / "task_048.py"), 0)]

        def get(self, timeout):
            if not self.items:
                raise RuntimeError("empty")
            return self.items.pop(0)

        def put(self, item):
            self.items.append(item)

    monkeypatch.setattr(osworld_v2, "load_task", lambda *args: {
        "id": "048", "instruction": "edit the file", "platform": "linux",
    })
    environments = []

    def make_env(**kwargs):
        env = _Env()
        env.closed = False
        env.close = lambda: setattr(env, "closed", True)
        environments.append(env)
        return env

    monkeypatch.setattr(osworld_v2, "DesktopEnv", make_env)
    attempts = []

    def run_task(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            return {
                "task_id": "048", "score": 0.0,
                "error": "agent install failed (prime-agent)",
                "failure_type": "agent_install_failure",
                "valid_for_fitness": False,
            }
        return {"task_id": "048", "score": 1.0, "valid_for_fitness": True}

    monkeypatch.setattr(osworld_v2, "run_single_task", run_task)
    results = []
    osworld_v2.worker(
        Queue(), results, _args(tmp_path, infra_retries=2),
        str(tmp_path), _model_cfg(),
    )

    assert len(attempts) == 2
    assert len(environments) == 2
    assert environments[0].closed is True
    assert results == [{
        "task_id": "048", "score": 1.0, "valid_for_fitness": True,
    }]
    result_dir = tmp_path / "shadow_eval" / "screenshot" / "fake" / "048"
    assert not (result_dir / "failure.json").exists()


def test_step_limited_trajectory_is_never_infra_requeued(tmp_path):
    result_dir = tmp_path / "048"
    result_dir.mkdir()
    (result_dir / "claude_trajectory.jsonl").write_bytes(
        b'{"type":"message_end","message":{"role":"assistant"}}\n'
        b'{"type":"prime_step_limit","limit":300,"steps":300}\n'
    )

    class Queue:
        items = []

        def put(self, item):
            self.items.append(item)

    results = []
    requeued = osworld.retry_or_record_infra_failure(
        Queue(), results, ("048", "task.py", 0),
        {
            "task_id": "048", "score": 0.0,
            "failure_type": "agent_execution_failure",
            "valid_for_fitness": False,
        },
        _args(tmp_path, infra_retries=2), str(result_dir),
        stage="agent", proc_name="test",
    )
    assert requeued is False
    assert Queue.items == []
    assert results[0]["failure_type"] == "agent_step_limit"
    assert results[0]["retryable"] is False
    assert results[0]["imputed_zero"] is False
    assert float((result_dir / "result.txt").read_text()) == 0.0


def test_worker_records_only_final_failure_after_retries(tmp_path, monkeypatch):
    class Queue:
        def __init__(self):
            self.items = [("069", str(tmp_path / "task_069.py"), 0)]

        def get(self, timeout):
            if not self.items:
                raise RuntimeError("empty")
            return self.items.pop(0)

        def put(self, item):
            self.items.append(item)

    monkeypatch.setattr(osworld_v2, "load_task", lambda *args: {
        "id": "069", "instruction": "complete both phases", "platform": "linux",
    })
    monkeypatch.setattr(osworld_v2, "DesktopEnv", lambda **kwargs: _Env())
    monkeypatch.setattr(osworld_v2, "run_single_task", lambda *args, **kwargs: {
        "task_id": "069", "score": 0.0,
        "error": "prime-agent daemon closed",
        "failure_type": "agent_execution_failure",
        "valid_for_fitness": False,
    })
    results = []
    osworld_v2.worker(
        Queue(), results, _args(tmp_path, infra_retries=2),
        str(tmp_path), _model_cfg(),
    )

    assert len(results) == 1
    assert results[0]["attempt"] == 3
    assert results[0]["max_attempts"] == 3
    assert results[0]["valid_for_fitness"] is True
    assert results[0]["imputed_zero"] is True
    result_dir = tmp_path / "shadow_eval" / "screenshot" / "fake" / "069"
    assert float((result_dir / "result.txt").read_text()) == 0.0
    details = json.loads((result_dir / "result.json").read_text())
    assert details["failure_type"] == "agent_execution_failure"
    assert details["imputed_zero"] is True
    manifest = json.loads((result_dir / "failure.json").read_text())
    assert manifest["attempt"] == 3
    assert manifest["status"] == "infra_failure"
    assert manifest["valid_for_fitness"] is True
    assert manifest["imputed_zero"] is True


class _Clock:
    """Monotonic clock the test advances by hand.

    The budget is a wall clock, so exercising it against the real one would
    make the assertions depend on how fast the suite runs.
    """

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _recording_model_cfg(seen):
    """A model config whose build_command records the in-VM episode timeout."""
    def build_command(**kwargs):
        seen.append(kwargs["timeout"])
        return kwargs["instruction"]

    return {**_model_cfg(), "build_command": build_command}


def test_task_budget_is_shared_across_phases_not_granted_per_phase(
        tmp_path, monkeypatch):
    # A 4-phase task under a 600s cap must get 600s in total, not 600s each.
    clock = _Clock()
    in_vm_timeouts = []
    driver_timeouts = []
    events = []
    monkeypatch.setattr(osworld.time, "monotonic", clock)
    monkeypatch.setattr(osworld.time, "sleep", lambda _: None)

    def fake_run(controller, command, timeout):
        driver_timeouts.append(timeout)
        clock.advance(250)
        return True

    monkeypatch.setattr(osworld, "run_agent_on_vm", fake_run)
    result = osworld_v2.run_single_task(
        _Env(), _MultiTask(events), "compat", str(tmp_path),
        _args(tmp_path, timeout=600), _recording_model_cfg(in_vm_timeouts),
        str(tmp_path),
    )

    # Each phase is launched with what its predecessors left, and phase 4
    # finds the budget spent. Four phases under a per-phase cap would have
    # been entitled to 2400s.
    assert driver_timeouts == [600, 350, 100]
    assert in_vm_timeouts == [600, 350, 100]
    # Every phase is still set up, evaluated and scored: a phase whose agent
    # never ran produced a real desktop state, and the run must reflect it.
    assert [e[:2] for e in events if e[0] == "evaluate"] == [
        ("evaluate", 1), ("evaluate", 2), ("evaluate", 3), ("evaluate", 4),
    ]
    assert result["score"] == 1.0
    assert len(json.loads((tmp_path / "phase_results.json").read_text())) == 4


def test_ask_user_continuation_draws_from_the_same_budget(tmp_path, monkeypatch):
    # A follow-up turn is another episode on the same task, so it spends the
    # task's remaining time rather than being handed a fresh cap.
    ask = json.dumps({"text": "ASK_USER: Which account should I use?"}).encode() + b"\n"
    done = json.dumps({"text": "done"}).encode() + b"\n"
    clock = _Clock()
    driver_timeouts = []
    monkeypatch.setattr(osworld.time, "monotonic", clock)

    def fake_run(controller, command, timeout):
        driver_timeouts.append(timeout)
        clock.advance(200)
        return True

    monkeypatch.setattr(osworld, "run_agent_on_vm", fake_run)
    env = _Env(_Controller([ask, done]))
    env.user_simulator = _Simulator()
    args = _args(tmp_path, timeout=600)
    osworld._run_agent_episode(
        env, args, _model_cfg(), "do it", "prompt", str(tmp_path),
        ask_user=osworld_v2._make_ask_user_bridge(env, args, str(tmp_path), []),
        budget=osworld.TaskBudget(600).start(),
    )
    assert driver_timeouts == [600, 400]


def test_budget_stops_short_of_an_episode_too_small_to_be_worth_a_vm_trip(
        tmp_path, monkeypatch):
    ask = json.dumps({"text": "ASK_USER: Which account should I use?"}).encode() + b"\n"
    done = json.dumps({"text": "done"}).encode() + b"\n"
    clock = _Clock()
    driver_timeouts = []
    monkeypatch.setattr(osworld.time, "monotonic", clock)

    def fake_run(controller, command, timeout):
        driver_timeouts.append(timeout)
        clock.advance(590)
        return True

    monkeypatch.setattr(osworld, "run_agent_on_vm", fake_run)
    env = _Env(_Controller([ask, done]))
    env.user_simulator = _Simulator()
    args = _args(tmp_path, timeout=600)
    osworld._run_agent_episode(
        env, args, _model_cfg(), "do it", "prompt", str(tmp_path),
        ask_user=osworld_v2._make_ask_user_bridge(env, args, str(tmp_path), []),
        budget=osworld.TaskBudget(600).start(),
    )
    # 10s left is under MIN_EPISODE_SECONDS, so the follow-up is not launched
    # and the state the first episode reached is what gets scored.
    assert driver_timeouts == [600]


def test_first_episode_runs_however_small_the_budget(tmp_path, monkeypatch):
    # A cap below the floor is a deliberately tiny diagnostic configuration,
    # not a reason to run no agent at all. The 1 also pins the rounding: the
    # sub-second remainder must round up, because a truncated 0 is how GNU
    # timeout spells "no limit".
    driver_timeouts = []
    monkeypatch.setattr(
        osworld, "run_agent_on_vm",
        lambda controller, command, timeout: driver_timeouts.append(timeout) or True,
    )
    osworld._run_agent_episode(
        _Env(), _args(tmp_path, timeout=1), _model_cfg(), "do it", "prompt",
        str(tmp_path), budget=osworld.TaskBudget(1).start(),
    )
    assert driver_timeouts == [1]


def test_no_time_limit_budget_leaves_every_episode_uncapped(tmp_path, monkeypatch):
    ask = json.dumps({"text": "ASK_USER: Which account should I use?"}).encode() + b"\n"
    done = json.dumps({"text": "done"}).encode() + b"\n"
    driver_timeouts = []
    monkeypatch.setattr(
        osworld, "run_agent_on_vm",
        lambda controller, command, timeout: driver_timeouts.append(timeout) or True,
    )
    env = _Env(_Controller([ask, done]))
    env.user_simulator = _Simulator()
    args = _args(tmp_path, timeout=600, no_time_limit=True)
    osworld._run_agent_episode(
        env, args, _model_cfg(), "do it", "prompt", str(tmp_path),
        ask_user=osworld_v2._make_ask_user_bridge(env, args, str(tmp_path), []),
        budget=osworld.TaskBudget(None).start(),
    )
    assert driver_timeouts == [None, None]


def test_budget_clock_starts_after_the_vm_is_prepared(tmp_path, monkeypatch):
    # Provisioning cost varies with the provider; charging it to the agent
    # would make the same cap mean different things on different mornings.
    clock = _Clock()
    driver_timeouts = []
    monkeypatch.setattr(osworld.time, "monotonic", clock)
    monkeypatch.setattr(osworld.time, "sleep", lambda _: None)

    def slow_prepare(env, example, args, model_cfg, shadow_dir):
        clock.advance(900)          # VM allocation, reset, interface upload
        return "prompt", False

    monkeypatch.setattr(osworld_v2, "prepare_task_vm", slow_prepare)
    monkeypatch.setattr(
        osworld, "run_agent_on_vm",
        lambda controller, command, timeout: driver_timeouts.append(timeout) or True,
    )
    monkeypatch.setattr(
        osworld_v2, "finalize_task_result",
        lambda *a, **k: {"score": k.get("score")},
    )
    env = _Env()
    env.evaluate = lambda: 1.0
    osworld_v2.run_single_task(
        env, {"id": "007", "instruction": "do it"}, "do it", str(tmp_path),
        _args(tmp_path, timeout=600), _model_cfg(), str(tmp_path),
    )
    assert driver_timeouts == [600]


def test_synthetic_zero_is_marked_ungraded_not_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(osworld.time, 'sleep', lambda _: None)
    monkeypatch.setattr(osworld_v2, 'prepare_task_vm', lambda *a: ('prompt', False))
    monkeypatch.setattr(osworld_v2, '_run_agent_episode', lambda *a, **kw: None)
    monkeypatch.setattr(osworld, '_extract_failure_summary', lambda p: pytest.fail('Dummy zero is not failure'))
    (tmp_path / 'claude_trajectory.jsonl').write_text('{}\n')
    env = _Env()
    env.evaluate = lambda: 0.0
    result = osworld_v2.run_single_task(
        env, {'id': 's_001', 'verification': 'dummy_constant_zero'}, 'do it', str(tmp_path),
        _args(tmp_path), _model_cfg(), str(tmp_path),
    )
    assert result['score'] == 0.0
    assert result['failure_summary'] == ''
    assert result['details']['verification'] == 'dummy_constant_zero'
    assert json.loads((tmp_path / 'result.json').read_text())['score_semantics'].startswith('UNGRADED')


def test_all_imputed_zero_summary_is_not_reused_as_a_completion(tmp_path):
    """An evaluator orphaned by a dying driver finishes every task as an
    imputed zero and writes a summary. Reusing it hands the next run a
    fabricated 0.0 that then drives selection, so it must be refused."""
    import json as _json
    from agent_s4.harness.ubuntu_vm import _load_complete_cached_summary

    summary = tmp_path / "latest_summary.json"
    summary.write_text(_json.dumps({
        "total": 2, "passed": 0, "failed": 2, "infrastructure_failed": 0,
        "pass_rate": 0.0, "mean_score": 0.0,
        "results": [
            {"task_id": "a", "score": 0.0, "imputed_zero": True,
             "failure_type": "agent_execution_failure", "valid_for_fitness": True},
            {"task_id": "b", "score": 0.0, "imputed_zero": True,
             "failure_type": "agent_execution_failure", "valid_for_fitness": True},
        ],
    }))
    assert _load_complete_cached_summary(
        summary, ["a", "b"], benchmark="osworld_v2") is None


def test_one_genuine_zero_keeps_the_cache_trustworthy(tmp_path):
    """The guard is narrow on purpose: a real graded zero is a real result."""
    import json as _json
    from agent_s4.harness.ubuntu_vm import _load_complete_cached_summary

    summary = tmp_path / "latest_summary.json"
    summary.write_text(_json.dumps({
        "total": 2, "passed": 0, "failed": 2, "infrastructure_failed": 0,
        "pass_rate": 0.0, "mean_score": 0.0,
        "results": [
            {"task_id": "a", "score": 0.0, "valid_for_fitness": True},
            {"task_id": "b", "score": 0.0, "imputed_zero": True,
             "failure_type": "agent_execution_failure", "valid_for_fitness": True},
        ],
    }))
    assert _load_complete_cached_summary(
        summary, ["a", "b"], benchmark="osworld_v2") is not None


_LENGTH_CUT = (
    b'{"type":"message_end","message":{"role":"assistant","stopReason":"length",'
    b'"content":[{"type":"text","text":"long reasoning\xc2\x85more"}]}}\n'
    b'{"type":"agent_end"}\n'
)


def test_length_truncated_reply_without_tool_call_is_detected():
    assert osworld_v2._length_truncated_without_tool(_LENGTH_CUT)
    # turn_end alone carries the truncated message too (no message_end).
    turn_only = (
        b'{"type":"turn_end","message":{"role":"assistant","stopReason":"length",'
        b'"content":[{"type":"text","text":"x"}]}}\n'
    )
    assert osworld_v2._length_truncated_without_tool(turn_only)
    with_tool = (
        b'{"type":"message_end","message":{"role":"assistant","stopReason":"length",'
        b'"content":[{"type":"toolCall","name":"ipython"}]}}\n'
    )
    assert not osworld_v2._length_truncated_without_tool(with_tool)
    normal = b'{"type":"message_end","message":{"role":"assistant","stopReason":"stop","content":[]}}\n'
    assert not osworld_v2._length_truncated_without_tool(normal)


def test_length_continuation_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("SHADOW_EVOL_LENGTH_CONTINUATIONS", raising=False)
    env = _Env(_Controller([]))
    ask = osworld_v2._make_ask_user_bridge(env, _args(tmp_path), str(tmp_path), [])
    assert ask(_LENGTH_CUT, 1, None) is None
    assert not (tmp_path / "length_continuations.jsonl").exists()


def test_length_continuation_resumes_up_to_the_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("SHADOW_EVOL_LENGTH_CONTINUATIONS", "2")
    env = _Env(_Controller([]))
    ask = osworld_v2._make_ask_user_bridge(env, _args(tmp_path), str(tmp_path), [])
    assert ask(_LENGTH_CUT, 1, None) == osworld_v2.LENGTH_CONTINUATION_PROMPT
    assert ask(_LENGTH_CUT, 2, None) == osworld_v2.LENGTH_CONTINUATION_PROMPT
    assert ask(_LENGTH_CUT, 3, None) is None
    records = (tmp_path / "length_continuations.jsonl").read_text().splitlines()
    assert [json.loads(r)["continuation"] for r in records] == [1, 2]
