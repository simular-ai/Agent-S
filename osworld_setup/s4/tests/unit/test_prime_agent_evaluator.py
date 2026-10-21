from __future__ import annotations

import base64
import json
import sys
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_vm_command_is_official_json_mode_with_isolated_compaction(monkeypatch):
    from agent_s4.harness.prime.evaluator import (
        build_prime_agent_vm_command,
    )

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    built = build_prime_agent_vm_command(
        "do the task", "system policy", "anthropic/claude-sonnet-4-6", 600,
        api_key="secret",
    )
    command = built.script
    assert base64.b64encode(b"do the task").decode() in command
    assert base64.b64encode(
        b"system policy\n\n## Prime Agent OSWorld runtime"
    ).decode() in command
    assert "/opt/prime-agent/bin/prime-agent --mode json" in command
    assert "--cwd /home/user" in command
    assert "--session-dir /tmp/claude_task/prime-sessions" in command
    assert "--no-context-files" in command
    assert "--append-system-prompt" in command
    assert " --system-prompt " not in command
    assert "--no-session" not in command
    assert "PRIME_AGENT_CODING_AGENT_DIR=/tmp/claude_task/prime-config" in command
    assert "PRIME_AGENT_KERNEL_PYTHON=/opt/prime-agent/python/bin/python3" in command
    assert '"compaction":{"enabled":true,"reserveTokens":16384,"keepRecentTokens":20000}' in command
    assert '"autoRefine":{"enabled":false}' in command
    assert '"defaultThinkingLevel":"medium"' in command
    assert '"telemetry":{"enabled":false}' in command
    assert "secret" not in command
    assert built.secret_env == {"ANTHROPIC_API_KEY": "secret"}
    assert ". /tmp/.shadow-agent-env" in command
    assert "--offline" in command
    assert "> /tmp/claude_task/trajectory.jsonl" in command
    assert "cp /tmp/claude_task/trajectory.jsonl /tmp/claude_task/stdout.log" in command
    assert "trap 'status=$?" in command
    assert "exit_code.txt' EXIT" in command
    assert built.accepted_exit_codes == (0, 124)
    assert built.activity_path == "/tmp/claude_task/trajectory.jsonl"
    assert built.idle_timeout == 420
    assert built.process_pattern == "[p]rime_step_runner.py"
    assert built.successful_terminal_events == (
        "agent_end", "prime_step_limit", "prime_agent_salvaged",
    )
    assert built.runner_exit_grace == 60
    assert built.salvage_min_steps == 50
    assert built.salvage_terminal_event == "prime_agent_salvaged"


def test_continuation_reuses_prime_session_and_only_updates_prompt(monkeypatch):
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    built = build_prime_agent_vm_command(
        "benchmark follow-up", "policy", "anthropic/model", 600,
        api_key="secret", continue_session=True,
    )
    assert "--continue < /tmp/claude_task/instruction.txt" in built.script
    assert "rm -rf /tmp/claude_task" not in built.script
    assert "mkdir -p /tmp/claude_task/prime-config" in built.script
    assert "secret" not in built.script


def test_fresh_episode_stops_daemon_before_replacing_task_state(monkeypatch):
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    built = build_prime_agent_vm_command(
        "phase two", "policy", "anthropic/model", 600, api_key="secret",
    )

    shutdown = "/opt/prime-agent/bin/prime-agent shutdown --force"
    reset = "rm -rf /tmp/claude_task"
    assert shutdown in built.script
    assert built.script.index(shutdown) < built.script.index(reset)


def test_continuation_does_not_stop_live_daemon(monkeypatch):
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    built = build_prime_agent_vm_command(
        "answer", "policy", "anthropic/model", 600,
        api_key="secret", continue_session=True,
    )

    assert "shutdown --force" not in built.script


def test_vm_command_registers_openai_compatible_qwen_provider(monkeypatch):
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.setenv("OPENAI_API_KEY", "qwen-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://qwen.example/v1")
    built = build_prime_agent_vm_command(
        "benchmark task", "policy", "openai//deployments/abc/Qwen/Qwen3.8", 600,
    )

    assert "--model shadow-openai-compatible//deployments/abc/Qwen/Qwen3.8" in built.script
    assert "> /tmp/claude_task/prime-config/models.json" in built.script
    assert '\"baseUrl\":\"https://qwen.example/v1\"' in built.script
    assert '\"api\":\"openai-completions\"' in built.script
    assert '\"thinkingFormat\":\"qwen\"' in built.script
    # Not a free parameter: an 8000 cap terminated 33 of 45 episodes under a
    # proxy that returns reasoning inline in content, because every reply that
    # reached it ended its episode. See DEFAULT_MAX_OUTPUT_TOKENS.
    assert '\"maxTokens\":32000' in built.script
    assert "qwen-secret" not in built.script
    assert built.secret_env["OPENAI_API_KEY"] == "qwen-secret"
    assert built.secret_env["OPENAI_BASE_URL"] == "https://qwen.example/v1"


def test_vm_command_sends_no_temperature_unless_opted_in(monkeypatch):
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.delenv("SHADOW_PRIME_AGENT_TEMPERATURE", raising=False)
    built = build_prime_agent_vm_command("task", "policy", "openai/qwen", 600)
    assert "--extension" not in built.script
    assert "sampling_temperature.ts" not in built.script


def test_vm_command_pins_opted_in_temperature_through_an_extension(monkeypatch):
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.setenv("SHADOW_PRIME_AGENT_TEMPERATURE", "1")
    for continue_session in (False, True):
        script = build_prime_agent_vm_command(
            "task", "policy", "openai/qwen", 600, continue_session=continue_session,
        ).script
        extension = "/tmp/claude_task/sampling_temperature.ts"
        assert f"--extension {extension}" in script
        # A Prime flag, placed before the stdin redirect that carries the task.
        assert script.index(f"--extension {extension}") < script.index(
            " < /tmp/claude_task/instruction.txt"
        )
        sources = [
            base64.b64decode(part.split(" ")[0]).decode()
            for part in script.split("echo ")[1:]
            if f"base64 -d > {extension}" in part
        ]
        assert len(sources) == 1
        assert "before_provider_request" in sources[0]
        assert "temperature: 1.0" in sources[0]
        assert "payload.temperature === undefined" in sources[0]


@pytest.mark.parametrize("raw", ["-0.1", "2.5", "hot"])
def test_vm_command_rejects_an_invalid_temperature(monkeypatch, raw):
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.setenv("SHADOW_PRIME_AGENT_TEMPERATURE", raw)
    with pytest.raises(ValueError):
        build_prime_agent_vm_command("task", "policy", "openai/qwen", 600)


def test_vm_command_registers_openrouter_qwen_with_provider_routing(monkeypatch):
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "alibaba,coreweave/fp8")
    monkeypatch.setenv("OPENROUTER_ALLOW_FALLBACKS", "true")
    monkeypatch.setenv("OPENROUTER_ZDR", "true")
    monkeypatch.setenv("OPENROUTER_DATA_COLLECTION", "deny")
    monkeypatch.setenv("SHADOW_PRIME_AGENT_THINKING_LEVEL", "high")
    built = build_prime_agent_vm_command(
        "benchmark task", "policy", "openrouter/qwen/qwen3.8-27b", 600,
        openrouter_api_key="or-secret",
    )

    assert "--model openrouter/qwen/qwen3.8-27b" in built.script
    assert "export SHADOW_GROUNDING_MODEL=openrouter/qwen/qwen3.8-27b" in built.script
    assert '"providers":{"openrouter":{"compat":' in built.script
    assert (
        '"openRouterRouting":{"allow_fallbacks":true,'
        '"order":["alibaba","coreweave/fp8"],"zdr":true,'
        '"data_collection":"deny"}' in built.script
    )
    assert '"defaultThinkingLevel":"high"' in built.script
    assert "or-secret" not in built.script
    assert built.secret_env["OPENROUTER_API_KEY"] == "or-secret"


def test_openrouter_context_window_is_explicitly_overridden_when_provided(monkeypatch):
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.delenv("OPENROUTER_PROVIDER_ORDER", raising=False)
    built = build_prime_agent_vm_command(
        "benchmark task", "policy", "openrouter/qwen/qwen3.8-27b", 600,
        openrouter_api_key="or-secret", context_window_tokens=131072,
    )

    assert (
        '"modelOverrides":{"qwen/qwen3.8-27b":{"contextWindow":131072}}'
        in built.script
    )


def test_openrouter_context_window_uses_model_catalog_when_omitted(monkeypatch):
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.delenv("OPENROUTER_PROVIDER_ORDER", raising=False)
    built = build_prime_agent_vm_command(
        "benchmark task", "policy", "openrouter/qwen/qwen3.8-27b", 600,
        openrouter_api_key="or-secret",
    )

    assert '"modelOverrides"' not in built.script
    assert '"contextWindow"' not in built.script


def test_builtin_provider_context_window_can_be_explicitly_overridden(monkeypatch):
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    built = build_prime_agent_vm_command(
        "benchmark task", "policy", "anthropic/claude-sonnet-4-6", 600,
        api_key="secret", context_window_tokens=131072,
    )

    assert (
        '"providers":{"anthropic":{"modelOverrides":'
        '{"claude-sonnet-4-6":{"contextWindow":131072}}}}'
        in built.script
    )


def test_osworld_passes_variant_context_window_to_prime(tmp_path):
    from agent_s4.eval.osworld import _build_agent_command

    shadow_dir = tmp_path / "shadow_interface"
    shadow_dir.mkdir()
    (shadow_dir / "context_compaction.json").write_text(
        json.dumps({"context_window_tokens": 131072}), encoding="utf-8",
    )
    captured = {}

    def build_command(**kwargs):
        captured.update(kwargs)
        return "prime command"

    args = SimpleNamespace(
        model="openrouter/qwen/qwen3.8-27b",
        timeout=1200,
        shadow_dir=str(shadow_dir),
        max_steps=300,
        no_time_limit=True,
        or_provider=None,
    )
    model_cfg = {
        "binary": "prime-agent",
        "model_flag": args.model,
        "build_command": build_command,
    }

    assert _build_agent_command(
        args, model_cfg, "task", "policy",
        api_key=None, openrouter_key="secret", parasail_key=None,
    ) == "prime command"
    assert captured["context_window_tokens"] == 131072


def test_osworld_omits_variant_context_override_when_file_is_absent(tmp_path):
    from agent_s4.eval.osworld import _build_agent_command

    captured = {}

    def build_command(**kwargs):
        captured.update(kwargs)
        return "prime command"

    args = SimpleNamespace(
        model="openrouter/qwen/qwen3.8-27b",
        timeout=1200,
        shadow_dir=str(tmp_path / "missing"),
        max_steps=300,
        no_time_limit=True,
        or_provider=None,
    )
    model_cfg = {
        "binary": "prime-agent",
        "model_flag": args.model,
        "build_command": build_command,
    }

    _build_agent_command(
        args, model_cfg, "task", "policy",
        api_key=None, openrouter_key="secret", parasail_key=None,
    )
    assert captured["context_window_tokens"] is None


def test_explicit_openrouter_provider_order_wins_over_environment(monkeypatch):
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.delenv("OPENROUTER_ALLOW_FALLBACKS", raising=False)
    monkeypatch.delenv("OPENROUTER_ZDR", raising=False)
    monkeypatch.delenv("OPENROUTER_DATA_COLLECTION", raising=False)
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "coreweave/fp8")
    built = build_prime_agent_vm_command(
        "benchmark task", "policy", "openrouter/qwen/qwen3.8-27b", 600,
        openrouter_api_key="or-secret", or_provider=["alibaba"],
    )

    assert (
        '"openRouterRouting":{"allow_fallbacks":true,"order":["alibaba"],'
        '"data_collection":"allow"}' in built.script
    )
    assert "coreweave/fp8" not in built.script


def test_openrouter_no_fallback_is_provider_exclusive(monkeypatch):
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "novita")
    monkeypatch.setenv("OPENROUTER_ALLOW_FALLBACKS", "false")
    monkeypatch.setenv("OPENROUTER_ZDR", "true")
    monkeypatch.setenv("OPENROUTER_DATA_COLLECTION", "deny")
    built = build_prime_agent_vm_command(
        "benchmark task", "policy", "openrouter/qwen/qwen3.8-27b", 600,
        openrouter_api_key="or-secret",
    )

    assert (
        '"openRouterRouting":{"allow_fallbacks":false,'
        '"order":["novita"],"only":["novita"],"zdr":true,'
        '"data_collection":"deny"}' in built.script
    )


def test_trajectory_validator_requires_native_terminal_events():
    from agent_s4.harness.prime.evaluator import validate_prime_agent_artifacts

    valid = b'{"type":"agent_start"}\n{"type":"agent_end"}\n'
    assert validate_prime_agent_artifacts({"claude_trajectory": valid}) is None
    assert "malformed" in validate_prime_agent_artifacts({
        "claude_trajectory": b'{"type":"agent_start"}\nnot-json\n',
    })
    assert "missing" in validate_prime_agent_artifacts({
        "claude_trajectory": b'{"type":"agent_start"}\n',
    })


def test_trajectory_validator_accepts_parseable_wall_clock_timeout():
    from agent_s4.harness.prime.evaluator import validate_prime_agent_artifacts

    artifacts = {
        "claude_trajectory": (
            b'{"type":"agent_start"}\n'
            b'{"type":"turn_start"}\n'
        ),
        "exit_code": b"124\n",
    }
    assert validate_prime_agent_artifacts(artifacts) is None


def test_trajectory_validator_keeps_raw_u0085_inside_a_record():
    from agent_s4.harness.prime.evaluator import validate_prime_agent_artifacts

    # Qwen writes an unescaped NEL (U+0085) inside JSON strings; it is legal
    # JSON and must not split the record.
    raw = '{"type":"agent_start"}\n{"type":"message_end","text":"a\x85b"}\n{"type":"agent_end"}\n'
    assert validate_prime_agent_artifacts({"claude_trajectory": raw.encode()}) is None


def test_trajectory_validator_accepts_wall_clock_timeout_cut_mid_record():
    from agent_s4.harness.prime.evaluator import validate_prime_agent_artifacts

    truncated = b'{"type":"agent_start"}\n{"type":"turn_start"}\n{"type":"message_upd'
    assert validate_prime_agent_artifacts({
        "claude_trajectory": truncated, "exit_code": b"124\n",
    }) is None
    # Without the timeout the same cut-off record is still a harness failure,
    # and so is a malformed record anywhere but the end.
    assert "malformed" in validate_prime_agent_artifacts({
        "claude_trajectory": truncated, "exit_code": b"1\n",
    })
    assert "line 2" in validate_prime_agent_artifacts({
        "claude_trajectory": b'{"type":"agent_start"}\nnot-json\n{"type":"turn_start"}\n',
        "exit_code": b"124\n",
    })


def test_vm_command_supports_step_limit_without_wall_clock_limit(monkeypatch):
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    built = build_prime_agent_vm_command(
        "task", "policy", "anthropic/model", 1200,
        api_key="secret", max_steps=300, no_time_limit=True,
    )

    assert (
        "prime_step_runner.py --step-limit 300 --idle-timeout 360 --"
        in built.script
    )
    assert "timeout 1200" not in built.script


def test_trajectory_validator_accepts_prime_step_limit():
    from agent_s4.harness.prime.evaluator import validate_prime_agent_artifacts

    raw = b'{"type":"agent_start"}\n{"type":"prime_step_limit","limit":300,"steps":300}\n'
    assert validate_prime_agent_artifacts({"claude_trajectory": raw}) is None


def test_trajectory_validator_accepts_salvaged_long_episode():
    from agent_s4.harness.prime.evaluator import validate_prime_agent_artifacts

    raw = (
        b'{"type":"agent_start"}\n'
        b'{"type":"prime_agent_salvaged","reason":"runner_disappeared",'
        b'"steps":176}\n'
    )
    assert validate_prime_agent_artifacts({"claude_trajectory": raw}) is None


def test_prime_step_counter_counts_only_assistant_message_end():
    import json
    from agent_s4.harness.prime.step_runner import _is_assistant_step

    assert _is_assistant_step(json.dumps({
        "type": "message_end", "message": {"role": "assistant"},
    }))
    assert not _is_assistant_step(json.dumps({
        "type": "message_end", "message": {"role": "toolResult"},
    }))
    assert not _is_assistant_step('{"type":"message_update"}')


def test_prime_step_runner_stops_a_silent_agent():
    import sys
    import time

    from agent_s4.harness.prime.step_runner import run

    started = time.monotonic()
    exit_code = run(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        step_limit=300,
        idle_timeout=1,
    )
    assert exit_code == 125
    assert time.monotonic() - started < 5


def test_prime_step_runner_drains_events_from_one_write(capsys):
    from agent_s4.harness.prime.step_runner import run

    # Both records arrive in one pipe write, then no new fd data arrives.
    # A buffered readline/select implementation loses the second event and
    # kills the agent as idle instead of recognizing its completed step.
    child = (
        "import os,time; "
        "os.write(1,b'{\"type\":\"turn_start\"}\\n"
        "{\"type\":\"message_end\"}\\n'); time.sleep(30)"
    )
    assert run([sys.executable, "-c", child], 1, idle_timeout=1) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["type"] for event in events] == [
        "turn_start", "message_end", "prime_step_limit",
    ]


def test_prime_step_runner_times_out_on_unfinished_line(capsys):
    from agent_s4.harness.prime.step_runner import run

    child = "import os,time; os.write(1,b'{'); time.sleep(30)"
    assert run([sys.executable, "-c", child], 1, idle_timeout=1) == 125
    assert json.loads(capsys.readouterr().out)["type"] == "prime_agent_idle_timeout"


def test_buffered_model_request_can_outlast_tool_idle_budget(capsys, tmp_path):
    from agent_s4.harness.prime.step_runner import run
    events = [
        {"type": "turn_start"},
        {"type": "message_end", "message": {"role": "user"}},
    ]
    first = "".join(json.dumps(e) + "\n" for e in events).encode()
    child = (f"import os,time; os.write(1,{first!r}); time.sleep(1.3); "
             "os.write(1,b'{\"type\":\"message_end\"}\\n'); time.sleep(30)")
    state = tmp_path / "state.json"
    assert run([sys.executable, "-c", child], 1, 1, 4, str(state)) == 0
    assert "prime_agent_idle_timeout" not in capsys.readouterr().out
    assert json.loads(state.read_text())["phase"] == "between_events"


def test_tool_still_uses_short_idle_budget(capsys):
    from agent_s4.harness.prime.step_runner import run
    child = ("import os,time; os.write(1,b'{\"type\":\"turn_start\"}\\n"
             "{\"type\":\"message_end\"}\\n{\"type\":\"tool_execution_start\"}\\n'); time.sleep(30)")
    assert run([sys.executable, "-c", child], 10, 1, 4) == 125
    event = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert event["type"] == "prime_agent_idle_timeout"
    assert event["phase"] == "tool"
    assert event["budget_seconds"] == 1


@pytest.mark.parametrize("heartbeat", ["message_update", "turn_start"])
def test_model_request_deadline_is_not_extended_by_heartbeat(capsys, heartbeat):
    from agent_s4.harness.prime.step_runner import run
    child = ("import os,time; os.write(1,b'{\"type\":\"turn_start\"}\\n'); "
             "exec(\"while True:\\n os.write(1,b'{\\\"type\\\":\\\"HEARTBEAT\\\"}\\\\n'); time.sleep(0.1)\")").replace("HEARTBEAT", heartbeat)
    assert run([sys.executable, "-c", child], 10, 1, 2) == 125
    event = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert event["type"] == "prime_agent_model_timeout"
    assert event["phase"] == "model"
    assert event["budget_seconds"] == 2
    assert 2 <= event["elapsed_seconds"] < 4


def test_buffered_deployment_gets_model_budget_without_changing_tool_budget(monkeypatch):
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command
    monkeypatch.delenv("SHADOW_PRIME_MODEL_REQUEST_TIMEOUT", raising=False)
    unset = build_prime_agent_vm_command(
        "task", "policy", "openai/qwen3.8-27b-deployment", 1200,
        max_steps=300, no_time_limit=True,
    )
    assert "--model-request-timeout" not in unset.script
    assert unset.model_request_timeout is None
    monkeypatch.setenv("SHADOW_PRIME_MODEL_REQUEST_TIMEOUT", "1800")
    command = build_prime_agent_vm_command(
        "task", "policy", "openai/qwen3.8-27b-deployment", 1200,
        max_steps=300, no_time_limit=True,
    )
    assert "--idle-timeout 360 --model-request-timeout 1800 --state-path" in command.script
    assert command.idle_timeout == 420
    assert command.model_request_timeout == 1860
    assert command.phase_path.endswith("runner_state.json")
    no_wrapper = build_prime_agent_vm_command(
        "task", "policy", "openai/qwen3.8-27b-deployment", 1200,
    )
    assert no_wrapper.model_request_timeout is None
    assert no_wrapper.phase_path is None


def test_timeout_cause_takes_priority_over_daemon_shutdown_stderr():
    from agent_s4.eval.osworld import _agent_failure_message
    event = {"type": "prime_agent_model_timeout", "phase": "model", "budget_seconds": 1800}
    message = _agent_failure_message({
        "claude_trajectory": json.dumps(event),
        "claude_stderr": "Daemon worker client closed",
    })
    assert "prime_agent_model_timeout" in message
    assert "1800" in message
    assert "Daemon worker" not in message


def test_prime_exit_trap_writes_plain_numeric_status(tmp_path):
    import subprocess
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    command = build_prime_agent_vm_command("task", "policy", "anthropic/model", 600)
    # Execute the actual generated trap under the VM's shell for both a
    # successful episode and an idle watchdog failure. Quotes in the marker
    # previously made even successful exit codes unparsable by the host.
    trap = command.script.split("trap ", 1)[1].split(" EXIT && ", 1)[0]
    trap = trap.replace("/tmp/claude_task", str(tmp_path))
    for status in (0, 124, 125):
        completed = subprocess.run(["/bin/sh", "-c", f"trap {trap} EXIT; exit {status}"])
        assert completed.returncode == status
        assert (tmp_path / "exit_code.txt").read_bytes() == f"{status}\n".encode()


def test_trajectory_validator_rejects_kernel_bootstrap_failure():
    import json
    from agent_s4.harness.prime.evaluator import validate_prime_agent_artifacts

    events = [
        {"type": "agent_start"},
        {
            "type": "tool_execution_end",
            "isError": True,
            "content": "PRIME_AGENT_KERNEL_PYTHON points to a Python missing default Python packages",
        },
        {"type": "agent_end"},
    ]
    raw = "".join(json.dumps(event) + "\n" for event in events).encode()
    assert validate_prime_agent_artifacts({"claude_trajectory": raw}) == (
        "Prime Agent kernel bootstrap validation failed"
    )


def test_installer_has_no_network_fallback(tmp_path, monkeypatch):
    from agent_s4.harness.prime import evaluator as module

    monkeypatch.setattr(module, "validate_prime_agent_source", lambda: "a" * 40)
    monkeypatch.setattr(module, "_resolve_runtime_cache", lambda: tmp_path / "missing")
    with patch.object(
        module, "_execute_shell_on_vm", return_value={"returncode": 1},
    ), patch.object(
        module, "validate_prime_agent_runtime_cache",
        side_effect=RuntimeError("missing"),
    ), patch.object(module, "upload_file_to_vm") as upload:
        assert module.install_prime_agent_on_vm(MagicMock()) is False
    upload.assert_not_called()


def test_runtime_probe_binds_commit_fingerprint_and_python(monkeypatch):
    from agent_s4.harness.prime import evaluator as module

    monkeypatch.setattr(module, "runtime_cache_fingerprint", lambda commit: "fp:" + commit)
    probe = module._runtime_probe("a" * 40)
    assert "/opt/prime-agent/upstream_commit" in probe
    assert "/opt/prime-agent/runtime_fingerprint" in probe
    assert "/opt/prime-agent/bin/prime-agent --version" in probe
    assert "--version 2>&1 | grep -Fqx" in probe
    assert "0.8.0-beta.546.1.a9b5d88" in probe
    assert "/opt/prime-agent/python/bin/python3" in probe


def test_coreweave_overflow_compatibility_patch_is_idempotent(tmp_path):
    from agent_s4.harness.prime.overflow_patch import (
        ANCHOR,
        COREWEAVE_PATTERN,
        patch_overflow_module,
    )

    overflow = tmp_path / "overflow.js"
    overflow.write_text(f"const OVERFLOW_PATTERNS = [\n{ANCHOR}\n];\n")
    assert patch_overflow_module(overflow) is True
    assert patch_overflow_module(overflow) is False
    assert overflow.read_text().count(COREWEAVE_PATTERN) == 1


def test_ipython_timeout_patch_is_retriable_and_idempotent(tmp_path):
    from agent_s4.harness.prime.ipython_timeout_patch import (
        CONSTANT_ANCHOR,
        DESCRIPTION_ANCHOR,
        DESTRUCTURE_ANCHOR,
        ERROR_TEXT_ANCHOR,
        EXECUTE_END,
        EXECUTE_START,
        IS_ERROR_ANCHOR,
        STATUS_ANCHOR,
        TIMEOUT_CONSTANT,
        patch_ipython_module,
    )

    module = tmp_path / "ipython.js"
    module.write_text("\n".join((
        CONSTANT_ANCHOR,
        EXECUTE_START,
        "    original body",
        EXECUTE_END.lstrip("\n"),
        DESTRUCTURE_ANCHOR,
        ERROR_TEXT_ANCHOR,
        STATUS_ANCHOR,
        IS_ERROR_ANCHOR,
        DESCRIPTION_ANCHOR,
    )))

    assert patch_ipython_module(module) is True
    assert patch_ipython_module(module) is False
    patched = module.read_text()
    assert TIMEOUT_CONSTANT in patched
    assert "IPython cell exceeded 60 seconds" in patched
    assert "timedOut ||" in patched
    assert "provisioner.kill()" in patched


def test_ipython_timeout_patch_covers_core_and_executable_bundle(tmp_path):
    from agent_s4.harness.prime.ipython_timeout_patch import (
        CONSTANT_ANCHOR,
        DESCRIPTION_ANCHOR,
        DESTRUCTURE_ANCHOR,
        ERROR_TEXT_ANCHOR,
        EXECUTE_END,
        EXECUTE_START,
        IS_ERROR_ANCHOR,
        STATUS_ANCHOR,
        TIMEOUT_CONSTANT,
        patch_ipython_runtime,
    )

    template = "\n".join((
        CONSTANT_ANCHOR,
        EXECUTE_START,
        "    original body",
        EXECUTE_END.lstrip("\n"),
        DESTRUCTURE_ANCHOR,
        ERROR_TEXT_ANCHOR,
        STATUS_ANCHOR,
        IS_ERROR_ANCHOR,
        DESCRIPTION_ANCHOR,
    ))
    core = tmp_path / "core" / "tools" / "ipython.js"
    bundle = tmp_path / "bundle" / "chunk-pinned.js"
    unrelated = tmp_path / "bundle" / "other.js"
    for path in (core, bundle, unrelated):
        path.parent.mkdir(parents=True, exist_ok=True)
    core.write_text(template)
    bundle.write_text(template)
    unrelated.write_text("console.log('not ipython')")

    assert patch_ipython_runtime(tmp_path) == [bundle, core]
    assert TIMEOUT_CONSTANT in core.read_text()
    assert TIMEOUT_CONSTANT in bundle.read_text()
    assert unrelated.read_text() == "console.log('not ipython')"


def test_installer_applies_coreweave_overflow_overlay(monkeypatch):
    from agent_s4.harness.prime import evaluator as module

    monkeypatch.setattr(module, "validate_prime_agent_source", lambda: "a" * 40)
    with patch.object(
        module, "_execute_shell_on_vm",
        side_effect=[{"returncode": 0}, {"returncode": 0}],
    ) as execute:
        assert module.install_prime_agent_on_vm(MagicMock()) is True
    command = execute.call_args_list[1].args[1]
    assert "prime_overflow_patch.py" in command
    assert "decoder prompt.*longer than the maximum model length" in command
    assert "prime_ipython_timeout_patch.py" in command
    assert "IPython cell exceeded 60 seconds" in command
    assert "/dist/bundle" in command


def test_official_artifact_pin_matches_vendored_commit():
    from agent_s4.harness.prime.source import (
        PINNED_PRIME_AGENT_COMMIT,
        PRIME_AGENT_ARTIFACT_SHA256,
        PRIME_AGENT_ARTIFACT_URL,
        PRIME_AGENT_VERSION,
    )

    assert PRIME_AGENT_VERSION.endswith(PINNED_PRIME_AGENT_COMMIT[:7])
    assert PRIME_AGENT_VERSION in PRIME_AGENT_ARTIFACT_URL
    assert PRIME_AGENT_ARTIFACT_SHA256 == (
        "830d2b10656a7e04bece4b0d895b66fa0aa86e04bfc2088d03b00c27ef015831"
    )


def test_osworld_v2_cli_accepts_prime_agent_headlessly(monkeypatch):
    from agent_s4.eval import osworld_v2

    monkeypatch.setenv("SHADOW_EVOL_DRYRUN", "1")
    with patch.object(sys, "argv", [
        "osworld_v2", "--harness", "prime_agent", "--model",
        "anthropic/claude-sonnet-4-6", "--task_ids", "task-1",
    ]):
        args = osworld_v2.config()
    assert args.harness == "prime_agent"
    assert args.model_cfg["binary"] == "prime-agent"


def test_installer_extracts_validated_archive(tmp_path, monkeypatch):
    from agent_s4.harness.prime import evaluator as module

    cache = tmp_path / "prime_agent_runtime.tar.gz"
    cache.write_bytes(b"archive")
    monkeypatch.setattr(module, "validate_prime_agent_source", lambda: "a" * 40)
    monkeypatch.setattr(module, "_resolve_runtime_cache", lambda: cache)
    monkeypatch.setattr(module, "validate_prime_agent_runtime_cache", lambda path: "a" * 40)
    with patch.object(
        module, "_execute_shell_on_vm",
        side_effect=[{"returncode": 1}, {"returncode": 0}],
    ) as execute, patch.object(module, "upload_file_to_vm", return_value=True):
        assert module.install_prime_agent_on_vm(MagicMock(), "pw") is True
    assert "sudo -S" in execute.call_args_list[1].args[1]


def test_osworld_omits_variant_context_override_when_pinned_to_catalog(
    tmp_path, monkeypatch,
):
    """The evaluator has to agree with the Builder on the pinned source.

    A variant whose interface says 131072 must still be evaluated at Prime's
    catalog window while a run is pinned there, or the resumed steps of
    prime25v15r4 would be scored in a different window from their parents.
    """
    from agent_s4.eval.osworld import _build_agent_command

    monkeypatch.setenv("SHADOW_PRIME_CONTEXT_WINDOW_SOURCE", "catalog")
    shadow_dir = tmp_path / "shadow_interface"
    shadow_dir.mkdir()
    (shadow_dir / "context_compaction.json").write_text(
        json.dumps({"context_window_tokens": 131072}), encoding="utf-8",
    )
    captured = {}

    def build_command(**kwargs):
        captured.update(kwargs)
        return "prime command"

    args = SimpleNamespace(
        model="openrouter/qwen/qwen3.8-27b",
        timeout=1200,
        shadow_dir=str(shadow_dir),
        max_steps=300,
        no_time_limit=True,
        or_provider=None,
    )
    model_cfg = {
        "binary": "prime-agent",
        "model_flag": args.model,
        "build_command": build_command,
    }

    _build_agent_command(
        args, model_cfg, "task", "policy",
        api_key=None, openrouter_key="secret", parasail_key=None,
    )
    assert captured["context_window_tokens"] is None


def test_vm_base_url_override_reaches_the_guest(monkeypatch):
    """A proxy inside the benchmark VPC has one address the orchestrator can
    route and another the task VM can. The guest config and the guest env must
    both carry the VM's address, not the host's."""
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.setenv("OPENAI_BASE_URL", "http://54.0.0.1:4000/v1")
    monkeypatch.setenv("SHADOW_EVOL_VM_OPENAI_BASE_URL", "http://10.0.0.10:4000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    command = build_prime_agent_vm_command(
        instruction="do the thing",
        system_prompt="be useful",
        model="openai/some-qwen-deployment",
        timeout=600,
    )
    assert '"baseUrl":"http://10.0.0.10:4000/v1"' in command.script
    assert "54.0.0.1" not in command.script
    assert command.secret_env["OPENAI_BASE_URL"] == "http://10.0.0.10:4000/v1"


def test_vm_base_url_falls_back_to_the_shared_one(monkeypatch):
    """With no split-horizon address, behaviour is unchanged."""
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.delenv("SHADOW_EVOL_VM_OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    command = build_prime_agent_vm_command(
        instruction="do the thing", system_prompt="be useful",
        model="openai/some-qwen-deployment", timeout=600,
    )
    assert command.secret_env["OPENAI_BASE_URL"] == "https://api.example.com/v1"


def test_idle_timeout_is_configurable_and_host_waits_longer(monkeypatch):
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("SHADOW_PRIME_IDLE_TIMEOUT", "720")
    built = build_prime_agent_vm_command(
        "task", "policy", "anthropic/model", 1200,
        api_key="secret", max_steps=300, no_time_limit=True,
    )

    assert "--idle-timeout 720 --" in built.script
    assert built.idle_timeout == 780


def test_task_text_never_reaches_the_prime_command_line(monkeypatch):
    """An agent's `pkill -f <app>` for an app named in its task must not match
    its own runtime (083: `pkill -f zotero` killed Prime, exit 143)."""
    from agent_s4.harness.prime.evaluator import build_prime_agent_vm_command

    monkeypatch.setenv("OPENAI_BASE_URL", "http://proxy:4000/v1")
    for continue_session in (False, True):
        built = build_prime_agent_vm_command(
            "Organise the papers in Zotero\n", "Use the Zotero helpers.\n",
            "openai/qwen3.8-27b-deployment", 600,
            max_steps=300, continue_session=continue_session,
        )
        assert "Zotero" not in built.script
        assert "$(cat" not in built.script
        assert "--append-system-prompt /tmp/claude_task/system_prompt.txt" in built.script
        assert " < /tmp/claude_task/instruction.txt > /tmp/claude_task/trajectory.jsonl" in built.script
        written = {
            path: base64.b64decode(encoded).decode()
            for encoded, path in (
                part.split(" | base64 -d > ", 1)
                for part in (chunk.split(" && ")[0] for chunk in built.script.split("echo ")[1:])
                if " | base64 -d > " in part
            )
        }
        # Same bytes the old "$(cat ...)" argv delivered: trailing newlines stripped.
        assert written["/tmp/claude_task/instruction.txt"] == "Organise the papers in Zotero"
        assert written["/tmp/claude_task/system_prompt.txt"].startswith("Use the Zotero helpers.")
        assert not written["/tmp/claude_task/system_prompt.txt"].endswith("\n")
