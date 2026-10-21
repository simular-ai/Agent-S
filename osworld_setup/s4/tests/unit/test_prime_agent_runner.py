from __future__ import annotations

import io
import json
from pathlib import Path

from agent_s4.harness.prime.runner import (
    build_prime_command,
    relay_jsonl,
)


def test_command_uses_official_headless_json_mode_and_keeps_session(tmp_path):
    command = build_prime_command(
        prime_bin="/opt/prime-agent/prime-agent",
        task="fix it",
        system_prompt="builder rules",
        model="openai/gpt-5",
        cwd=tmp_path,
        session_dir=tmp_path / "sessions",
    )
    assert command[:3] == ["/opt/prime-agent/prime-agent", "--mode", "json"]
    assert command[command.index("--daemon-socket") + 1] == str(
        tmp_path / "daemon.sock"
    )
    assert command[command.index("--cwd") + 1] == str(tmp_path)
    assert command[command.index("--append-system-prompt") + 1] == "builder rules"
    assert "--system-prompt" not in command
    assert "--no-context-files" in command
    assert "--offline" in command
    assert "--session-dir" in command
    assert "--no-session" not in command
    assert command[-2:] == ["--", "fix it"]


def test_relay_preserves_native_events_and_counts_message_end_cost_once():
    events = [
        {"type": "session", "version": 3},
        {"type": "compaction_start", "reason": "threshold"},
        {"type": "message_end", "message": {
            "role": "assistant", "usage": {"cost": {"total": 0.125}},
        }},
        {"type": "turn_end", "message": {
            "role": "assistant", "usage": {"cost": {"total": 0.125}},
        }},
        {"type": "compaction_end", "reason": "threshold", "aborted": False},
        {"type": "agent_end", "messages": []},
    ]
    native = [json.dumps(event, separators=(",", ":")) + "\n" for event in events]
    output = io.StringIO()
    cost, ended, errored = relay_jsonl(native, output)
    assert output.getvalue() == "".join(native)
    assert cost == 0.125
    assert ended is True
    assert errored is False


def test_relay_wraps_unexpected_stdout_as_json():
    output = io.StringIO()
    cost, ended, errored = relay_jsonl(["startup diagnostic\n"], output)
    assert json.loads(output.getvalue()) == {
        "type": "prime_agent_raw", "text": "startup diagnostic",
    }
    assert cost == 0.0
    assert ended is False
    assert errored is False


def test_relay_marks_assistant_stop_error_even_when_agent_ends():
    native = [
        json.dumps({
            "type": "message_end",
            "message": {"role": "assistant", "stopReason": "error"},
        }) + "\n",
        '{"type":"agent_end","messages":[]}\n',
    ]
    cost, ended, errored = relay_jsonl(native, io.StringIO())
    assert cost == 0.0
    assert ended is True
    assert errored is True


def test_relay_allows_recovery_after_assistant_timeout():
    native = [
        json.dumps({
            "type": "message_end",
            "message": {
                "role": "assistant",
                "stopReason": "error",
                "errorMessage": "A Timeout Occurred",
            },
        }) + "\n",
        json.dumps({
            "type": "turn_end",
            "message": {
                "role": "assistant",
                "stopReason": "error",
                "errorMessage": "A Timeout Occurred",
            },
        }) + "\n",
        json.dumps({
            "type": "message_end",
            "message": {"role": "assistant", "stopReason": "stop"},
        }) + "\n",
        '{"type":"agent_end","messages":[]}\n',
    ]
    _, ended, errored = relay_jsonl(native, io.StringIO())
    assert ended is True
    assert errored is False


def test_relay_does_not_recover_explicit_agent_error():
    native = [
        '{"type":"agent_error","error":"fatal"}\n',
        json.dumps({
            "type": "message_end",
            "message": {"role": "assistant", "stopReason": "stop"},
        }) + "\n",
        '{"type":"agent_end","messages":[]}\n',
    ]
    _, ended, errored = relay_jsonl(native, io.StringIO())
    assert ended is True
    assert errored is True


def test_relay_rejects_agent_end_when_every_tool_failed():
    native = [
        '{"type":"tool_execution_end","toolName":"ipython",'
        '"isError":true,"result":{"content":[]}}\n',
        '{"type":"agent_end","messages":[]}\n',
    ]
    _, ended, errored = relay_jsonl(native, io.StringIO())
    assert ended is True
    assert errored is True


def test_relay_allows_recovery_after_tool_failure():
    native = [
        '{"type":"tool_execution_end","toolName":"ipython",'
        '"isError":true,"result":{"content":[]}}\n',
        '{"type":"tool_execution_end","toolName":"ipython",'
        '"isError":false,"result":{"content":[]}}\n',
        '{"type":"agent_end","messages":[]}\n',
    ]
    _, ended, errored = relay_jsonl(native, io.StringIO())
    assert ended is True
    assert errored is False


def test_command_loads_extensions_before_the_task(tmp_path):
    extension = tmp_path / "sampling_temperature.ts"
    command = build_prime_command(
        prime_bin="prime-agent", task="fix it", system_prompt="rules",
        model="m", cwd=tmp_path, session_dir=tmp_path / "sessions",
        extensions=[extension],
    )
    assert command[command.index("--extension") + 1] == str(extension)
    assert command.index("--extension") < command.index("--")
    assert command[-1] == "fix it"


def test_command_has_no_extension_by_default(tmp_path):
    command = build_prime_command(
        prime_bin="prime-agent", task="fix it", system_prompt="rules",
        model="m", cwd=tmp_path, session_dir=tmp_path / "sessions",
    )
    assert "--extension" not in command
