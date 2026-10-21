"""JSONL compatibility adapter for the official Prime Agent CLI.

This module deliberately does not implement an agent loop.  It launches
``prime-agent --mode json`` and relays every upstream event unchanged, so
session persistence, tool execution, and automatic context compaction remain
owned by Prime Agent.  The only added record is the terminal ``result`` event
expected by :func:`agent_s4.harness.builder_common.parse_builder_log`.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, TextIO


def build_prime_command(
    *,
    prime_bin: str,
    task: str,
    system_prompt: str,
    model: str,
    cwd: Path,
    session_dir: Path,
    extensions: Iterable[Path] = (),
) -> list[str]:
    """Build the supported headless invocation without disabling sessions."""
    extension_args = [arg for path in extensions for arg in ("--extension", str(path))]
    return [
        prime_bin,
        "--mode",
        "json",
        # Do not use Prime's TMPDIR-derived shared socket here. Builder TMPDIR
        # is a short-lived Landlock-private directory and the detached daemon
        # startup can race its default socket-directory bookkeeping. Keep the
        # socket beside this builder's explicitly persisted session directory.
        "--daemon-socket",
        str(session_dir.parent / "daemon.sock"),
        "--cwd",
        str(cwd),
        "--model",
        model,
        # Keep Prime Agent's upstream RLM/tool instructions and add only the
        # benchmark-specific policy. Replacing the system prompt would remove
        # part of the harness the integration is intended to preserve.
        "--append-system-prompt",
        system_prompt,
        "--no-context-files",
        "--offline",
        "--session-dir",
        str(session_dir),
        *extension_args,
        "--",
        task,
    ]


def _assistant_cost(event: dict[str, Any]) -> float:
    """Return cost from the one non-duplicated terminal message event."""
    if event.get("type") != "message_end":
        return 0.0
    message = event.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return 0.0
    usage = message.get("usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    total = cost.get("total") if isinstance(cost, dict) else None
    try:
        return float(total or 0.0)
    except (TypeError, ValueError):
        return 0.0


def relay_jsonl(lines: Iterable[str], output: TextIO) -> tuple[float, bool, bool]:
    """Relay native events and return ``(cost, saw_agent_end, saw_error)``.

    Prime Agent promises JSON lines in this mode.  A future version emitting a
    diagnostic on stdout is still represented as JSON, keeping the builder log
    parseable while preserving the original text.
    """
    total_cost = 0.0
    saw_agent_end = False
    saw_fatal_error = False
    pending_assistant_error = False
    tool_executions = 0
    successful_tool_executions = 0
    for line in lines:
        raw = line.rstrip("\r\n")
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            event = {"type": "prime_agent_raw", "text": raw}
            output.write(json.dumps(event, ensure_ascii=False) + "\n")
        else:
            # Preserve valid native JSON bytes except for newline normalization.
            output.write(raw + "\n")
            if isinstance(event, dict):
                total_cost += _assistant_cost(event)
                saw_agent_end = saw_agent_end or event.get("type") == "agent_end"
                message = event.get("message")
                saw_fatal_error = saw_fatal_error or event.get("type") in {
                    "agent_error",
                    "prime_agent_launch_error",
                }
                # Provider timeouts are emitted as an assistant message_end
                # with stopReason=error, but Prime can retry the turn and
                # continue to a normal agent_end. Keep only the most recent
                # completed assistant turn's status so a recovered transient
                # error does not poison the entire builder run. Explicit
                # agent_error/launch-error events remain fatal above.
                if (
                    event.get("type") == "message_end"
                    and isinstance(message, dict)
                    and message.get("role") == "assistant"
                ):
                    pending_assistant_error = message.get("stopReason") == "error"
                if event.get("type") == "tool_execution_end":
                    tool_executions += 1
                    if event.get("isError") is not True:
                        successful_tool_executions += 1
        output.flush()
    # A model can gracefully explain that it is blocked and emit agent_end
    # even when its only tool failed on every attempt. That is not a usable
    # mutation. Allow isolated/recovered tool failures, but reject a run with
    # no successful tool execution at all.
    all_tools_failed = tool_executions > 0 and successful_tool_executions == 0
    return (
        total_cost,
        saw_agent_end,
        saw_fatal_error or pending_assistant_error or all_tools_failed,
    )


def run(args: argparse.Namespace) -> int:
    args.session_dir.mkdir(parents=True, exist_ok=True)
    command = build_prime_command(
        prime_bin=args.prime_bin,
        task=args.task,
        system_prompt=args.system_prompt,
        model=args.model,
        cwd=args.cwd,
        session_dir=args.session_dir,
        extensions=args.extension,
    )
    stderr_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
    try:
        try:
            process = subprocess.Popen(
                command,
                cwd=str(args.cwd),
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                encoding="utf-8",
            )
        except OSError as exc:
            print(
                json.dumps(
                    {
                        "type": "prime_agent_launch_error",
                        "error": str(exc),
                    }
                ),
                flush=True,
            )
            return 127

        # The outer builder timeout targets this adapter process. Forward its
        # termination to Prime Agent so a timed-out run cannot leave an orphaned
        # model/tool process behind.
        def forward_signal(signum: int, _frame: Any) -> None:
            if process.poll() is None:
                process.send_signal(signum)

        signal.signal(signal.SIGTERM, forward_signal)
        signal.signal(signal.SIGINT, forward_signal)

        assert process.stdout is not None
        total_cost, saw_agent_end, saw_error = relay_jsonl(
            process.stdout,
            sys.stdout,
        )
        exit_code = process.wait()
        stderr_file.seek(0)
        stderr_text = stderr_file.read()
        if stderr_text:
            print(
                json.dumps(
                    {
                        "type": "prime_agent_stderr",
                        "text": stderr_text[-32768:],
                        "truncated": len(stderr_text) > 32768,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        print(
            json.dumps(
                {
                    "type": "result",
                    "subtype": "prime_agent_result",
                    "total_cost_usd": total_cost,
                    "prime_agent_exit_code": exit_code,
                    "saw_agent_end": saw_agent_end,
                    "is_error": exit_code != 0 or not saw_agent_end or saw_error,
                }
            ),
            flush=True,
        )
        if exit_code != 0:
            return exit_code
        return 0 if saw_agent_end and not saw_error else 1
    finally:
        stderr_file.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime-bin", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--system-prompt", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--extension", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
