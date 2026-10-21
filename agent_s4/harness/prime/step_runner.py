"""Relay Prime Agent JSONL while enforcing a model-step ceiling."""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import subprocess
import sys
import time


def _stop_process_group(process: subprocess.Popen, grace_seconds: int = 10) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _is_assistant_step(line: str) -> bool:
    """Treat each completed assistant message as one Prime model step."""
    try:
        event = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(event, dict) or event.get("type") != "message_end":
        return False
    message = event.get("message")
    # Observed Prime versions either attach the role or omit it because a
    # message_end event is intrinsically assistant-side.
    return (
        not isinstance(message, dict) or message.get("role", "assistant") == "assistant"
    )


def run(
    command: list[str],
    step_limit: int,
    idle_timeout: int = 0,
    model_request_timeout: int = 0,
    state_path: str | None = None,
) -> int:
    if step_limit <= 0:
        raise ValueError("step_limit must be positive")
    if idle_timeout < 0:
        raise ValueError("idle_timeout must be non-negative")
    if model_request_timeout < 0:
        raise ValueError("model_request_timeout must be non-negative")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
        )
    except OSError as exc:
        print(
            json.dumps(
                {
                    "type": "prime_agent_launch_error",
                    "error": str(exc),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 127

    def forward_signal(signum, _frame):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)

    steps = 0
    assert process.stdout is not None
    idle_deadline = time.monotonic() + idle_timeout if idle_timeout else None
    pending = b""
    phase = "startup"
    last_event = None
    phase_started = time.monotonic()
    model_deadline = None

    def publish_phase():
        if state_path:
            temporary = f"{state_path}.{os.getpid()}.tmp"
            with open(temporary, "w") as state:
                json.dump({"phase": phase, "started_at": time.time()}, state)
            os.replace(temporary, state_path)

    publish_phase()
    while True:
        deadline = model_deadline if model_deadline is not None else idle_deadline
        if deadline is not None:
            remaining = max(0.0, deadline - time.monotonic())
            readable, _, _ = select.select([process.stdout], [], [], remaining)
            if not readable or (
                model_deadline is not None and time.monotonic() >= model_deadline
            ):
                print(
                    json.dumps(
                        {
                            "type": (
                                "prime_agent_model_timeout"
                                if model_deadline is not None
                                else "prime_agent_idle_timeout"
                            ),
                            "idle_seconds": (
                                idle_timeout if model_deadline is None else None
                            ),
                            "budget_seconds": (
                                model_request_timeout
                                if model_deadline is not None
                                else idle_timeout
                            ),
                            "elapsed_seconds": round(
                                time.monotonic() - phase_started, 3
                            ),
                            "phase": phase,
                            "last_event": last_event,
                            "steps": steps,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                _stop_process_group(process)
                return 125
        # Never combine select(fd) with TextIOWrapper.readline(): readline
        # can prefetch several JSONL records, leaving the fd empty while
        # complete events remain in its private buffer. That falsely trips
        # the idle watchdog and kills a healthy agent.
        chunk = os.read(process.stdout.fileno(), 65536)
        if not chunk:
            if pending:
                sys.stdout.write(pending.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            return process.wait()
        if idle_deadline is not None:
            idle_deadline = time.monotonic() + idle_timeout
        pending += chunk
        records = pending.split(b"\n")
        pending = records.pop()
        for record in records:
            line = record.decode("utf-8", errors="replace") + "\n"
            sys.stdout.write(line)
            sys.stdout.flush()
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                event = {}
            if isinstance(event, dict):
                last_event = event.get("type")
                # The pinned runtime emits compaction_start/end. Older Prime
                # versions used auto_compaction_*; both wrap a model request
                # and need the same buffered-provider budget as a normal turn.
                # Prime does not always emit compaction_start: a threshold
                # compaction can follow agent_end with no event until the
                # summary returns, so agent_end opens the model budget too.
                if (
                    last_event
                    in {
                        "turn_start",
                        "compaction_start",
                        "auto_compaction_start",
                        "agent_end",
                    }
                    and phase != "model"
                ):
                    phase = "model"
                    phase_started = time.monotonic()
                    if model_request_timeout:
                        model_deadline = phase_started + model_request_timeout
                    publish_phase()
                elif (
                    last_event == "message_end" and _is_assistant_step(line)
                ) or last_event in {
                    "compaction_end",
                    "auto_compaction_end",
                    "tool_execution_start",
                }:
                    phase = (
                        "tool"
                        if last_event == "tool_execution_start"
                        else "between_events"
                    )
                    phase_started = time.monotonic()
                    model_deadline = None
                    publish_phase()
            if _is_assistant_step(line):
                steps += 1
            if steps >= step_limit:
                print(
                    json.dumps(
                        {
                            "type": "prime_step_limit",
                            "limit": step_limit,
                            "steps": steps,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                _stop_process_group(process)
                return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-limit", type=int, required=True)
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=0,
        help="Kill Prime Agent after this many seconds without a JSONL event; "
        "zero disables the no-progress watchdog.",
    )
    parser.add_argument(
        "--model-request-timeout",
        type=int,
        default=0,
        help="Hard request budget for buffered model providers, beginning at turn_start; "
        "tool/startup inactivity still uses --idle-timeout.",
    )
    parser.add_argument(
        "--state-path", help="Atomic phase metadata for host liveness checks."
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a Prime Agent command is required after --")
    return run(
        command,
        args.step_limit,
        args.idle_timeout,
        args.model_request_timeout,
        args.state_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
