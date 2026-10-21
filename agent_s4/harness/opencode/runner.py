"""Standalone OpenCode supervisor, staged into Builder clones and task VMs.

The runner and CLI take file paths, never task/prompt text, in argv. Process
cleanup targets only the CLI's new process group, never pkill -f or killall.
This is accidental-termination protection, not a sandbox against malicious
tools running as the same Unix user.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

EXPECTED_VERSION = "1.18.34"
TOOLCALL_PROMPT = (
    "Your previous reply could not be executed: its tool call was not valid JSON, "
    "so nothing ran. Continue the task in the same session and environment. Call "
    "one tool, written as plain JSON, for example\n<tool_call>\n"
    '{"name": "bash", "arguments": {"command": "ls /home/user"}}\n</tool_call>\n'
    "Write long file contents in several smaller steps instead of one large argument."
)
# The prompt-emulated tool-calling proxy's give-up message (HTTP 422).
TOOLCALL_ERROR = "No valid tool call"
LENGTH_PROMPT = (
    "Your last reply reached the output-token limit before finishing the task. "
    "Continue in the same session and environment. Keep reasoning short, act "
    "using a tool, and verify the result. Do not repeat your previous analysis."
)


def emit(kind: str, **fields) -> None:
    print(json.dumps({"type": kind, **fields}, ensure_ascii=False), flush=True)


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value) + "\n", encoding="utf-8")
    tmp.replace(path)


def isolated_env(config: dict, state_dir: Path, inherited: dict | None = None) -> dict:
    env = dict(os.environ if inherited is None else inherited)
    # Stale user flags must not disable compaction or replace the selected model.
    env = {k: v for k, v in env.items() if not k.startswith("OPENCODE_")}
    for name in ("config", "data", "state", "cache"):
        directory = state_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        env[f"XDG_{name.upper()}_HOME"] = str(directory)
    provider = config["enabled_providers"][0]
    output = next(iter(config["provider"][provider]["models"].values()))["limit"][
        "output"
    ]
    env.update(
        {
            "OPENCODE_CONFIG_CONTENT": json.dumps(config),
            "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
            "OPENCODE_DISABLE_CLAUDE_CODE": "true",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "true",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
            "OPENCODE_DISABLE_MODELS_FETCH": "true",
            "OPENCODE_DISABLE_LSP_DOWNLOAD": "true",
            "OPENCODE_DISABLE_AUTOUPDATE": "true",
            "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": str(output),
        }
    )
    return env


def stop_group(proc: subprocess.Popen, grace: float = 2) -> None:
    """Reap the CLI and any surviving tools, including after a normal exit."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        proc.poll()
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            proc.wait()
            return
        time.sleep(0.02)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run(args: argparse.Namespace) -> int:
    state = args.state_dir.resolve()
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    settings = json.loads(args.settings.read_text()) if args.settings else {}
    idle_timeout = settings.get("idle_timeout", 1920)
    max_continuations = settings.get("length_continuations", 0)
    max_recoveries = settings.get("toolcall_recoveries", 0)
    recoveries = 0
    env = isolated_env(config, state)
    # `opencode run` resolves its workspace from PWD before process.cwd().
    # Popen(cwd=...) alone leaves the parent's PWD in the environment.
    env["PWD"] = str(args.cwd.resolve())
    check = subprocess.run(
        [args.binary, "--version"], env=env, capture_output=True, text=True, timeout=30
    )
    if check.returncode or check.stdout.strip() != EXPECTED_VERSION:
        raise ValueError(f"OpenCode version must be {EXPECTED_VERSION}")
    session_path = state / "session.json"
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    session = None
    steps = 0
    if args.resume:
        record = json.loads(session_path.read_text())
        if record.get("config_sha256") != signature:
            raise ValueError(
                "refusing to resume an OpenCode session with changed configuration"
            )
        session = record["session_id"]
        steps = record["steps"]
    elif session_path.exists():
        raise ValueError(
            "OpenCode state already contains a session; use --resume or a fresh state directory"
        )
    total_cost = 0.0
    continuations = 0
    interrupted = []
    old_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(
            signum, lambda sig, frame: interrupted.append(sig)
        )
    started = time.monotonic()
    task = args.task_file.read_bytes()
    # A file rather than a pipe avoids deadlocks for large prompts and keeps
    # the task out of every parent/child argv, including the Python supervisor.
    input_path = state / "stdin.txt"
    reason = "incomplete"
    code = 1
    try:
        while True:
            if args.max_steps is not None and steps >= args.max_steps:
                reason, code = "step_limit", 0
                break
            input_path.write_bytes(task)
            command = [
                args.binary,
                "run",
                "--format",
                "json",
                "--model",
                config["model"],
                "--agent",
                "shadow",
                "--dir",
                str(args.cwd.resolve()),
            ]
            if session:
                command.extend(["--session", session])
            last_finish = None
            saw_error = False
            toolcall_error = False
            malformed = False
            seen_parts = set()
            last_activity = time.monotonic()
            buffer = b""
            stop_reason = None
            with input_path.open("rb") as stdin, args.stderr_file.open("ab") as stderr:
                proc = subprocess.Popen(
                    command,
                    cwd=args.cwd,
                    env=env,
                    stdin=stdin,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    start_new_session=True,
                )
                selector = selectors.DefaultSelector()
                selector.register(proc.stdout, selectors.EVENT_READ)

                def consume(line: bytes) -> None:
                    nonlocal session, steps, total_cost, last_finish, saw_error, malformed, toolcall_error
                    if not line.strip():
                        return
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            raise ValueError("event must be an object")
                    except (ValueError, UnicodeDecodeError):
                        malformed = True
                        emit("opencode_protocol_error", message="malformed JSONL event")
                        return
                    if (
                        event.get("type") == "error"
                        and recoveries < max_recoveries
                        and TOOLCALL_ERROR in json.dumps(event.get("error"))
                    ):
                        # Recoverable: recorded under its own type so the
                        # trajectory validator does not treat it as fatal.
                        toolcall_error = True
                        emit(
                            "opencode_toolcall_error",
                            sessionID=event.get("sessionID"),
                            error=event.get("error"),
                        )
                        return
                    print(line.decode("utf-8"), flush=True)
                    event_session = event.get("sessionID")
                    if event_session:
                        if session and session != event_session:
                            malformed = True
                            return
                        session = event_session
                    if event.get("type") == "error":
                        saw_error = True
                    if event.get("type") == "step_finish":
                        part = event.get("part", {})
                        identifier = part.get("id")
                        if identifier is None or identifier not in seen_parts:
                            seen_parts.add(identifier)
                            steps += 1
                            total_cost += float(part.get("cost", 0) or 0)
                        last_finish = part.get("reason")
                    if session:
                        atomic_json(
                            session_path,
                            {
                                "session_id": session,
                                "steps": steps,
                                "config_sha256": signature,
                            },
                        )

                try:
                    while True:
                        now = time.monotonic()
                        if interrupted:
                            stop_reason = "signal"
                        elif args.timeout is not None and now - started >= args.timeout:
                            stop_reason = "wall_timeout"
                        elif now - last_activity >= idle_timeout:
                            stop_reason = "idle_timeout"
                        elif toolcall_error:
                            stop_reason = "toolcall_error"
                        elif malformed or saw_error:
                            stop_reason = (
                                "protocol_error" if malformed else "provider_error"
                            )
                        elif args.max_steps is not None and steps >= args.max_steps:
                            stop_reason = "step_limit"
                        if stop_reason:
                            break
                        ready = selector.select(timeout=0.1)
                        if ready:
                            chunk = os.read(proc.stdout.fileno(), 65536)
                            if not chunk:
                                break
                            last_activity = now
                            buffer += chunk
                            while b"\n" in buffer:
                                line, buffer = buffer.split(b"\n", 1)
                                consume(line)
                        elif proc.poll() is not None:
                            # Tools may inherit stdout after the CLI dies. Do
                            # not wait for those tools to close the pipe.
                            break
                    if buffer.strip() and stop_reason is None:
                        consume(buffer)
                    returncode = proc.poll()
                    if returncode is None and stop_reason is None:
                        try:
                            returncode = proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            stop_reason = "incomplete"
                finally:
                    selector.close()
                    stop_group(proc)
                    proc.stdout.close()
            if toolcall_error and session:
                recoveries += 1
                emit(
                    "opencode_toolcall_recovery", attempt=recoveries, sessionID=session
                )
                task = TOOLCALL_PROMPT.encode()
                continue
            if saw_error or malformed:
                reason, code = ("provider_error" if saw_error else "protocol_error"), 1
                break
            if returncode is not None and returncode != 0:
                # A real crash that races the step budget must not be turned
                # into a successful budget stop by the supervisor.
                reason, code = "process_exit", (
                    128 - returncode if returncode < 0 else returncode
                )
                break
            if stop_reason:
                reason = stop_reason
                code = (
                    0
                    if reason == "step_limit"
                    else (
                        124
                        if reason == "wall_timeout"
                        else 128 + interrupted[0] if reason == "signal" else 1
                    )
                )
                break
            if (
                last_finish == "length"
                and continuations < max_continuations
                and session
            ):
                continuations += 1
                emit(
                    "opencode_length_continuation",
                    attempt=continuations,
                    sessionID=session,
                )
                task = LENGTH_PROMPT.encode()
                continue
            if last_finish == "stop" and session:
                reason, code = "completed", 0
            elif last_finish == "length" and session:
                # Continuations exhausted. Prime ends such an episode as a
                # normal finish and the grader scores the VM state, so this is
                # an explicit, scoreable budget stop rather than a failure.
                reason, code = "length_limit", 0
            else:
                reason, code = "incomplete", 1
            break
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    diagnostics = {}
    if code not in {0, 124} and args.stderr_file.exists():
        with args.stderr_file.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 4000))
            diagnostics["stderr_tail"] = stream.read().decode("utf-8", errors="replace")
    emit(
        "result",
        subtype=reason,
        is_error=code not in {0, 124},
        exit_code=code,
        total_cost_usd=total_cost,
        steps=steps,
        session_id=session,
        **diagnostics,
    )
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--stderr-file", type=Path, required=True)
    parser.add_argument("--exit-file", type=Path)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    code = 1
    try:
        if args.timeout is not None and args.timeout <= 0:
            raise ValueError("timeout must be positive")
        if args.max_steps is not None and args.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        code = run(args)
    except Exception as exc:
        emit(
            "result",
            subtype="runner_error",
            is_error=True,
            exit_code=1,
            error=f"{type(exc).__name__}: {exc}",
            total_cost_usd=0,
        )
    finally:
        if args.exit_file:
            args.exit_file.write_text(str(code) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
