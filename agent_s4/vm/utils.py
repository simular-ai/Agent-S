"""Generic VM mechanics shared by the evaluator harness and scratch-vm
provisioning.

Imports from desktop_env.* resolve via the benchmark checkout's
desktop_env package — no sys.path manipulation needed.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import shlex
import tarfile
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger("agent_s4.vm.utils")

VM_SHADOW_DIR = "/home/user/shadow_interface"
# Each block is embedded in a shell command that the VM runs as a single
# argument to /bin/sh -c, so the *encoded* block has to stay under Linux's
# MAX_ARG_STRLEN (32 pages = 131072 bytes) -- a per-argument cap that ARG_MAX
# does not relax. 64 KiB raw encodes to 87384 bytes, leaving room for the
# command around it. Going higher fails on the VM with
# "[Errno 7] Argument list too long", not with a clean error.
FILE_UPLOAD_RAW_CHUNK_SIZE = 64 * 1024  # 87384 bytes after base64
MAX_ARG_STRLEN = 32 * 4096


@dataclass(frozen=True)
class AgentCommand:
    """A launch script plus credentials that must not appear in its argv."""

    script: str
    secret_env: dict[str, str] = field(default_factory=dict)
    accepted_exit_codes: tuple[int, ...] = (0,)
    # Optional host-side liveness supervision for agents whose in-VM wrapper
    # can disappear before publishing exit_code.txt. This is deliberately an
    # inactivity limit, not an episode wall-clock limit.
    activity_path: str | None = None
    idle_timeout: int | None = None
    model_request_timeout: int | None = None
    phase_path: str | None = None
    process_pattern: str | None = None
    successful_terminal_events: tuple[str, ...] = ()
    # A runner can disappear before its parent shell publishes exit_code.txt.
    # Give that shell a short chance to finish normally, rather than waiting
    # for the much longer no-activity timeout.
    runner_exit_grace: int | None = None
    # Long episodes have already produced a meaningful desktop state. If the
    # runner disappears after this many completed assistant steps, preserve
    # that state as a scoreable agent outcome instead of consuming an infra
    # retry and restarting the task from zero.
    salvage_min_steps: int | None = None
    salvage_terminal_event: str = "agent_episode_salvaged"


def _probe_agent_liveness(controller, command: AgentCommand):
    """Return lightweight in-VM state without downloading a huge trajectory."""
    if not command.activity_path:
        return None
    # The bracketed pgrep pattern matches the real process but not the probe's
    # own argv. Read only the final JSONL record, which is small for Prime's
    # terminal events even when the complete trajectory is hundreds of MiB.
    probe_source = """\
import json, os, subprocess, time
p = %r
result = {"exists": os.path.isfile(p), "age_s": None, "last_event": None,
          "runner_alive": None}
if result["exists"]:
    result["age_s"] = max(0.0, time.time() - os.path.getmtime(p))
    with open(p, "rb") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        if end:
            stream.seek(end - 1)
            if stream.read(1) == b"\\n":
                end -= 1
        pos = end
        start = 0
        while pos > 0:
            take = min(8192, pos)
            pos -= take
            stream.seek(pos)
            block = stream.read(take)
            newline = block.rfind(b"\\n")
            if newline >= 0:
                start = pos + newline + 1
                break
        stream.seek(start)
        last_line = stream.read(end - start).strip()
    if last_line:
        try:
            result["last_event"] = json.loads(last_line).get("type")
        except (json.JSONDecodeError, AttributeError):
            result["last_event"] = "malformed"
pattern = %r
phase_path = %r
if phase_path:
    try:
        with open(phase_path) as stream:
            phase = json.load(stream)
        result["phase"] = phase["phase"]
        result["phase_elapsed_s"] = max(0.0, time.time() - phase["started_at"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
if pattern:
    result["runner_alive"] = subprocess.run(
        ["pgrep", "-f", pattern], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
print(json.dumps(result, separators=(",", ":")))
""" % (command.activity_path, command.process_pattern, command.phase_path)
    encoded = base64.b64encode(probe_source.encode()).decode("ascii")
    result = _execute_shell_on_vm(
        controller,
        "python3 -c "
        + shlex.quote("import base64;exec(base64.b64decode(" + repr(encoded) + "))"),
        timeout=20,
    )
    if result is None or result.get("returncode", -1) != 0:
        return None
    try:
        return json.loads(result.get("output", ""))
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse agent liveness probe output")
        return None


def _count_completed_assistant_steps(controller, activity_path: str) -> int | None:
    """Count completed assistant messages without downloading the trajectory."""
    source = """\
import json
p = %r
steps = 0
try:
    with open(p, "rb") as stream:
        for raw in stream:
            if b'"type":"message_end"' not in raw:
                continue
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(event, dict) or event.get("type") != "message_end":
                continue
            message = event.get("message")
            if not isinstance(message, dict) or message.get("role", "assistant") == "assistant":
                steps += 1
except OSError:
    raise SystemExit(2)
print(steps)
""" % activity_path
    encoded = base64.b64encode(source.encode()).decode("ascii")
    result = _execute_shell_on_vm(
        controller,
        "python3 -c "
        + shlex.quote("import base64;exec(base64.b64decode(" + repr(encoded) + "))"),
        timeout=120,
    )
    if result is None or result.get("returncode", -1) != 0:
        return None
    try:
        return int(result.get("output", "").strip())
    except (TypeError, ValueError):
        return None


def _mark_agent_episode_salvaged(
    controller,
    command: AgentCommand,
    steps: int,
) -> bool:
    """Append a scoreable terminal event and publish a successful marker."""
    source = """\
import json, os, time
p = %r
event_type = %r
steps = %d
if not os.path.isfile(p):
    raise SystemExit(2)
with open(p, "ab+") as stream:
    stream.seek(0, os.SEEK_END)
    if stream.tell():
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) != b"\\n":
            stream.write(b"\\n")
    event = {
        "type": event_type,
        "reason": "runner_disappeared",
        "steps": steps,
        "timestamp": int(time.time() * 1000),
    }
    stream.write((json.dumps(event, separators=(",", ":")) + "\\n").encode())
with open("/tmp/claude_task/exit_code.txt", "w") as marker:
    marker.write("0\\n")
""" % (command.activity_path, command.salvage_terminal_event, steps)
    encoded = base64.b64encode(source.encode()).decode("ascii")
    result = _execute_shell_on_vm(
        controller,
        "python3 -c "
        + shlex.quote("import base64;exec(base64.b64decode(" + repr(encoded) + "))"),
        timeout=30,
    )
    return result is not None and result.get("returncode", -1) == 0


def _stop_agent_processes(controller):
    _execute_shell_on_vm(
        controller,
        # Keep this in sync with the command builders. In particular Qwen
        # uses ``qwen -p ... --yolo`` (there is no --approval-mode flag).
        "pkill -f '[c]laude -p|[q]wen -p|[o]pencode run|[p]rime-agent|"
        "[p]rime_step_runner.py' 2>/dev/null || true",
        timeout=15,
    )


# ---------------------------------------------------------------------------
# Low-level VM communication (shared by all agents)
# ---------------------------------------------------------------------------


def _execute_shell_on_vm(controller, command, timeout=120):
    # The VM server enforces its own limit (~120 s unless the payload names
    # one), so the client timeout alone never lengthened setup.sh.
    payload = json.dumps({"command": command, "shell": True, "timeout": timeout})
    try:
        response = requests.post(
            controller.http_server + "/execute",
            headers={"Content-Type": "application/json"},
            data=payload,
            timeout=timeout,
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(
                "Shell command failed (status %d): %s",
                response.status_code,
                response.text,
            )
            return None
    except Exception as e:
        logger.error("Shell command error: %s", e)
        return None


def _launch_on_vm(controller, command, shell=True):
    payload = json.dumps({"command": command, "shell": shell})
    try:
        response = requests.post(
            controller.http_server + "/setup/launch",
            headers={"Content-Type": "application/json"},
            data=payload,
            timeout=30,
        )
        if response.status_code == 200:
            # The setup server echoes the launched shell command in its
            # response. Agent commands commonly contain provider credentials,
            # so logging even a prefix can disclose a complete API key.
            logger.info("VM command launched successfully")
            return True
        else:
            # Error responses may echo the same command and credentials too.
            logger.error("VM command launch failed (status %d)", response.status_code)
            return False
    except Exception as e:
        logger.error("Launch error: %s", e)
        return False


def _stage_agent_secret_env(controller, secret_env):
    """Write a short-lived mode-0600 env file before the agent is launched."""
    if not secret_env:
        return True
    lines = "".join(
        f"export {name}={shlex.quote(value)}\n"
        for name, value in sorted(secret_env.items())
    )
    encoded = base64.b64encode(lines.encode()).decode("ascii")
    command = (
        "umask 077 && echo "
        + shlex.quote(encoded)
        + " | base64 -d > /tmp/.shadow-agent-env"
    )
    payload = json.dumps({"command": command, "shell": True})
    try:
        response = requests.post(
            controller.http_server + "/execute",
            headers={"Content-Type": "application/json"},
            data=payload,
            timeout=30,
        )
        if response.status_code != 200:
            logger.error("Credential staging failed (status %d)", response.status_code)
            return False
        result = response.json()
        if result.get("returncode") != 0:
            logger.error("Credential staging command failed")
            return False
        return True
    except Exception as exc:
        logger.error("Credential staging error: %s", exc)
        return False


def _sudo_cmd(command, password="password"):
    return f"echo '{password}' | sudo -S {command}"


# ---------------------------------------------------------------------------
# Shared VM setup
# ---------------------------------------------------------------------------


def copy_credentials_to_vm(controller, creds_json):
    encoded = base64.b64encode(creds_json.encode("utf-8")).decode("ascii")
    command = (
        f"mkdir -p /home/user/.claude && "
        f"echo '{encoded}' | base64 -d > /home/user/.claude/.credentials.json && "
        f"chmod 600 /home/user/.claude/.credentials.json && "
        f"chown -R user:user /home/user/.claude"
    )
    result = _execute_shell_on_vm(controller, command, timeout=30)
    if result is None or result.get("returncode", -1) != 0:
        logger.error("Failed to copy credentials: %s", result)
        return False
    return True


# ---------------------------------------------------------------------------
# Shadow interface deployment (shared)
# ---------------------------------------------------------------------------


def _tar_directory(dir_path, archive_name="shadow_interface"):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [
                d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")
            ]
            for f in files:
                if f.endswith(".pyc"):
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, dir_path)
                arcname = os.path.join(archive_name, rel_path)
                tar.add(full_path, arcname=arcname)
    return buf.getvalue()


def upload_file_to_vm(controller, src_path, vm_path):
    """Ship a local file to the VM as opaque bytes.

    Same chunked-base64 transport as upload_shadow_interface, but no
    inner tar — the source is treated as a blob. Used for shipping
    pre-built runtime archives where re-tarring
    on the host would be wasteful.

    Returns True on success, False on any failure.
    """
    src_path = str(src_path)
    logger.info("Uploading %s -> %s ...", src_path, vm_path)
    size = os.path.getsize(src_path)
    # Decode each bounded base64 block directly into the destination. This
    # keeps peak memory constant and roughly halves the HTTP round trips for
    # the Prime Agent runtime archive on every task VM.
    result = _execute_shell_on_vm(
        controller,
        f"rm -f {vm_path} && : > {vm_path}",
        timeout=30,
    )
    if result is None or result.get("returncode", -1) != 0:
        return False
    with open(src_path, "rb") as source:
        while raw := source.read(FILE_UPLOAD_RAW_CHUNK_SIZE):
            encoded = base64.b64encode(raw).decode("ascii")
            command = f"echo -n '{encoded}' | base64 -d >> {vm_path}"
            if len(command) >= MAX_ARG_STRLEN:
                logger.error(
                    "Upload block for %s is %d bytes, over MAX_ARG_STRLEN "
                    "(%d); lower FILE_UPLOAD_RAW_CHUNK_SIZE",
                    vm_path,
                    len(command),
                    MAX_ARG_STRLEN,
                )
                return False
            result = _execute_shell_on_vm(controller, command, timeout=30)
            if result is None or result.get("returncode", -1) != 0:
                return False

    if result is None or result.get("returncode", -1) != 0:
        logger.error("Failed to upload %s: %s", src_path, result)
        return False
    logger.info("Uploaded %s (%.1f MB)", vm_path, size / 1024 / 1024)
    return True


def upload_shadow_interface(controller, shadow_dir):
    logger.info("Uploading shadow interface to VM at %s ...", VM_SHADOW_DIR)
    tar_bytes = _tar_directory(shadow_dir)
    tar_b64 = base64.b64encode(tar_bytes).decode("ascii")

    chunk_size = 50000
    if len(tar_b64) > 100000:
        chunks = [
            tar_b64[i : i + chunk_size] for i in range(0, len(tar_b64), chunk_size)
        ]
        result = _execute_shell_on_vm(
            controller,
            f"rm -rf {VM_SHADOW_DIR} && echo -n '{chunks[0]}' > /tmp/shadow_tar_b64",
            timeout=30,
        )
        if result is None or result.get("returncode", -1) != 0:
            return False
        for chunk in chunks[1:]:
            result = _execute_shell_on_vm(
                controller,
                f"echo -n '{chunk}' >> /tmp/shadow_tar_b64",
                timeout=30,
            )
            if result is None or result.get("returncode", -1) != 0:
                return False
        result = _execute_shell_on_vm(
            controller,
            "base64 -d /tmp/shadow_tar_b64 | tar xz -C /home/user/ && rm /tmp/shadow_tar_b64",
            timeout=60,
        )
    else:
        result = _execute_shell_on_vm(
            controller,
            f"rm -rf {VM_SHADOW_DIR} && mkdir -p /home/user/ && "
            f"echo '{tar_b64}' | base64 -d | tar xz -C /home/user/",
            timeout=60,
        )

    if result is None or result.get("returncode", -1) != 0:
        logger.error("Failed to upload shadow interface: %s", result)
        return False
    logger.info("Shadow interface uploaded successfully")
    return True


def run_setup_script(controller, client_password="osworld-public-evaluation"):
    """Run the shadow_interface's setup.sh on the VM. Passes the VM password
    via CLIENT_PASSWORD so setup.sh can apt-install utilities under sudo.
    Timeout=600s — on a freshly booted VM the first apt update pulls every
    index from the image's hk.archive.ubuntu.com mirror, which measured 55 s
    even on a warm VM on 2026-10-07; 180 s left fresh VMs failing setup."""
    result = _execute_shell_on_vm(
        controller,
        f"cd {VM_SHADOW_DIR} && CLIENT_PASSWORD='{client_password}' bash setup.sh",
        timeout=600,
    )
    if result is None or result.get("returncode", -1) != 0:
        logger.warning("Setup script failed (non-fatal): %s", result)
        return False
    return True


def _get_agent_exit_marker(controller):
    """Poll an optional marker once; its absence is normal while running."""
    path = "/tmp/claude_task/exit_code.txt"
    server = getattr(controller, "http_server", None)
    if not server:
        return controller.get_file(path)
    try:
        response = requests.post(
            server + "/file",
            data={"file_path": path},
            timeout=15,
        )
        if response.status_code == 200:
            return response.content
        if response.status_code != 404:
            logger.warning(
                "Agent exit-marker poll failed (HTTP %d)", response.status_code
            )
    except requests.RequestException as exc:
        logger.warning("Agent exit-marker poll failed: %s", exc)
    return None


def run_agent_on_vm(controller, command, timeout):
    """Generic agent-on-VM driver: launch and poll /tmp/claude_task/exit_code.txt
    until it appears (or max_wait elapses). Used by Claude and Qwen harnesses."""
    accepted_exit_codes = (0,)
    liveness_command = None
    if isinstance(command, AgentCommand):
        liveness_command = command
        if not _stage_agent_secret_env(controller, command.secret_env):
            return False
        accepted_exit_codes = command.accepted_exit_codes
        command = command.script

    # Clear the marker synchronously before launching. Cleaning it only in the
    # asynchronous command leaves a race where the first poll can read the
    # previous episode's zero and falsely report success.
    cleared = _execute_shell_on_vm(
        controller,
        "rm -f /tmp/claude_task/exit_code.txt",
        timeout=15,
    )
    if cleared is None or cleared.get("returncode", -1) != 0:
        logger.error("Could not clear stale agent exit marker: %s", cleared)
        _execute_shell_on_vm(
            controller,
            "rm -f /tmp/.shadow-agent-env",
            timeout=15,
        )
        return False
    if not _launch_on_vm(controller, command, shell=True):
        _execute_shell_on_vm(
            controller,
            "rm -f /tmp/.shadow-agent-env",
            timeout=15,
        )
        return False
    poll_interval = 10
    max_wait = None if timeout is None else timeout + 120
    waited = 0
    runner_missing_since = None
    while max_wait is None or waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval
        invalid_exit_marker = False
        try:
            data = _get_agent_exit_marker(controller)
            if data is not None:
                exit_code = data.decode("utf-8").strip()
                logger.info(
                    "Agent finished with exit code: %s (after %ds)", exit_code, waited
                )
                try:
                    return int(exit_code) in accepted_exit_codes
                except ValueError:
                    logger.error("Invalid agent exit-code marker: %r", exit_code)
                    # A daemon/controller shutdown can corrupt the tiny exit
                    # marker even though the trajectory was copied intact.  Do
                    # not discard an otherwise completed episode before giving
                    # the terminal-event recovery below a chance to validate it.
                    invalid_exit_marker = True
        except Exception:
            pass
        if liveness_command is not None and liveness_command.activity_path:
            probe = _probe_agent_liveness(controller, liveness_command)
            if probe is not None:
                last_event = probe.get("last_event")
                runner_alive = probe.get("runner_alive")
                age_s = probe.get("age_s")
                # Prime also emits agent_end before compaction and continuation.
                # Only recover a missing marker after its runner has exited.
                if (
                    last_event in liveness_command.successful_terminal_events
                    and runner_alive is False
                ):
                    logger.warning(
                        "Agent emitted terminal event %s without an exit marker; "
                        "recovering the completed episode",
                        last_event,
                    )
                    marker = _execute_shell_on_vm(
                        controller,
                        "printf '0\\n' > /tmp/claude_task/exit_code.txt",
                        timeout=15,
                    )
                    return marker is not None and marker.get("returncode", -1) == 0
                if (
                    runner_alive is False
                    and liveness_command.runner_exit_grace is not None
                ):
                    if runner_missing_since is None:
                        runner_missing_since = waited
                        logger.warning(
                            "Agent runner disappeared; allowing %ds for normal "
                            "episode finalization",
                            liveness_command.runner_exit_grace,
                        )
                    missing_for = waited - runner_missing_since
                    if missing_for < liveness_command.runner_exit_grace:
                        continue

                    steps = None
                    if (
                        probe.get("exists")
                        and liveness_command.salvage_min_steps is not None
                    ):
                        steps = _count_completed_assistant_steps(
                            controller,
                            liveness_command.activity_path,
                        )
                    if (
                        steps is not None
                        and liveness_command.salvage_min_steps is not None
                        and steps >= liveness_command.salvage_min_steps
                    ):
                        logger.warning(
                            "Agent runner remained absent for %ds after %d "
                            "completed steps; preserving the desktop state "
                            "as a scoreable agent outcome",
                            missing_for,
                            steps,
                        )
                        if _mark_agent_episode_salvaged(
                            controller,
                            liveness_command,
                            steps,
                        ):
                            return True
                        logger.error("Could not publish salvaged agent outcome")
                    else:
                        logger.error(
                            "Agent runner remained absent for %ds with only "
                            "%s completed steps; treating it as infrastructure "
                            "failure",
                            missing_for,
                            "unknown" if steps is None else steps,
                        )
                    _stop_agent_processes(controller)
                    return False
                if runner_alive is not False:
                    runner_missing_since = None
                if (
                    not probe.get("exists")
                    and runner_alive is False
                    and liveness_command.idle_timeout is not None
                    and waited >= liveness_command.idle_timeout
                ):
                    logger.error(
                        "Agent runner disappeared before creating its activity file"
                    )
                    _stop_agent_processes(controller)
                    return False
                idle_budget = liveness_command.idle_timeout
                idle_age = age_s
                if (
                    liveness_command.model_request_timeout is not None
                    and probe.get("phase") == "model"
                ):
                    # The in-VM runner owns the hard model deadline. Allow it
                    # to expire before host liveness cleanup, while retaining
                    # the short inactivity budget for tool execution.
                    idle_budget = liveness_command.model_request_timeout
                    idle_age = probe.get("phase_elapsed_s", age_s)
                if (
                    idle_budget is not None
                    and isinstance(idle_age, (int, float))
                    and idle_age >= idle_budget
                ):
                    logger.error(
                        "Agent trajectory has been idle for %.1fs "
                        "(last_event=%s, runner_alive=%s)",
                        age_s,
                        last_event,
                        runner_alive,
                    )
                    _stop_agent_processes(controller)
                    return False
        if invalid_exit_marker:
            return False
        if waited % 60 == 0:
            logger.info("Still waiting for agent... (%ds elapsed)", waited)
    logger.warning("Agent did not finish within %ds", max_wait)
    _stop_agent_processes(controller)
    return False
