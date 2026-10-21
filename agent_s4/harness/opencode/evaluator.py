"""Pinned OpenCode VM harness with shared configuration and supervised sessions."""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
import shlex

from agent_s4.harness.opencode.config import (
    HERMES_PLUGIN,
    TOOL_LIMITS_PLUGIN,
    build_config,
    runner_settings,
)
from agent_s4.harness.opencode.source import (
    OPENCODE_VERSION,
    VM_BINARY,
    assets_dir,
    validate_runtime_cache,
)
from agent_s4.paths import PACKAGE_ROOT
from agent_s4.vm.utils import (
    AgentCommand,
    _execute_shell_on_vm,
    _sudo_cmd,
    upload_file_to_vm,
)

logger = logging.getLogger(__name__)
TASK_ROOT = "/tmp/claude_task"
STATE_DIR = f"{TASK_ROOT}/opencode-state"
VM_RUNTIME_TARBALL = "/tmp/opencode_runtime.tar.gz"
RUNNER = PACKAGE_ROOT / "harness" / "opencode" / "runner.py"
VM_HERMES_PLUGIN = f"{TASK_ROOT}/{HERMES_PLUGIN.name}"
VM_TOOL_LIMITS_PLUGIN = f"{TASK_ROOT}/{TOOL_LIMITS_PLUGIN.name}"


def _resolve_runtime_cache() -> Path:
    return assets_dir() / "opencode_runtime.tar.gz"


def install_opencode_on_vm(controller, client_password="password"):
    """Never reuse a random system OpenCode or download at evaluation time."""
    cache = _resolve_runtime_cache()
    try:
        validate_runtime_cache(cache)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return False
    result = _execute_shell_on_vm(controller, f"{VM_BINARY} --version", timeout=30)
    if (
        result
        and result.get("returncode") == 0
        and result.get("output", "").strip() == OPENCODE_VERSION
    ):
        return True
    if not upload_file_to_vm(controller, cache, VM_RUNTIME_TARBALL):
        logger.error("OpenCode runtime upload to %s failed", VM_RUNTIME_TARBALL)
        return False
    result = _execute_shell_on_vm(
        controller,
        _sudo_cmd(f"tar xzf {VM_RUNTIME_TARBALL} -C /", client_password)
        + f" && rm -f {VM_RUNTIME_TARBALL} && {VM_BINARY} --version",
        timeout=120,
    )
    if (
        result
        and result.get("returncode") == 0
        and result.get("output", "").strip() == OPENCODE_VERSION
    ):
        return True
    logger.error("OpenCode runtime extraction/verification failed: %s", result)
    return False


def _write(path: str, value: bytes) -> str:
    data = base64.b64encode(value).decode("ascii")
    return f"printf %s {shlex.quote(data)} | base64 -d > {shlex.quote(path)}"


def vm_config_env(model: str) -> dict[str, str]:
    """Select the VM-reachable deployment without changing the host Builder.

    SHADOW_EVOL_VM_OPENAI_BASE_URL is the existing shared deployment contract,
    not a Prime sampling/context setting. AWS VMs may need a VPC address while
    the orchestrator uses the same service's public address.
    """
    env = dict(os.environ)
    base = env.get("SHADOW_OPENCODE_VM_BASE_URL", "").strip()
    if (
        not base
        and model.startswith("openai/")
        and not env.get("SHADOW_OPENCODE_BASE_URL", "").strip()
    ):
        base = env.get("SHADOW_EVOL_VM_OPENAI_BASE_URL", "").strip()
    if base:
        env["SHADOW_OPENCODE_BASE_URL"] = base
    return env


def build_opencode_vm_command(
    instruction,
    system_prompt,
    model,
    timeout,
    max_turns=200,
    api_key=None,
    parasail_api_key=None,
    *,
    max_steps=None,
    no_time_limit=False,
    continue_session=False,
    context_compaction=None,
):
    config = build_config(
        model,
        system_prompt,
        context_compaction,
        max_steps=max_steps if max_steps is not None else max_turns,
        env=vm_config_env(model),
        hermes_plugin_path=VM_HERMES_PLUGIN,
        tool_limits_plugin_path=VM_TOOL_LIMITS_PLUGIN,
    )
    settings = runner_settings()
    secret_env = {}
    key = {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "parasail": "PARASAIL_API_KEY",
    }[model.split("/", 1)[0]]
    value = {"OPENROUTER_API_KEY": api_key, "PARASAIL_API_KEY": parasail_api_key}.get(
        key
    ) or os.environ.get(key)
    if value:
        secret_env[key] = value
    steps = max_steps if max_steps is not None else max_turns
    argv = [
        "python3",
        f"{TASK_ROOT}/opencode_runner.py",
        "--binary",
        VM_BINARY,
        "--config",
        f"{TASK_ROOT}/opencode.json",
        "--settings",
        f"{TASK_ROOT}/runner.json",
        "--task-file",
        f"{TASK_ROOT}/instruction.txt",
        "--state-dir",
        STATE_DIR,
        "--cwd",
        "/home/user",
        "--stderr-file",
        f"{TASK_ROOT}/stderr.log",
        "--exit-file",
        f"{TASK_ROOT}/exit_code.txt",
    ]
    if steps is not None:
        argv.extend(["--max-steps", str(steps)])
    if not no_time_limit:
        if timeout is None or timeout <= 0:
            raise ValueError("timeout must be positive unless no_time_limit is set")
        argv.extend(["--timeout", str(timeout)])
    if continue_session:
        argv.append("--resume")
    setup = [
        "umask 077",
        "if [ -f /tmp/.shadow-agent-env ]; then . /tmp/.shadow-agent-env; rm -f /tmp/.shadow-agent-env; fi",
        "export DISPLAY=:0 HOME=/home/user",
        f"export SHADOW_GROUNDING_MODEL={shlex.quote(model)}",
        f"mkdir -p {TASK_ROOT}",
    ]
    if not continue_session:
        setup.append(f"rm -rf {STATE_DIR}")
    setup.extend(
        [
            f'trap \'status=$?; printf "%s\\n" "$status" > {TASK_ROOT}/exit_code.txt\' EXIT',
            _write(f"{TASK_ROOT}/instruction.txt", instruction.encode()),
            _write(f"{TASK_ROOT}/opencode.json", json.dumps(config).encode()),
            _write(f"{TASK_ROOT}/runner.json", json.dumps(settings).encode()),
            _write(f"{TASK_ROOT}/opencode_runner.py", RUNNER.read_bytes()),
            _write(VM_HERMES_PLUGIN, HERMES_PLUGIN.read_bytes()),
            _write(VM_TOOL_LIMITS_PLUGIN, TOOL_LIMITS_PLUGIN.read_bytes()),
            shlex.join(argv)
            + f" > {TASK_ROOT}/trajectory.jsonl 2> {TASK_ROOT}/stderr.log",
        ]
    )
    return AgentCommand(
        script=" && ".join(setup),
        secret_env=secret_env,
        accepted_exit_codes=(0, 124),
        activity_path=f"{TASK_ROOT}/trajectory.jsonl",
        idle_timeout=settings["idle_timeout"] + 60,
        process_pattern="[o]pencode_runner.py",
        runner_exit_grace=30,
    )


def validate_opencode_artifacts(artifacts: dict) -> str | None:
    raw = artifacts.get("claude_trajectory") or b""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    records = []
    for line in raw.split("\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            return "OpenCode emitted malformed JSONL"
        if not isinstance(event, dict):
            return "OpenCode emitted a non-object event"
        if event.get("type") == "error":
            return "OpenCode reported a provider/session error"
        records.append(event)
    if not records:
        return "OpenCode produced no trajectory"
    terminal = records[-1]
    if terminal.get("type") != "result" or terminal.get("is_error") is not False:
        return "OpenCode did not produce a successful terminal result"
    if terminal.get("subtype") not in {
        "completed",
        "step_limit",
        "wall_timeout",
        "length_limit",
    }:
        return "OpenCode ended without completion or an explicit task budget limit"
    if not terminal.get("session_id") or not terminal.get("steps", 0):
        return "OpenCode produced no completed model steps"
    exit_code = artifacts.get("exit_code", b"")
    if isinstance(exit_code, bytes):
        exit_code = exit_code.decode()
    expected = 124 if terminal["subtype"] == "wall_timeout" else 0
    if str(exit_code).strip() != str(expected) or terminal.get("exit_code") != expected:
        return "OpenCode exit marker disagrees with terminal result"
    return None


MODEL_CONFIG = {
    "opencode": {
        "binary": "opencode",
        "model_flag": None,
        "installer": install_opencode_on_vm,
        # Some V2 tasks (048) revoke sudo during setup; install from the clean
        # snapshot first, as Prime does. The post-setup call is then a no-op probe.
        "install_before_task_setup": True,
        "build_command": build_opencode_vm_command,
        "validate_artifacts": validate_opencode_artifacts,
        "accepted_exit_codes": (0, 124),
    }
}
