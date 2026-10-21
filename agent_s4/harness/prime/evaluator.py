"""Prime Agent runtime installation and headless OSWorld VM command."""

from __future__ import annotations

import base64
import json
import logging
import os
import shlex
from pathlib import Path

from agent_s4.paths import PACKAGE_ROOT, SHARED_ROOT
from agent_s4.harness.prime.source import (
    KERNEL_REQUIRED_IMPORTS,
    KERNEL_SKILL_IMPORTS,
    PRIME_AGENT_VERSION,
    runtime_cache_fingerprint,
    validate_prime_agent_runtime_cache,
    validate_prime_agent_source,
)
from agent_s4.harness.prime.openrouter import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    OPENAI_COMPATIBLE_PROVIDER,
    openai_compatible_models_config,
    openrouter_routing,
)
from agent_s4.harness.prime.compaction import compaction_settings
from agent_s4.harness.prime.temperature import (
    EXTENSION_FILENAME,
    sampling_temperature,
    temperature_extension_source,
)
from agent_s4.vm.utils import (
    AgentCommand,
    _execute_shell_on_vm,
    _sudo_cmd,
    upload_file_to_vm,
)

logger = logging.getLogger("agent_s4.harness.prime.evaluator")
RUNTIME_ROOT = "/opt/prime-agent"
RUNTIME_BINARY = f"{RUNTIME_ROOT}/bin/prime-agent"
RUNTIME_PYTHON = f"{RUNTIME_ROOT}/python/bin/python3"
OVERFLOW_MODULE = (
    f"{RUNTIME_ROOT}/node_modules/@earendil-works/pi-ai/dist/utils/overflow.js"
)
IPYTHON_MODULE = f"{RUNTIME_ROOT}/node_modules/prime-agent/dist/core/tools/ipython.js"
IPYTHON_DIST_ROOT = f"{RUNTIME_ROOT}/node_modules/prime-agent/dist"
IPYTHON_BUNDLE_ROOT = f"{IPYTHON_DIST_ROOT}/bundle"
VM_RUNTIME_TARBALL = "/tmp/prime_agent_runtime.tar.gz"
TASK_ROOT = "/tmp/claude_task"
AGENT_CONFIG_DIR = f"{TASK_ROOT}/prime-config"
SESSION_DIR = f"{TASK_ROOT}/prime-sessions"
STEP_RUNNER = f"{TASK_ROOT}/prime_step_runner.py"
TEMPERATURE_EXTENSION = f"{TASK_ROOT}/{EXTENSION_FILENAME}"
PRIME_SYSTEM_ADDENDUM = """

## Prime Agent OSWorld runtime

Your model-facing tool is a persistent IPython kernel. Use Python subprocess
APIs (or a %%bash cell) for shell commands. For visual inspection, first run
the shadow-interface screenshot command from a subprocess, then in a later
IPython cell call `await attach_image(<absolute screenshot path>)`. Do not infer
screen contents from metadata or Base64. If context compaction occurs, keep
working afterward and rely on the persistent kernel state.
"""
PRIME_THINKING_LEVELS = {
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
}


# Seconds of tool/startup silence before the in-VM step runner ends an episode.
# The host liveness check waits RUNNER_IDLE_GRACE longer so the runner's own,
# better-labelled timeout always fires first.
DEFAULT_IDLE_TIMEOUT = 360
RUNNER_IDLE_GRACE = 60


def _idle_timeout() -> int:
    raw = os.environ.get("SHADOW_PRIME_IDLE_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_IDLE_TIMEOUT
    value = int(raw)
    if value <= 0:
        raise ValueError(f"SHADOW_PRIME_IDLE_TIMEOUT must be positive, got {raw!r}")
    return value


def _model_request_timeout() -> int:
    """Per-request model timeout in seconds; 0 leaves the idle watchdog alone."""
    raw = os.environ.get("SHADOW_PRIME_MODEL_REQUEST_TIMEOUT", "").strip()
    if not raw:
        return 0
    value = int(raw)
    if value < 0:
        raise ValueError(
            f"SHADOW_PRIME_MODEL_REQUEST_TIMEOUT must be >= 0, got {raw!r}"
        )
    return value


def _project_root() -> Path:
    """Root of the shared assets (vm_assets/), common to every benchmark."""
    return SHARED_ROOT


def _step_runner_source() -> Path:
    return PACKAGE_ROOT / "harness" / "prime" / "step_runner.py"


def _overflow_patch_source() -> Path:
    return PACKAGE_ROOT / "harness" / "prime" / "overflow_patch.py"


def _overflow_patch_command() -> str:
    source_b64 = base64.b64encode(_overflow_patch_source().read_bytes()).decode()
    patcher = "/tmp/prime_overflow_patch.py"
    return (
        f"echo {shlex.quote(source_b64)} | base64 -d > {patcher} && "
        f"python3 {patcher} {OVERFLOW_MODULE} && rm -f {patcher} && "
        f"grep -Fq {shlex.quote('/decoder prompt.*longer than the maximum model length/i')} "
        f"{OVERFLOW_MODULE}"
    )


def _ipython_timeout_patch_command() -> str:
    source = PACKAGE_ROOT / "harness" / "prime" / "ipython_timeout_patch.py"
    source_b64 = base64.b64encode(source.read_bytes()).decode()
    patcher = "/tmp/prime_ipython_timeout_patch.py"
    return (
        f"echo {shlex.quote(source_b64)} | base64 -d > {patcher} && "
        f"python3 {patcher} {IPYTHON_DIST_ROOT} && rm -f {patcher} && "
        f"grep -Fq {shlex.quote('IPython cell exceeded 60 seconds')} "
        f"{IPYTHON_MODULE} && "
        f"grep -RFq {shlex.quote('IPython cell exceeded 60 seconds')} "
        f"{IPYTHON_BUNDLE_ROOT}"
    )


def _resolve_runtime_cache() -> Path:
    override = os.environ.get("SHADOW_VM_ASSETS_DIR")
    if override:
        return Path(override) / "prime_agent_runtime.tar.gz"
    return _project_root() / "vm_assets" / "prime_agent_runtime.tar.gz"


def _runtime_probe(expected_commit: str) -> str:
    fingerprint = runtime_cache_fingerprint(expected_commit)
    imports = "; ".join(
        f"import {name}"
        for name in ("rlm", *KERNEL_REQUIRED_IMPORTS, *KERNEL_SKILL_IMPORTS)
    )
    return " && ".join(
        (
            f"test -x {RUNTIME_BINARY}",
            f"test -x {RUNTIME_PYTHON}",
            f"test $(cat {RUNTIME_ROOT}/upstream_commit) = {shlex.quote(expected_commit)}",
            f"test $(cat {RUNTIME_ROOT}/runtime_fingerprint) = {shlex.quote(fingerprint)}",
            # The pinned official CLI writes its version banner to stderr.
            f"{RUNTIME_BINARY} --version 2>&1 | grep -Fqx {shlex.quote(PRIME_AGENT_VERSION)}",
            f"{RUNTIME_PYTHON} -c {shlex.quote(imports)}",
        )
    )


def _openrouter_routing(or_provider) -> dict:
    """Compatibility wrapper around the shared Prime routing policy."""
    return openrouter_routing(or_provider)


def install_prime_agent_on_vm(controller, client_password="password") -> bool:
    """Install only the verified offline archive; never fetch from the VM."""
    expected_commit = validate_prime_agent_source()
    probe = _execute_shell_on_vm(
        controller, _runtime_probe(expected_commit), timeout=30
    )
    install_prefix = ""
    if not probe or probe.get("returncode") != 0:
        cache = _resolve_runtime_cache()
        try:
            validate_prime_agent_runtime_cache(cache)
        except RuntimeError as exc:
            logger.error("Prime Agent runtime cache validation failed: %s", exc)
            return False
        if not upload_file_to_vm(controller, cache, VM_RUNTIME_TARBALL):
            return False
        install_prefix = (
            _sudo_cmd(f"rm -rf {RUNTIME_ROOT}", client_password)
            + " && "
            + _sudo_cmd(f"tar xzf {VM_RUNTIME_TARBALL} -C /", client_password)
            + f" && rm -f {VM_RUNTIME_TARBALL} && "
        )
    # Apply this even to a cached runtime. It is idempotent, and keeping the
    # overlay outside the pinned archive preserves upstream provenance.
    command = (
        install_prefix
        + _overflow_patch_command()
        + " && "
        + _ipython_timeout_patch_command()
        + " && "
        + _runtime_probe(expected_commit)
    )
    result = _execute_shell_on_vm(controller, command, timeout=240)
    if result and result.get("returncode") == 0:
        return True
    logger.error("Prime Agent runtime verification failed: %s", result)
    return False


def build_prime_agent_vm_command(
    instruction,
    system_prompt,
    model,
    timeout,
    api_key=None,
    openrouter_api_key=None,
    parasail_api_key=None,
    or_provider=None,
    continue_session=False,
    max_steps=None,
    no_time_limit=False,
    context_window_tokens=None,
    context_compaction=None,
    **_unused,
):
    """Build Prime Agent's official JSON/headless CLI invocation.

    Compaction uses the interface policy in a task-local config directory;
    sessions remain enabled so Prime can preserve compacted history.
    """
    if timeout < 90:
        raise ValueError("Prime Agent evaluator timeout must be at least 90 seconds")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if context_window_tokens is not None and (
        isinstance(context_window_tokens, bool)
        or not isinstance(context_window_tokens, int)
        or context_window_tokens <= 0
    ):
        raise ValueError("context_window_tokens must be a positive integer")
    # Task text reaches Prime through files, never argv: an agent that runs
    # `pkill -f <app>` for an app named in its task would otherwise match (and
    # kill) its own runtime, whose command line carried the whole task. The
    # trailing-newline strip keeps the bytes identical to the old "$(cat ...)".
    instruction_b64 = base64.b64encode(instruction.rstrip("\n").encode()).decode()
    system_prompt_b64 = base64.b64encode(
        (system_prompt.rstrip() + PRIME_SYSTEM_ADDENDUM).rstrip("\n").encode()
    ).decode()
    thinking_level = (
        os.environ.get("SHADOW_PRIME_AGENT_THINKING_LEVEL", "medium").strip().lower()
    )
    if thinking_level not in PRIME_THINKING_LEVELS:
        raise ValueError(
            "SHADOW_PRIME_AGENT_THINKING_LEVEL must be one of: "
            + ", ".join(sorted(PRIME_THINKING_LEVELS))
        )
    prime_model = model
    models_config = None
    effective_context_window = context_window_tokens
    max_output_tokens = None
    # The model endpoint has to be named at an address the TASK VM can route.
    # When that differs from the orchestrator's own view of the same service --
    # a proxy inside the benchmark VPC, reachable from the host only by its
    # public address and from the VMs only by its private one -- the VM's view
    # wins here, because this config and this env are consumed inside the VM.
    openai_base_url = (
        os.environ.get("SHADOW_EVOL_VM_OPENAI_BASE_URL", "").strip()
        or os.environ.get("OPENAI_BASE_URL", "").strip()
    )
    if model.startswith("openrouter/"):
        # pi-ai's pinned OpenAI-completions adapter requests min(catalog cap,
        # 32000), so reserving 32000 is safe even without reading the catalog.
        max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
        model_id = model.removeprefix("openrouter/")
        provider_config = {
            "compat": {
                "openRouterRouting": _openrouter_routing(or_provider),
            },
        }
        if context_window_tokens is not None:
            provider_config["modelOverrides"] = {
                model_id: {"contextWindow": context_window_tokens},
            }
        models_config = json.dumps(
            {
                "providers": {
                    "openrouter": provider_config,
                },
            },
            separators=(",", ":"),
        )
    elif model.startswith("openai/") and openai_base_url:
        model_id = model.removeprefix("openai/")
        prime_model = f"{OPENAI_COMPATIBLE_PROVIDER}/{model_id}"
        models_config = openai_compatible_models_config(
            model_id,
            openai_base_url,
            context_window_tokens,
        )
        declared_model = json.loads(models_config)["providers"][
            OPENAI_COMPATIBLE_PROVIDER
        ]["models"][0]
        effective_context_window = declared_model["contextWindow"]
        max_output_tokens = min(declared_model["maxTokens"], DEFAULT_MAX_OUTPUT_TOKENS)
    elif context_window_tokens is not None and "/" in model:
        provider, model_id = model.split("/", 1)
        models_config = json.dumps(
            {
                "providers": {
                    provider: {
                        "modelOverrides": {
                            model_id: {"contextWindow": context_window_tokens},
                        },
                    },
                },
            },
            separators=(",", ":"),
        )
    settings = json.dumps(
        {
            "compaction": compaction_settings(
                context_compaction,
                context_window=effective_context_window,
                max_output_tokens=max_output_tokens,
            ),
            # Shadow Evolution is the outer evolutionary loop. Prime's separate
            # opportunistic 25-turn auto-refine caused silent turn-boundary model
            # stalls in OSWorld trajectories and must not block task evaluation.
            "autoRefine": {"enabled": False},
            "defaultThinkingLevel": thinking_level,
            "telemetry": {"enabled": False},
        },
        separators=(",", ":"),
    )
    secret_env = {}
    if api_key:
        secret_env["ANTHROPIC_API_KEY"] = api_key
    if openrouter_api_key:
        secret_env["OPENROUTER_API_KEY"] = openrouter_api_key
    if parasail_api_key:
        secret_env["PARASAIL_API_KEY"] = parasail_api_key
    # Preserve explicitly configured provider environments without copying
    # credentials into the task-local settings file or command arguments.
    preserved_env_names = [
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ]
    if model.startswith("openrouter/"):
        preserved_env_names.append("OPENROUTER_API_KEY")
    for name in preserved_env_names:
        value = os.environ.get(name, "")
        if name == "OPENAI_BASE_URL" and openai_base_url:
            value = openai_base_url
        if value and name not in secret_env:
            secret_env[name] = value
    command_parts = [
        RUNTIME_BINARY,
        "--mode",
        "json",
        "--model",
        shlex.quote(prime_model),
        "--cwd",
        "/home/user",
        "--session-dir",
        SESSION_DIR,
        "--no-context-files",
        "--offline",
        # Prime reads an existing path here as the prompt file's contents.
        "--append-system-prompt",
        f"{TASK_ROOT}/system_prompt.txt",
    ]
    temperature = sampling_temperature()
    temperature_setup = ""
    if temperature is not None:
        extension_b64 = base64.b64encode(
            temperature_extension_source(temperature).encode()
        ).decode()
        temperature_setup = f"echo {shlex.quote(extension_b64)} | base64 -d > {TEMPERATURE_EXTENSION} && "
        command_parts.extend(("--extension", TEMPERATURE_EXTENSION))
    if continue_session:
        command_parts.append("--continue")
    # The first message comes from piped stdin (see the redirect below).
    command = " ".join(command_parts)
    # A buffering endpoint (the LiteLLM Hermes proxy replays SSE only after the
    # whole upstream completion, and permits 1800s per request) is silent
    # during a model turn, which is not tool inactivity. Keep the short
    # tool/startup watchdog and, when configured, use the provider's
    # per-request limit for an in-flight model turn instead.
    model_request_timeout = _model_request_timeout() if max_steps is not None else 0
    idle_timeout = _idle_timeout()
    step_runner_setup = ""
    if max_steps is not None:
        runner_b64 = base64.b64encode(_step_runner_source().read_bytes()).decode()
        step_runner_setup = (
            f"echo {shlex.quote(runner_b64)} | base64 -d > {STEP_RUNNER} && "
        )
        command = (
            f"{RUNTIME_PYTHON} {STEP_RUNNER} --step-limit {int(max_steps)} "
            f"--idle-timeout {idle_timeout} "
            + (
                f"--model-request-timeout {model_request_timeout} --state-path {TASK_ROOT}/runner_state.json "
                if model_request_timeout
                else ""
            )
            + "-- "
            + command
        )
    credential_prefix = (
        "set -a && . /tmp/.shadow-agent-env && rm -f /tmp/.shadow-agent-env && set +a && "
        if secret_env
        else ""
    )
    if continue_session:
        task_setup = f"mkdir -p {AGENT_CONFIG_DIR} {SESSION_DIR} && "
    else:
        # The official CLI keeps a detached daemon after a JSON episode. A
        # fresh phase used to delete that daemon's live config directory first,
        # so its socket closed as soon as the next client connected. Shut it
        # down through the supported CLI command before replacing local state.
        task_setup = (
            f"{RUNTIME_BINARY} shutdown --force >/dev/null 2>&1 || true; "
            f"rm -rf {TASK_ROOT} && mkdir -p {AGENT_CONFIG_DIR} {SESSION_DIR} && "
        )
    models_setup = (
        f"echo {shlex.quote(models_config)} > {AGENT_CONFIG_DIR}/models.json && "
        if models_config is not None
        else ""
    )
    script = (
        credential_prefix + f"export DISPLAY=:0 && export HOME=/home/user && "
        f"export SHADOW_GROUNDING_MODEL={shlex.quote(model)} && "
        f"export PRIME_AGENT_CODING_AGENT_DIR={AGENT_CONFIG_DIR} && "
        f"export PRIME_AGENT_KERNEL_PYTHON={RUNTIME_PYTHON} && "
        + task_setup
        + f"trap 'status=$?; cp {TASK_ROOT}/trajectory.jsonl "
        + f"{TASK_ROOT}/stdout.log 2>/dev/null || true; "
        + 'printf "%s\\n" "$status" > '
        + f"{TASK_ROOT}/exit_code.txt' EXIT && "
        + f"echo {shlex.quote(instruction_b64)} | base64 -d > {TASK_ROOT}/instruction.txt && "
        f"echo {shlex.quote(system_prompt_b64)} | base64 -d > {TASK_ROOT}/system_prompt.txt && "
        + step_runner_setup
        + f"echo {shlex.quote(settings)} > {AGENT_CONFIG_DIR}/settings.json && "
        + models_setup
        + temperature_setup
        + (command if no_time_limit else f"timeout {int(timeout)} {command}")
        + f" < {TASK_ROOT}/instruction.txt"
        + f" > {TASK_ROOT}/trajectory.jsonl "
        + f"2> {TASK_ROOT}/stderr.log"
    )
    # GNU timeout returns 124 when the task consumes its complete wall-clock
    # budget. That is a valid benchmark episode: OSWorld must score the desktop
    # state reached at the deadline. Prime's daemon writes "Daemon worker client
    # closed" while it is being terminated, which is not an infrastructure
    # failure. Other harnesses retain the default exit-code-0-only behavior.
    return AgentCommand(
        script=script,
        secret_env=secret_env,
        accepted_exit_codes=(0, 124),
        activity_path=f"{TASK_ROOT}/trajectory.jsonl",
        idle_timeout=idle_timeout + RUNNER_IDLE_GRACE,
        model_request_timeout=(
            model_request_timeout + 60 if model_request_timeout else None
        ),
        phase_path=f"{TASK_ROOT}/runner_state.json" if model_request_timeout else None,
        process_pattern="[p]rime_step_runner.py",
        successful_terminal_events=(
            "agent_end",
            "prime_step_limit",
            "prime_agent_salvaged",
        ),
        runner_exit_grace=60,
        salvage_min_steps=50,
        salvage_terminal_event="prime_agent_salvaged",
    )


def validate_prime_agent_artifacts(artifacts: dict) -> str | None:
    """Reject a Prime episode whose native JSONL is not a valid run."""
    raw = artifacts.get("claude_trajectory")
    if not raw:
        return "Prime Agent produced no trajectory"
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    exit_code = artifacts.get("exit_code", b"")
    if isinstance(exit_code, bytes):
        exit_code = exit_code.decode("utf-8", errors="replace")
    timed_out = exit_code.strip() == "124"
    # split("\n"), not splitlines(): Qwen emits raw U+0085 inside JSON strings
    # and splitlines() breaks the record there, faking a "malformed JSONL"
    # infra failure (linear if-001 s_009/s_023/s_025/014, if-011 s_005).
    lines = raw.split("\n")
    last_line_number = max(
        (n for n, line in enumerate(lines, 1) if line.strip()),
        default=0,
    )
    saw_start = saw_end = saw_step_limit = saw_salvaged = False
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Running out the wall clock is a valid, scoreable episode.
            # `timeout` kills Prime mid-write, so the final record can be cut
            # off; treating that as an infrastructure failure re-ran a
            # 2-hour train task from scratch (if-011 s_005, 2026-09-24).
            if timed_out and line_number == last_line_number:
                continue
            return f"Prime Agent emitted malformed JSONL at line {line_number}"
        if not isinstance(event, dict):
            return f"Prime Agent emitted a non-object event at line {line_number}"
        event_type = event.get("type")
        saw_start = saw_start or event_type == "agent_start"
        saw_end = saw_end or event_type == "agent_end"
        saw_step_limit = saw_step_limit or event_type == "prime_step_limit"
        saw_salvaged = saw_salvaged or event_type == "prime_agent_salvaged"
        if event_type in {"prime_agent_launch_error", "agent_error"}:
            return f"Prime Agent reported terminal event {event_type}"
        if event_type == "tool_execution_end" and event.get("isError"):
            content = json.dumps(event, ensure_ascii=False)
            if (
                "PRIME_AGENT_KERNEL_PYTHON points to a Python missing default Python packages"
                in content
            ):
                return "Prime Agent kernel bootstrap validation failed"
    if not saw_start or (
        not saw_end and not timed_out and not saw_step_limit and not saw_salvaged
    ):
        return "Prime Agent trajectory is missing agent_start or agent_end"
    return None


MODEL_CONFIG = {
    "prime_agent": {
        "binary": "prime-agent",
        "model_flag": None,
        "installer": install_prime_agent_on_vm,
        "install_before_task_setup": True,
        "build_command": build_prime_agent_vm_command,
        "validate_artifacts": validate_prime_agent_artifacts,
    }
}
