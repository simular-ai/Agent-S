"""Host-side builder harness backed by the official Prime Agent CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from agent_s4.harness.base import BuilderHarness, BuilderResult
from agent_s4.harness.builder_common import (
    build_subprocess_env,
    parse_builder_log,
    run_builder_subprocess,
)
from agent_s4.paths import PACKAGE_ROOT
from agent_s4.harness.prime.source import PRIME_AGENT_VERSION
from agent_s4.harness.prime.context_window import interface_window_is_honoured
from agent_s4.harness.prime.temperature import (
    EXTENSION_FILENAME,
    sampling_temperature,
    temperature_extension_source,
)
from agent_s4.harness.prime.compaction import (
    compaction_settings,
    context_window_from_config,
    load_compaction_config,
)
from agent_s4.harness.prime.openrouter import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    OPENAI_COMPATIBLE_PROVIDER,
    openai_compatible_models_config,
    openrouter_routing,
)

RUNNER = PACKAGE_ROOT / "harness" / "prime" / "runner.py"

BUILDER_SYSTEM_PROMPT = """\
You are the autonomous builder inside a shadow-interface evolution run.
Follow the task exactly, inspect the available evaluation evidence, make the
requested changes directly in the allowed files, verify them, and finish
without asking the user questions.

Your only model-facing tool is a persistent IPython kernel. Run shell commands
from IPython subprocess APIs or a %%bash cell. To inspect the GUI visually,
first use a subprocess to run the shadow interface screenshot command, for
example:

    subprocess.run(["python3", "-m", "shadow_interface", "call",
                    "gui.take_screenshot", "/tmp/screenshot.png"], check=True)

Then, in a subsequent IPython cell, load the image into model context with:

    print(await attach_image("/tmp/screenshot.png"))

Do not Base64-encode screenshots or infer their visual contents from file
metadata. Prime Agent owns the agent loop, session history, and configured
context compaction; continue working through compaction when it occurs.

The .builder_tmp directory contains your live runtime, daemon registry,
sessions, and configuration. Never delete, rename, or clean it or its contents.
For disposable validation files, create your own subdirectory under /tmp and
clean only that directory. Leave runtime cleanup to the orchestrator.
"""


def resolve_prime_agent_binary() -> Path:
    """Resolve the explicitly pinned binary or ``prime-agent`` on PATH."""
    configured = os.environ.get("SHADOW_PRIME_AGENT_BIN")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise RuntimeError(f"SHADOW_PRIME_AGENT_BIN does not name a file: {path}")
        resolved = path.resolve()
        validate_prime_agent_binary(resolved)
        return resolved
    resolved = shutil.which("prime-agent")
    if resolved is None:
        raise RuntimeError(
            "prime-agent is not installed or is not on PATH; install the "
            "official CLI or set SHADOW_PRIME_AGENT_BIN"
        )
    resolved_path = Path(resolved).resolve()
    validate_prime_agent_binary(resolved_path)
    return resolved_path


def validate_prime_agent_binary(binary: Path) -> None:
    """Require the host builder CLI to match the vendored/runtime pin."""
    try:
        version = subprocess.check_output(
            [str(binary), "--version"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"could not verify Prime Agent binary {binary}: {exc}"
        ) from exc
    if version != PRIME_AGENT_VERSION:
        raise RuntimeError(
            "Prime Agent host binary does not match the vendored integration: "
            f"expected {PRIME_AGENT_VERSION}, found {version or '<empty>'}"
        )


def prime_agent_runtime_root(binary: Path) -> Path:
    """Return the CLI package root that Landlock must make readable.

    Release/source JS launchers commonly resolve to
    ``<package>/dist/bundle/cli.js``. Granting only ``cli.js``'s directory
    hides sibling bundled skills and ``prime-agent-runtime``. A standalone
    native release has no ``dist`` ancestor and needs only its containing
    directory.
    """
    resolved = Path(binary).resolve()
    for parent in resolved.parents:
        if parent.name == "dist":
            return parent.parent
    # The pinned offline runtime uses <package>/bin/prime-agent as a shell
    # launcher, which execs sibling <package>/node/bin/node and
    # <package>/node_modules/.bin/prime-agent.  Granting only bin/ lets the
    # launcher start under Landlock but makes its exec fail with EACCES (126).
    if resolved.parent.name == "bin":
        package = resolved.parent.parent
        if (package / "node_modules").is_dir() and (package / "node").is_dir():
            return package
    return resolved.parent


def bundled_kernel_python(runtime_root: Path) -> Path | None:
    """Return Prime's pre-provisioned kernel Python when the runtime has one."""
    candidate = Path(runtime_root) / "python" / "bin" / "python3.11"
    return candidate.resolve() if candidate.is_file() else None


def context_window_tokens(clone_dir: Path) -> int | None:
    """Read the interface's explicit Prime context-window override.

    A missing file, a disabled policy, or an absent field all mean "no
    override", so Prime keeps the window its own model catalog advertises.
    Only a present-but-invalid value is an error, because that is a mutation
    that wrote something it should not have. A run pinned to Prime's catalog
    window ignores the field entirely; see
    agent_s4.harness.prime.context_window.
    """
    if not interface_window_is_honoured():
        return None
    try:
        return context_window_from_config(
            load_compaction_config(Path(clone_dir) / "shadow_interface")
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


class PrimeAgentBuilder(BuilderHarness):
    """Run Prime Agent headlessly while retaining its native session engine."""

    def launch(
        self,
        *,
        clone_dir: Path,
        prompt: str,
        model: str,
        timeout: int | None,
        env_extra: dict[str, str] | None,
        variant_id: str,
        logs_dir: Path,
        log_name: str,
        builder_bin: str | None = None,
    ) -> BuilderResult:
        # builder_bin remains the legacy Claude-only pin.
        del builder_bin
        if timeout is not None and timeout < 90:
            raise ValueError("prime-agent builder timeout must be at least 90 seconds")
        prime_binary = resolve_prime_agent_binary()
        runtime_root = prime_agent_runtime_root(prime_binary)
        kernel_python = bundled_kernel_python(runtime_root)
        clone_dir = Path(clone_dir)
        sandbox_tmp = clone_dir / ".builder_tmp"
        sandbox_tmp.mkdir(mode=0o700, exist_ok=True)
        runner = sandbox_tmp / "prime_agent_runner.py"
        shutil.copy2(RUNNER, runner)
        session_dir = sandbox_tmp / "prime-sessions"
        agent_dir = sandbox_tmp / "prime-home"
        agent_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        context_compaction = load_compaction_config(clone_dir / "shadow_interface")
        model_context_window = context_window_from_config(context_compaction)
        max_output_tokens = None
        # The Builder writes the interface that the evaluator will run under,
        # so both must reach the same deployment through the same declared
        # capabilities. Prime knows nothing about a self-hosted
        # OpenAI-compatible endpoint, so declare it here exactly as the
        # evaluator does.
        prime_model = model
        openai_base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        # Without a base URL this is a real OpenAI model and Prime's own
        # provider handles it; only a self-hosted deployment needs declaring.
        if model.startswith("openai/") and openai_base_url:
            model_id = model.removeprefix("openai/")
            prime_model = f"{OPENAI_COMPATIBLE_PROVIDER}/{model_id}"
            models_config = openai_compatible_models_config(
                model_id,
                openai_base_url,
                model_context_window,
            )
            declared_model = json.loads(models_config)["providers"][
                OPENAI_COMPATIBLE_PROVIDER
            ]["models"][0]
            model_context_window = declared_model["contextWindow"]
            max_output_tokens = min(
                declared_model["maxTokens"], DEFAULT_MAX_OUTPUT_TOKENS
            )
            (agent_dir / "models.json").write_text(
                models_config + "\n",
                encoding="utf-8",
            )
        elif model.startswith("openrouter/"):
            max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
            provider_config = {
                "compat": {
                    "openRouterRouting": openrouter_routing(),
                },
            }
            # The Builder writes the interface that the evaluator will run
            # under, so it has to reason inside the same window.
            if model_context_window is not None:
                provider_config["modelOverrides"] = {
                    model.removeprefix("openrouter/"): {
                        "contextWindow": model_context_window,
                    },
                }
            (agent_dir / "models.json").write_text(
                json.dumps(
                    {
                        "providers": {
                            "openrouter": provider_config,
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        (agent_dir / "settings.json").write_text(
            json.dumps(
                {
                    "compaction": compaction_settings(
                        context_compaction,
                        context_window=model_context_window,
                        max_output_tokens=max_output_tokens,
                    ),
                    "autoRefine": {"enabled": False},
                    "telemetry": {"enabled": False},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        argv = [
            sys.executable,
            str(runner),
            "--prime-bin",
            str(prime_binary),
            "--task",
            prompt,
            "--system-prompt",
            BUILDER_SYSTEM_PROMPT,
            "--model",
            prime_model,
            "--cwd",
            str(clone_dir),
            "--session-dir",
            str(session_dir),
        ]
        temperature = sampling_temperature()
        if temperature is not None:
            extension = agent_dir / EXTENSION_FILENAME
            extension.write_text(
                temperature_extension_source(temperature),
                encoding="utf-8",
            )
            argv.extend(("--extension", str(extension)))
        env = build_subprocess_env(env_extra, strip_keys=("CLAUDECODE",))
        # Isolate all Prime configuration and session writes inside the clone.
        # Set both current and legacy upstream names for version compatibility.
        env.update(
            {
                "HOME": str(agent_dir),
                "PRIME_AGENT_CODING_AGENT_DIR": str(agent_dir),
                "PI_CODING_AGENT_DIR": str(agent_dir),
                "PRIME_AGENT_SESSION_DIR": str(session_dir),
                "PI_SESSION_DIR": str(session_dir),
            }
        )
        if kernel_python is not None:
            # The pinned offline package already carries ipykernel, rlm
            # (prime-agent-runtime), and the default analysis packages. Using
            # it avoids a first-use `uv python install` inside Landlock.
            env["PRIME_AGENT_KERNEL_PYTHON"] = str(kernel_python)
        result = run_builder_subprocess(
            argv,
            clone_dir=clone_dir,
            logs_dir=logs_dir,
            log_name=log_name,
            timeout=timeout,
            env=env,
            variant_id=variant_id,
            binary_hint=f"Prime Agent adapter via {sys.executable}",
            # The adapter is staged in the clone, but its child CLI and bundled
            # runtime must remain executable after the V2 Landlock boundary.
            sandbox_read_paths=[
                runtime_root,
                # Prime's persistent IPython tool launches uv from PATH. The
                # host install commonly lives under ~/.local/bin, outside the
                # Builder's default Landlock roots; grant only this executable.
                *(
                    [Path(uv_binary).resolve()]
                    if (uv_binary := shutil.which("uv"))
                    else []
                ),
            ],
        )
        _, saw_result, _ = parse_builder_log(result.log_path)
        result.crashed = result.crashed or result.exit_code != 0 or not saw_result
        return result
