"""OpenCode Builder. Uses the same configuration/supervisor as the VM path."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

from agent_s4.harness.base import BuilderHarness, BuilderResult
from agent_s4.harness.builder_common import (
    build_subprocess_env,
    run_builder_subprocess,
)
from agent_s4.harness.opencode.config import (
    HERMES_PLUGIN,
    TOOL_LIMITS_PLUGIN,
    build_config,
    load_context_config,
    runner_settings,
)
from agent_s4.harness.opencode.source import resolve_binary
from agent_s4.paths import PACKAGE_ROOT

RUNNER = PACKAGE_ROOT / "harness" / "opencode" / "runner.py"
BUILDER_SYSTEM_PROMPT = """You are the autonomous builder in a shadow-interface evolution run.
Inspect the permitted evidence, make the requested changes in the allowed
workspace, verify them, and finish without asking questions. Use your native
tools to run commands, edit files, and read screenshots. Keep working through
context compaction. Stop applications by their exact PID, not broad process
name patterns. Never terminate the agent, its supervisor, or unrelated runs.

The .builder_tmp directory contains your live runtime, sessions, and
configuration. Never delete, rename, or clean it or its contents. For
disposable validation files, create your own subdirectory under /tmp and clean
only that directory. Leave runtime cleanup to the orchestrator.
"""


class OpenCodeBuilder(BuilderHarness):
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
        del builder_bin  # legacy Claude-only argument
        binary = resolve_binary()
        clone_dir = Path(clone_dir).resolve()
        temporary = clone_dir / ".builder_tmp"
        temporary.mkdir(mode=0o700, exist_ok=True)
        runtime = temporary / "opencode"
        # Each mutation is a new session. Only evaluator follow-ups resume.
        if runtime.exists():
            shutil.rmtree(runtime)
        runtime.mkdir(mode=0o700)
        runner = temporary / "opencode_runner.py"
        shutil.copy2(RUNNER, runner)
        source_env = {**os.environ, **(env_extra or {})}
        max_steps = int(source_env.get("SHADOW_OPENCODE_BUILDER_MAX_STEPS", "200"))
        plugin = runtime / HERMES_PLUGIN.name
        shutil.copy2(HERMES_PLUGIN, plugin)
        limits = runtime / TOOL_LIMITS_PLUGIN.name
        shutil.copy2(TOOL_LIMITS_PLUGIN, limits)
        config = build_config(
            model,
            BUILDER_SYSTEM_PROMPT,
            load_context_config(clone_dir / "shadow_interface"),
            max_steps=max_steps,
            env=source_env,
            hermes_plugin_path=plugin,
            tool_limits_plugin_path=limits,
        )
        settings = runner_settings(source_env)
        config_path, settings_path = runtime / "opencode.json", runtime / "runner.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        settings_path.write_text(json.dumps(settings) + "\n")
        task_file = runtime / "instruction.txt"
        task_file.write_text(prompt, encoding="utf-8")
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        # Preserve the effective config outside .builder_tmp's lifecycle.
        (logs_dir / f"{log_name}.opencode-config.json").write_text(
            config_path.read_text()
        )
        (logs_dir / f"{log_name}.opencode-runner.json").write_text(
            settings_path.read_text()
        )
        argv = [
            sys.executable,
            str(runner),
            "--binary",
            str(binary),
            "--config",
            str(config_path),
            "--settings",
            str(settings_path),
            "--task-file",
            str(task_file),
            "--state-dir",
            str(runtime / "state"),
            "--cwd",
            str(clone_dir),
            "--stderr-file",
            str(runtime / "stderr.log"),
            "--max-steps",
            str(max_steps),
        ]
        if timeout is not None:
            if timeout <= 0:
                raise ValueError("OpenCode builder timeout must be positive")
            argv.extend(["--timeout", str(timeout)])
        env = build_subprocess_env(env_extra, strip_keys=("CLAUDECODE",))
        # PARASAIL_API_KEY is evaluator-only in the generic helper; expose it
        # only when it is this Builder's explicitly selected model provider.
        if model.startswith("parasail/") and source_env.get("PARASAIL_API_KEY"):
            env["PARASAIL_API_KEY"] = source_env["PARASAIL_API_KEY"]
        home = runtime / "home"
        home.mkdir()
        env["HOME"] = str(home)
        result = run_builder_subprocess(
            argv,
            clone_dir=clone_dir,
            logs_dir=logs_dir,
            log_name=log_name,
            timeout=None if timeout is None else timeout + 15,
            env=env,
            variant_id=variant_id,
            binary_hint="pinned OpenCode",
            sandbox_read_paths=[binary.parent],
        )
        result.crashed = result.crashed or result.exit_code != 0
        return result
