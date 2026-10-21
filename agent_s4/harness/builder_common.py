"""Shared subprocess + log-parsing helpers for builder harnesses.

The subprocess plumbing (env build, file-based stdout, timeout handling,
post-run log parse, crash detection, BuilderResult assembly) lives here so
the builder harness only owns its argv construction and auth env vars.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from agent_s4.harness.base import BuilderResult

logger = logging.getLogger("agent_s4.harness.builder_common")

_CRASH_ELAPSED_CEILING = 30.0
_EVALUATOR_ONLY_ENV_KEYS = {
    "GITLAB_URL",
    "GITLAB_PRIVATE_TOKEN",
    "WEBSITE_HOST_SUFFIX",
    "OSWORLD_FILE_BASE_URL",
    "PROXY_CONFIG_FILE",
    "PARASAIL_API_KEY",
}
_EVALUATOR_ONLY_ENV_PREFIXES = (
    "AWS_",
    "OSWORLD_EVAL_",
    "OSWORLD_USER_SIM_",
)


def build_subprocess_env(
    env_extra: dict | None,
    *,
    strip_keys: tuple[str, ...] = (),
) -> dict:
    """Copy os.environ minus strip_keys, merge env_extra on top."""

    def protected(key: str) -> bool:
        return key in _EVALUATOR_ONLY_ENV_KEYS or key.startswith(
            _EVALUATOR_ONLY_ENV_PREFIXES
        )

    env = {
        k: v for k, v in os.environ.items() if k not in strip_keys and not protected(k)
    }
    if env_extra:
        env.update({k: v for k, v in env_extra.items() if not protected(k)})
    return env


def parse_builder_log(log_path: Path) -> tuple[float, bool, bool]:
    """Read a stream-json log and return (cost, saw_result, saw_rate_limit_blocked)."""
    cost = 0.0
    saw_result = False
    saw_rate_limit_blocked = False
    with open(log_path) as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except (json.JSONDecodeError, TypeError):
                continue
            t = data.get("type", "")
            if t == "result":
                saw_result = True
                cost = float(data.get("total_cost_usd", 0) or 0)
            elif t == "rate_limit_event":
                info = data.get("rate_limit_info", {})
                if info.get("status") == "blocked":
                    saw_rate_limit_blocked = True
    return cost, saw_result, saw_rate_limit_blocked


def run_builder_subprocess(
    argv: list[str],
    *,
    clone_dir: Path,
    logs_dir: Path,
    log_name: str,
    timeout: int | None,
    env: dict,
    variant_id: str,
    binary_hint: str = "builder binary",
    sandbox_read_paths: list[Path] | None = None,
) -> BuilderResult:
    """Run a builder subprocess with file-based stdout, parse the log, and
    return a BuilderResult.

    ``sandbox_read_paths`` names extra read-only roots the Landlock ruleset
    must grant when the clone is an OSWorld V2 island. An agent whose runtime
    lives outside the clone cannot import itself without them.
    """
    log_path = Path(logs_dir) / f"{log_name}.jsonl"
    os.makedirs(logs_dir, exist_ok=True)

    timeout_label = "none" if timeout is None else f"{timeout}s"
    logger.info(
        "Builder for %s argv[0]=%s timeout=%s",
        variant_id,
        argv[0],
        timeout_label,
    )
    logger.info("  Log: %s", log_path)

    start_time = time.time()
    sandbox_tmp: Path | None = None
    proc: subprocess.Popen | None = None

    # The V2 Base checkout contains gated evaluator logic and sits beside the
    # writable island clone. Cwd alone is not an isolation boundary: enforce a
    # kernel filesystem allowlist before the agent gets autonomous tool use.
    # ScienceBoard_Base likewise holds every official evaluator and answer.
    if Path(clone_dir).name.startswith(("OSWorldV2_Evol_", "ScienceBoard_Evol_")):
        from agent_s4.builder_sandbox import (
            landlock_abi,
            sandboxed_builder_argv,
        )

        if landlock_abi() == 0:
            raise RuntimeError(
                "Landlock is unavailable; refusing an unsandboxed Builder "
                "beside gated benchmark tasks"
            )
        sandbox_tmp = Path(clone_dir) / ".builder_tmp"
        sandbox_tmp.mkdir(mode=0o700, exist_ok=True)
        argv = sandboxed_builder_argv(
            argv,
            clone_dir,
            read_paths=sandbox_read_paths,
        )

    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                argv,
                cwd=str(clone_dir),
                stdout=log_f,
                stderr=subprocess.DEVNULL,
                text=True,
                env=env,
            )
            try:
                if timeout is None:
                    proc.wait()
                else:
                    proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("Builder timed out after %ds, terminating", timeout)
                proc.terminate()
                proc.wait(timeout=30)
    except FileNotFoundError:
        logger.error("%s not found on PATH (argv[0]=%r)", binary_hint, argv[0])
        return BuilderResult(
            exit_code=1,
            cost_usd=0.0,
            log_path=log_path,
            crashed=True,
        )
    finally:
        if sandbox_tmp is not None:
            try:
                if proc is not None and proc.returncode not in {None, 0}:
                    runtime_logs = sandbox_tmp / "prime-home" / "logs"
                    if runtime_logs.is_dir():
                        diagnostic_dir = Path(logs_dir) / f"{log_name}_runtime_logs"
                        if diagnostic_dir.exists():
                            shutil.rmtree(diagnostic_dir)
                        shutil.copytree(runtime_logs, diagnostic_dir)
                        logger.warning(
                            "Preserved failed Builder runtime logs at %s",
                            diagnostic_dir,
                        )
                if sandbox_tmp.is_symlink():
                    sandbox_tmp.unlink()
                elif sandbox_tmp.exists():
                    shutil.rmtree(sandbox_tmp)
            except OSError as exc:
                logger.warning("Could not remove Builder private temp dir: %s", exc)

    elapsed = time.time() - start_time
    cost, saw_result, saw_rate_limit_blocked = parse_builder_log(log_path)

    crashed = False
    if elapsed < _CRASH_ELAPSED_CEILING and not saw_result and cost == 0:
        logger.error(
            "Builder CRASHED for %s (elapsed=%.1fs, no result event, $0 cost)",
            variant_id,
            elapsed,
        )
        crashed = True
    if saw_rate_limit_blocked:
        logger.error("Builder hit BLOCKED rate limit for %s", variant_id)
        crashed = True

    exit_code = proc.returncode if proc.returncode is not None else 1
    logger.info(
        "Builder done for %s (exit=%d, cost=$%.2f, crashed=%s)",
        variant_id,
        exit_code,
        cost,
        crashed,
    )
    return BuilderResult(
        exit_code=exit_code,
        cost_usd=cost,
        log_path=log_path,
        crashed=crashed,
    )
