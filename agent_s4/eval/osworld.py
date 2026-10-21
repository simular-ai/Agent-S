"""
Shadow Interface evaluation runner — OSWorld v1.

Deploys the shadow interface to VMs, runs tasks in parallel using
Prime Agent, and returns results.

This module also holds the pieces both OSWorld generations share, because
they were written here first: the system prompts, the agent-command and
episode plumbing, result persistence, the worker/main process fan-out, and
the common CLI surface. agent_s4.eval.osworld_v2 imports them and
overrides only what V2 changes — the same arrangement
agent_s4.eval.scienceboard_prime uses.

The dividing line: a helper whose body branches on the OSWorld generation
belongs in the generation's own module. A helper that reads task attributes
and is correct for both stays here.

Usage:
    # Run specific tasks
    python -m agent_s4.eval.osworld \
        --task_ids abc123 def456 --num_envs 2 --model openai/<deployment>

    # Run all tasks
    python -m agent_s4.eval.osworld \
        --all --num_envs 20 --model openai/<deployment>

    # Baseline run (no shadow interface)
    python -m agent_s4.eval.osworld \
        --all --num_envs 20 --model openai/<deployment> --no_tool
"""

import argparse
import datetime
import json
import logging
import math
import os
import signal
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from multiprocessing import Process, Manager, current_process

from desktop_env.desktop_env import DesktopEnv
from agent_s4.harness.opencode.evaluator import (
    MODEL_CONFIG as OPENCODE_HARNESS_CONFIG,
)
from agent_s4.harness.prime.evaluator import (
    MODEL_CONFIG as PRIME_AGENT_HARNESS_CONFIG,
)
from agent_s4.harness.opencode.config import (
    load_context_config as load_opencode_context,
)
from agent_s4.harness.prime.context_window import interface_window_is_honoured
from agent_s4.harness.prime.compaction import (
    context_window_from_config,
    load_compaction_config,
)
from agent_s4.utils import compact_prime_trajectory_jsonl
from agent_s4.scoring import canonicalize_full_credit, is_binary_success
from agent_s4.vm.owner_tag import OWNER_TAG_ENV, install_owner_tag_hook
from agent_s4.vm.utils import (
    AgentCommand,
    _execute_shell_on_vm,
    upload_shadow_interface,
    run_setup_script,
    run_agent_on_vm,
)

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("desktopenv.shadow_eval")

DEFAULT_TASK_TIMEOUT_SECONDS = 1200


class HarnessFailure(RuntimeError):
    def __init__(self, message: str, failure_type: str):
        super().__init__(message)
        self.failure_type = failure_type


NO_TOOL_SYSTEM_PROMPT = """\
You are on an Ubuntu 22.04 desktop. Complete the task below.

## Important

- You can use any approach: GUI interaction, scripts, command-line tools.
- To take a screenshot: scrot /tmp/screenshot.png
- If the task is impossible, create /tmp/INFEASIBLE and exit.
- When done, simply exit.
"""

INSTRUCTION_CONTEXT_SUFFIX = (
    "\n\nThis instruction is in the context of whatever is currently open or "
    "running on the computer. You may need to investigate that for yourself. "
    "Do not ask user any follow-up questions."
)


TOOL_SYSTEM_PROMPT = """\
You are an autonomous agent on a live Ubuntu 22.04 desktop VM. You have FULL access to the screen, keyboard, mouse, filesystem, browser, and all installed applications. Complete the task below by EXECUTING commands — not by giving instructions.

## Rules

- NEVER say "I don't have access to the browser/screen/GUI" — you DO.  
- NEVER give the user step-by-step instructions. EXECUTE the steps yourself.
- NEVER ask the user for file paths, URLs, or clarification. FIND what you need.
- If you're unsure about something, EXPLORE first then ACT.
- When done, simply exit.

## shadow_interface

An automation toolkit is installed at /home/user/shadow_interface/. Check what it offers:

    python3 -m shadow_interface --help
    python3 -m shadow_interface capabilities

Use it as your primary approach. Read the code directly if the docs are unclear:

    cat /home/user/shadow_interface/helpers/*.py
"""


@dataclass(frozen=True)
class EnvSpec:
    """Canonical task-level VM configuration used to decide VM reuse."""

    os_type: str
    snapshot_name: str
    instance_type: str | None
    volume_size: int | None
    enable_proxy: bool
    force_disable_vnc: bool
    force_disable_recording: bool
    cache_dir: str
    is_v2: bool


def _task_get(example, key, default=None):
    """Read a task setting from either dict data or a BaseTask class attr."""
    missing = object()
    value = example.get(key, missing)
    if value is missing:
        value = getattr(example, key, default)
    return value


def resolve_cache_dir(example, args) -> str:
    override = getattr(args, "cache_dir", None)
    return override or os.path.join(
        os.path.abspath(args.result_dir),
        "_task_cache",
        str(example.get("id", "unknown")),
    )


def build_env_spec(example, args) -> EnvSpec:
    """Resolve the VM spec for a v1 JSON task.

    v1 tasks carry no VM configuration of their own, so everything comes from
    the CLI: the region's default image, Ubuntu, no instance/volume override.
    """
    screen_size = (args.screen_width, args.screen_height)
    snapshot_name = getattr(args, "snapshot_name", "init_state")
    if args.provider_name == "aws":
        from desktop_env.providers.aws.manager import IMAGE_ID_MAP

        region_images = IMAGE_ID_MAP[args.region]
        snapshot_name = region_images.get(screen_size, region_images[(1920, 1080)])

    proxy_override = getattr(args, "enable_proxy", None)
    return EnvSpec(
        os_type="Ubuntu",
        snapshot_name=snapshot_name,
        instance_type=None,
        volume_size=getattr(args, "volume_size", None),
        enable_proxy=False if proxy_override is None else bool(proxy_override),
        force_disable_vnc=False,
        force_disable_recording=False,
        cache_dir=resolve_cache_dir(example, args),
        is_v2=False,
    )


def build_env_kwargs(spec: EnvSpec, args) -> dict:
    """Translate an EnvSpec to the DesktopEnv generation selected by cwd.

    Only kwargs both generations accept. osworld_v2.build_env_kwargs adds the
    ones that exist solely in V2's DesktopEnv.
    """
    kwargs = {
        "path_to_vm": args.path_to_vm,
        "action_space": "pyautogui",
        "provider_name": args.provider_name,
        "screen_size": (args.screen_width, args.screen_height),
        "headless": args.headless,
        "os_type": spec.os_type,
        "require_a11y_tree": False,
        "enable_proxy": spec.enable_proxy,
        "client_password": args.client_password,
        "cache_dir": spec.cache_dir,
    }
    if args.provider_name == "aws":
        kwargs.update(region=args.region, snapshot_name=spec.snapshot_name)
    return kwargs


def persist_evaluation_result(
    raw_result, result_dir: str, *, suffix: str = ""
) -> tuple[float, dict | None]:
    """Persist an evaluator result without losing rich payloads.

    OSWorld v1 evaluators return a float. V2's return either a float or a dict
    carrying partial-credit and safety detail alongside its 'score'. Both are
    normalised to a numeric result.txt so every downstream reader can parse
    it; a dict additionally lands verbatim in result.json.
    """
    details = raw_result if isinstance(raw_result, dict) else None
    if details is not None:
        if "score" not in details:
            raise ValueError("dict evaluation result is missing required 'score'")
        try:
            score = float(details["score"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"dict evaluation result has non-numeric score: {details['score']!r}"
            ) from exc
    else:
        score = float(raw_result)
    if not math.isfinite(score):
        raise ValueError(f"evaluation returned non-finite score: {score!r}")
    evaluator_raw_score = score
    score = canonicalize_full_credit(score)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"evaluation score is outside [0, 1]: {score!r}")
    if details is not None:
        if score != evaluator_raw_score:
            details = dict(details)
            details["evaluator_raw_score"] = evaluator_raw_score
            details["score"] = score
        with open(
            os.path.join(result_dir, f"result{suffix}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(details, f, indent=2, ensure_ascii=False, default=str)
    with open(
        os.path.join(result_dir, f"result{suffix}.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(f"{score}\n")
    return score, details


def build_task_index(test_config_base_dir):
    """Build task_id -> definition-file path for the v1-style corpus.

    examples/<domain>/<uuid>.json, which both OSWorld generations ship.
    osworld_v2.build_task_index extends this with V2's task_class/ corpus.
    """
    index = {}
    examples_dir = os.path.join(test_config_base_dir, "examples")
    if os.path.isdir(examples_dir):
        for folder in os.listdir(examples_dir):
            folder_path = os.path.join(examples_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.endswith(".json"):
                    task_id = fname[:-5]  # strip .json
                    index[task_id] = os.path.join(folder_path, fname)
    return index


def load_task(task_path, test_config_base_dir):
    """Load one v1 task definition."""
    with open(task_path) as f:
        return json.load(f)


def load_all_task_ids(test_all_meta_path):
    """Load all task IDs from a v1 manifest: a flat list, or domain -> [ids]."""
    with open(test_all_meta_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return list(data)
    all_ids = []
    for task_ids in data.values():
        all_ids.extend(task_ids)
    return all_ids


def load_system_prompt(shadow_dir, addendum: str = ""):
    prompt = TOOL_SYSTEM_PROMPT
    if addendum:
        prompt = prompt + "\n\n" + addendum
    return prompt


def _load_prime_context_window_tokens(args):
    """Load the evolvable Prime context limit from the active interface.

    Omitting the file or field intentionally returns ``None`` so Prime keeps
    the context window advertised by its built-in model catalog. A run pinned
    to that catalog window ignores the field entirely; see
    agent_s4.harness.prime.context_window.
    """
    if not interface_window_is_honoured():
        return None
    return context_window_from_config(
        load_compaction_config(getattr(args, "shadow_dir", None) or None)
    )


# Below this, a fresh episode has no room to do anything but pay its own
# startup cost, so the budget is treated as spent instead.
MIN_EPISODE_SECONDS = 60


class TaskBudget:
    """One wall clock for a whole task, started once its VM is ready.

    ``--timeout`` bounds a single agent episode, and a V2 task is often
    several: one per phase, plus one more per ASK_USER follow-up turn. A cap
    that is meant to bound *a task* therefore cannot be handed to each episode
    independently -- a 4-phase task answering the user simulator up to 5 times
    would be entitled to twenty times the cap. The budget is shared across
    every episode of the task instead, and each one is launched with whatever
    is left.

    The clock starts after VM allocation, reset and interface deployment.
    Provisioning is infrastructure time whose cost varies with the provider,
    not agent work, and charging it to the agent would make the cap mean a
    different thing on a slow morning than on a fast one.

    ``seconds=None`` is an unlimited budget, which is what --no-time-limit
    passes; remaining() then stays None and every episode runs uncapped.
    """

    def __init__(self, seconds):
        self.seconds = seconds
        self._deadline = None
        self._launched = 0

    def start(self):
        """Begin the clock. Idempotent, so a re-entered task cannot restart it."""
        if self.seconds is not None and self._deadline is None:
            self._deadline = time.monotonic() + self.seconds
        return self

    def remaining(self):
        """Seconds left, or None when unlimited. Never negative."""
        if self.seconds is None:
            return None
        if self._deadline is None:
            return self.seconds
        return max(0.0, self._deadline - time.monotonic())

    def note_launch(self):
        self._launched += 1

    def spent(self):
        """True when what is left is too small to be worth another episode.

        The task always gets its first episode, however small the budget: a
        cap below the floor is a deliberately tiny configuration (a smoke
        test, a diagnostic), and refusing to run anything under it would turn
        that configuration into a silent no-op. The floor only decides whether
        the task gets *another* episode -- a later phase, or an ASK_USER
        continuation -- once time has actually been consumed.
        """
        remaining = self.remaining()
        if remaining is None or self._launched == 0:
            return False
        return remaining < MIN_EPISODE_SECONDS


def _build_agent_command(
    args,
    model_cfg,
    instruction,
    system_prompt,
    *,
    api_key,
    openrouter_key,
    parasail_key,
    continuation_attempt=False,
    episode_timeout=None,
):
    """Build the in-VM agent command.

    ``episode_timeout`` is the wall clock this one episode may use, which is
    what the harnesses wrap their runner in. It is what remains of the task's
    budget rather than the flat ``--timeout`` whenever a TaskBudget is in
    play; None keeps the flat value, so every caller that does not budget is
    unaffected.
    """
    build_command = model_cfg["build_command"]
    # Rounded up, and never below 1: int() truncation would hand GNU timeout a
    # 0, which it reads as "no limit" -- the cap would silently invert into its
    # own opposite in exactly the last seconds of a budget.
    timeout = (
        args.timeout if episode_timeout is None else max(1, math.ceil(episode_timeout))
    )
    if model_cfg["binary"] == "opencode":
        command = build_command(
            instruction=instruction,
            system_prompt=system_prompt,
            model=args.model,
            timeout=timeout,
            api_key=openrouter_key,
            parasail_api_key=parasail_key,
            max_steps=getattr(args, "max_steps", None),
            no_time_limit=getattr(args, "no_time_limit", False),
            continue_session=continuation_attempt,
            context_compaction=load_opencode_context(getattr(args, "shadow_dir", None)),
        )
    else:
        context_compaction = load_compaction_config(
            getattr(args, "shadow_dir", None) or None
        )
        command = build_command(
            instruction=instruction,
            system_prompt=system_prompt,
            model=args.model,
            timeout=timeout,
            api_key=api_key,
            openrouter_api_key=openrouter_key,
            parasail_api_key=parasail_key,
            or_provider=getattr(args, "or_provider", None),
            max_steps=getattr(args, "max_steps", None),
            no_time_limit=getattr(args, "no_time_limit", False),
            continue_session=continuation_attempt,
            context_window_tokens=context_window_from_config(context_compaction),
            context_compaction=context_compaction,
        )
    accepted_exit_codes = model_cfg.get("accepted_exit_codes")
    if accepted_exit_codes is not None and not isinstance(command, AgentCommand):
        command = AgentCommand(
            script=command,
            accepted_exit_codes=tuple(accepted_exit_codes),
        )
    return command


def _agent_failure_message(artifacts: dict) -> str:
    """Describe an agent failure using whatever the VM managed to write.

    run_agent_on_vm already distinguishes a missing, unparseable, or non-zero
    exit marker; what it cannot report is *why*. The runner's stderr usually
    can, so put its tail in the message that reaches the summary.
    """
    base = "agent launch/execution failed"
    trajectory = artifacts.get("claude_trajectory") or ""
    if isinstance(trajectory, bytes):
        trajectory = trajectory.decode("utf-8", errors="replace")
    try:
        terminal = json.loads(trajectory.rstrip().rsplit("\n", 1)[-1])
    except (json.JSONDecodeError, TypeError):
        terminal = {}
    if isinstance(terminal, dict) and terminal.get("type") in {
        "prime_agent_idle_timeout",
        "prime_agent_model_timeout",
    }:
        details = {
            key: terminal[key]
            for key in (
                "type",
                "phase",
                "last_event",
                "elapsed_seconds",
                "budget_seconds",
                "idle_seconds",
                "steps",
            )
            if key in terminal
        }
        return f"{base}: {json.dumps(details, separators=(',', ':'))}"
    stderr = artifacts.get("claude_stderr") or b""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    tail = stderr.strip()[-500:]
    return f"{base}: {tail}" if tail else base


def _terminal_agent_event(trajectory) -> str | None:
    """Return a valid terminal event from the last JSONL record, if present.

    The exit-code marker is transport metadata.  A complete trajectory is the
    authoritative agent outcome, especially for ``prime_step_limit``: reaching
    the configured step budget is an agent failure/outcome, not an
    infrastructure failure that should consume an infra retry.
    """
    if not trajectory:
        return None
    if isinstance(trajectory, bytes):
        trajectory = trajectory.decode("utf-8", errors="replace")
    # split("\n"): splitlines() also breaks on U+0085 inside JSON strings.
    lines = [line for line in trajectory.split("\n") if line.strip()]
    if not lines:
        return None
    try:
        event = json.loads(lines[-1])
    except (json.JSONDecodeError, TypeError):
        return None
    event_type = event.get("type") if isinstance(event, dict) else None
    return event_type if event_type in {"agent_end", "prime_step_limit"} else None


def _run_agent_episode(
    env,
    args,
    model_cfg,
    instruction,
    system_prompt,
    example_result_dir,
    *,
    phase_index=None,
    artifact_chunks=None,
    ask_user=None,
    budget=None,
):
    """Run the coding agent until it stops asking the benchmark for input.

    ``ask_user`` is the generation's follow-up protocol. It receives the
    episode's trajectory bytes, the 1-based attempt number and the phase
    index, and returns text to append for a continuation episode on the same
    VM — or None to finish. OSWorld v1 has no such protocol, so the default
    runs exactly one episode.

    ``budget`` is the task's shared wall clock (see TaskBudget). When given,
    each episode is launched with what remains of it rather than the flat
    ``--timeout``, and the task stops asking for another once the remainder is
    too small to be worth a VM round trip. Stopping is not a failure: the
    desktop state reached so far is what the evaluator scores, exactly as it
    scores a single episode that hit its own deadline. None keeps the flat
    per-episode behaviour for callers that do not budget.
    """
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    openrouter_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    parasail_key = getattr(args, "parasail_api_key", None) or os.environ.get(
        "PARASAIL_API_KEY"
    )
    continuation = instruction

    no_time_limit = getattr(args, "no_time_limit", False)
    attempt = 0
    while True:
        attempt += 1
        if budget is not None and budget.spent():
            logger.warning(
                "Task budget spent%s; scoring the state reached rather than "
                "starting an episode with %.0fs left",
                "" if phase_index is None else f" during phase {phase_index}",
                budget.remaining(),
            )
            return
        episode_timeout = None if budget is None else budget.remaining()
        command = _build_agent_command(
            args,
            model_cfg,
            continuation,
            system_prompt,
            api_key=api_key,
            openrouter_key=openrouter_key,
            parasail_key=parasail_key,
            continuation_attempt=attempt > 1,
            episode_timeout=episode_timeout,
        )
        if no_time_limit:
            driver_timeout = None
        elif episode_timeout is None:
            driver_timeout = args.timeout
        else:
            driver_timeout = max(1, math.ceil(episode_timeout))
        if budget is not None:
            budget.note_launch()
        launched = run_agent_on_vm(env.controller, command, driver_timeout)

        # Pull artefacts before deciding the episode failed. Raising first
        # would discard exactly the stderr/trajectory needed to diagnose the
        # failure, leaving the VM as the only copy.
        artifacts = {}
        artifact_specs = [
            ("/tmp/claude_task/trajectory.jsonl", "claude_trajectory", ".jsonl"),
            ("/tmp/claude_task/stderr.log", "claude_stderr", ".log"),
            ("/tmp/claude_task/exit_code.txt", "exit_code", ".txt"),
        ]
        # /tmp/claude_task/stdout.log is not pulled: Prime's EXIT trap copies
        # trajectory.jsonl byte-for-byte to it, so downloading it doubles host
        # storage without adding any diagnostic evidence.
        if model_cfg["binary"] == "opencode":
            artifact_specs.extend(
                [
                    ("/tmp/claude_task/opencode.json", "opencode_config", ".json"),
                    ("/tmp/claude_task/runner.json", "opencode_runner", ".json"),
                    (
                        "/tmp/claude_task/opencode-state/session.json",
                        "opencode_session",
                        ".json",
                    ),
                ]
            )
        for vm_path, stem, extension in artifact_specs:
            try:
                data = env.controller.get_file(vm_path)
            except Exception:
                data = None
            if data is None:
                continue
            if stem == "claude_trajectory" and model_cfg["binary"] == "prime-agent":
                data = compact_prime_trajectory_jsonl(data)
            label = ""
            if phase_index is not None:
                label += f"_phase_{phase_index}"
            if attempt > 1:
                label += f"_attempt_{attempt}"
            path = os.path.join(example_result_dir, f"{stem}{label}{extension}")
            with open(path, "wb") as f:
                f.write(data)
            artifacts[stem] = data
            if stem == "claude_trajectory" and artifact_chunks is not None:
                artifact_chunks.append(data)

        terminal_event = _terminal_agent_event(artifacts.get("claude_trajectory"))
        if not launched and terminal_event:
            logger.warning(
                "Agent transport reported failure, but trajectory ended with %s; "
                "treating it as a completed agent outcome (no infra retry)",
                terminal_event,
            )
            launched = True

        if not launched:
            raise HarnessFailure(
                _agent_failure_message(artifacts),
                "agent_execution_failure",
            )

        validate_artifacts = model_cfg.get("validate_artifacts")
        validation_error = validate_artifacts(artifacts) if validate_artifacts else None
        if validation_error:
            raise HarnessFailure(validation_error, "agent_execution_failure")

        if ask_user is None:
            return
        follow_up = ask_user(artifacts.get("claude_trajectory"), attempt, phase_index)
        if not follow_up:
            return
        # Session continuation already carries the preceding prompt and compacted
        # context; send only the new benchmark response.
        continuation = follow_up


def _build_eval_constraint_command(client_password: str) -> str:
    """Remove builder-controlled instruction files before agent execution.

    Do not restrict task-side package managers: some official V2 tasks (for
    example task 030) explicitly require the agent to build an environment.
    A writable PATH shim is neither a security boundary nor benchmark-faithful.
    Interface dependency policy belongs in a separate unprivileged sandbox.
    """
    return (
        "set -e; "
        "rm -f /home/user/CLAUDE.md /home/user/.claude/CLAUDE.md "
        "/home/user/shadow_interface/CLAUDE.md 2>/dev/null; "
        "echo 'Eval constraints enforced'"
    )


def _exception_chain(exc: BaseException) -> str:
    """Render nested setup failures without discarding the socket root cause."""
    parts = []
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " <- caused by ".join(parts)


def prepare_task_vm(env, example, args, model_cfg, shadow_dir):
    """Reset the VM for a task and stage the interface and coding agent on it.

    Returns ``(system_prompt, recording_enabled)``, or a ``_fail_result`` dict
    that the caller should return unchanged.
    """
    example_id = example.get("id", "unknown")

    client_password = args.client_password or env.client_password or "password"
    installer = model_cfg["installer"]

    def stage_harness():
        if not args.no_tool:
            if not upload_shadow_interface(env.controller, shadow_dir):
                return _fail_result(
                    example_id,
                    "shadow interface upload failed",
                    "task_setup_failure",
                )
            if not run_setup_script(env.controller, client_password=client_password):
                return _fail_result(
                    example_id,
                    "shadow interface setup failed",
                    "task_setup_failure",
                )

        if not installer(env.controller, client_password):
            return _fail_result(
                example_id,
                f"agent install failed ({model_cfg['binary']})",
                "agent_install_failure",
            )
        return None

    install_before_setup = bool(model_cfg.get("install_before_task_setup", False))
    if install_before_setup:
        # Some V2 tasks deliberately revoke sudo during setup. First return a
        # reused VM to its clean snapshot, then install while deployment
        # privileges still exist. reset(example) then applies only task setup.
        try:
            env.reset(task_config=None)
        except Exception as exc:
            return _fail_result(
                example_id,
                f"clean VM reset failed: {_exception_chain(exc)}",
                "task_setup_failure",
            )
        failure = stage_harness()
        if failure is not None:
            return failure

    try:
        env.reset(task_config=example)
    except Exception as exc:
        return _fail_result(
            example_id,
            f"task reset/setup failed: {_exception_chain(exc)}",
            "task_setup_failure",
        )
    time.sleep(60)

    recording_enabled = not bool(_task_get(example, "disable_recording", False))
    if recording_enabled:
        env.controller.start_recording()

    if install_before_setup:
        # reset() can retry by reverting internally. Re-run the idempotent
        # installer as a probe/repair so a lost runtime is classified here.
        if not installer(env.controller, client_password):
            return _fail_result(
                example_id,
                f"agent runtime missing after task setup ({model_cfg['binary']})",
                "agent_install_failure",
            )
    else:
        failure = stage_harness()
        if failure is not None:
            return failure

    if not args.no_tool:
        # Enforce constraints after setup.sh runs
        constraint_result = _execute_shell_on_vm(
            env.controller,
            _build_eval_constraint_command(client_password),
            timeout=30,
        )
        if constraint_result is None or constraint_result.get("returncode", -1) != 0:
            return _fail_result(
                example_id,
                "eval constraint enforcement failed",
                "task_setup_failure",
            )

        system_prompt = load_system_prompt(
            shadow_dir, getattr(args, "prompt_addendum", "")
        )
    else:
        system_prompt = NO_TOOL_SYSTEM_PROMPT

    return system_prompt, recording_enabled


def finalize_task_result(
    env,
    example_id,
    args,
    model_cfg,
    example_result_dir,
    *,
    base_instruction,
    score,
    result_details,
    artifact_chunks,
    start_time,
    phase_count,
    hitl_turns,
    recording_enabled,
):
    """Write a task's artifacts and build its result record."""
    elapsed = time.time() - start_time
    if artifact_chunks:
        with open(
            os.path.join(example_result_dir, "claude_trajectory.jsonl"), "wb"
        ) as f:
            for chunk in artifact_chunks:
                f.write(chunk)
                if not chunk.endswith(b"\n"):
                    f.write(b"\n")

    with open(os.path.join(example_result_dir, "traj.jsonl"), "w") as f:
        f.write(
            json.dumps(
                {
                    "agent": "shadow_eval",
                    "model": args.model,
                    "model_flag": model_cfg["model_flag"],
                    "instruction": base_instruction,
                    "elapsed_seconds": elapsed,
                    "no_tool": args.no_tool,
                    "phase_count": phase_count,
                    "hitl_turns": hitl_turns,
                }
            )
            + "\n"
        )

    # Save instruction separately for builder context
    with open(os.path.join(example_result_dir, "instruction.txt"), "w") as f:
        f.write(base_instruction)

    if recording_enabled:
        try:
            env.controller.end_recording(
                os.path.join(example_result_dir, "recording.mp4")
            )
        except Exception:
            pass

    failure_summary = ""
    if (
        score == 0
        and (result_details or {}).get("verification") != "dummy_constant_zero"
    ):
        traj_path = os.path.join(example_result_dir, "claude_trajectory.jsonl")
        if os.path.exists(traj_path):
            failure_summary = _extract_failure_summary(traj_path)

    return {
        "task_id": example_id,
        "score": score,
        "elapsed": elapsed,
        "failure_summary": failure_summary,
        "details": result_details,
        "valid_for_fitness": True,
    }


def run_single_task(
    env, example, instruction, example_result_dir, args, model_cfg, shadow_dir
):
    """Run a single v1 task: reset VM, deploy tool, run agent, evaluate."""
    start_time = time.time()
    example_id = example.get("id", "unknown")
    base_instruction = instruction + INSTRUCTION_CONTEXT_SUFFIX

    staged = prepare_task_vm(env, example, args, model_cfg, shadow_dir)
    if isinstance(staged, dict):
        return staged
    system_prompt, recording_enabled = staged

    artifact_chunks = []
    _run_agent_episode(
        env,
        args,
        model_cfg,
        base_instruction,
        system_prompt,
        example_result_dir,
        artifact_chunks=artifact_chunks,
    )
    infeasible_data = env.controller.get_file("/tmp/INFEASIBLE")
    env.action_history = ["FAIL"] if infeasible_data is not None else ["DONE"]
    env.is_environment_used = True
    time.sleep(20)
    score, result_details = persist_evaluation_result(
        env.evaluate(),
        example_result_dir,
    )

    return finalize_task_result(
        env,
        example_id,
        args,
        model_cfg,
        example_result_dir,
        base_instruction=base_instruction,
        score=score,
        result_details=result_details,
        artifact_chunks=artifact_chunks,
        start_time=start_time,
        phase_count=1,
        hitl_turns=0,
        recording_enabled=recording_enabled,
    )


def _fail_result(
    task_id, error, failure_type="agent_failure", *, valid_for_fitness=False
):
    return {
        "task_id": task_id,
        "score": 0.0,
        "error": error,
        "failure_type": failure_type,
        "valid_for_fitness": valid_for_fitness,
    }


def persist_task_failure(result_dir: str, result: dict, *, stage: str) -> None:
    """Atomically persist an infra/config failure independently of the summary."""
    payload = {
        **result,
        "status": "infra_failure",
        "stage": stage,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    os.makedirs(result_dir, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".failure.",
        suffix=".json.tmp",
        dir=result_dir,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, os.path.join(result_dir, "failure.json"))
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def clear_task_failure(result_dir: str) -> None:
    """Remove a stale failure manifest after a successful resumed attempt."""
    try:
        os.unlink(os.path.join(result_dir, "failure.json"))
    except FileNotFoundError:
        pass


def try_persist_task_failure(result_dir: str, result: dict, *, stage: str) -> None:
    """Keep failure reporting from masking the original task failure."""
    try:
        persist_task_failure(result_dir, result, stage=stage)
    except Exception as exc:
        logger.error(
            "Could not persist failure manifest for %s: %s",
            result.get("task_id", "unknown"),
            exc,
        )


def _task_reached_prime_step_limit(result_dir: str) -> bool:
    """Check the canonical artifact without loading a potentially huge trace."""
    trajectory_path = os.path.join(result_dir, "claude_trajectory.jsonl")
    try:
        with open(trajectory_path, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 65536))
            tail = stream.read()
    except OSError:
        return False
    return _terminal_agent_event(tail) == "prime_step_limit"


def retry_or_record_infra_failure(
    task_queue,
    result_list,
    task_item,
    result: dict,
    args,
    result_dir: str,
    *,
    stage: str,
    proc_name: str,
) -> bool:
    """Retry an invalid task result, then score exhausted failures as zero.

    Returns true when the task was requeued.  Invalid attempts are deliberately
    not appended to ``result_list``: every requested task must contribute
    exactly one final result to the summary, and a later successful attempt
    must replace rather than coexist with the transient infrastructure error.
    Once retries are exhausted, persist an explicit imputed zero so a single
    broken VM/agent cannot block an unattended evolution batch.  The typed
    failure metadata remains in both result.json and failure.json for audits.
    """
    example_id, task_path, attempt = task_item
    max_retries = getattr(args, "infra_retries", 0)
    failure = {
        **result,
        "attempt": attempt + 1,
        "max_attempts": max_retries + 1,
    }
    if _task_reached_prime_step_limit(result_dir):
        # A downstream marker/controller/evaluator error must not cause the
        # already exhausted agent episode to run from zero on a fresh VM.
        # Preserve it as a real agent-budget outcome, not an infra imputation.
        terminal = {
            **failure,
            "score": 0.0,
            "error": "agent reached configured prime step limit",
            "failure_type": "agent_step_limit",
            "valid_for_fitness": True,
            "imputed_zero": False,
            "status": "agent_failure",
            "retryable": False,
            "terminal_event": "prime_step_limit",
        }
        logger.warning(
            "[%s] %s reached prime_step_limit; recording agent zero without "
            "an infrastructure retry",
            proc_name,
            example_id,
        )
        try_persist_task_failure(result_dir, terminal, stage=stage)
        persist_evaluation_result(terminal, result_dir)
        result_list.append(terminal)
        return False
    if attempt < max_retries:
        try_persist_task_failure(result_dir, failure, stage=stage)
        logger.warning(
            "[%s] Infrastructure failure on %s attempt %d/%d; "
            "requeueing on a fresh VM: %s",
            proc_name,
            example_id,
            attempt + 1,
            max_retries + 1,
            result.get("error", result.get("failure_type", "unknown failure")),
        )
        task_queue.put((example_id, task_path, attempt + 1))
        return True
    logger.error(
        "[%s] Infrastructure retries exhausted for %s after %d attempt(s); "
        "recording an imputed zero",
        proc_name,
        example_id,
        attempt + 1,
    )
    imputed = {
        **failure,
        "score": 0.0,
        "valid_for_fitness": True,
        "imputed_zero": True,
    }
    try_persist_task_failure(result_dir, imputed, stage=stage)
    try:
        persist_evaluation_result(imputed, result_dir)
    except Exception as exc:
        logger.error(
            "Could not persist imputed zero for %s: %s",
            example_id,
            exc,
        )
        # A missing result.txt would later be indistinguishable from an
        # interrupted task. Keep it invalid so the enclosing evaluator refuses
        # to ingest an incomplete cache instead of silently fabricating data.
        imputed["valid_for_fitness"] = False
        imputed["imputation_error"] = f"{type(exc).__name__}: {exc}"
    result_list.append(imputed)
    return False


def _extract_failure_summary(traj_path):
    try:
        lines = []
        with open(traj_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))

        summary_parts = []
        count = 0
        for entry in reversed(lines):
            if entry.get("type") == "assistant":
                for block in entry.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        summary_parts.append(f"[text] {block['text'][:200]}")
                    elif block.get("type") == "tool_use":
                        inp = str(block.get("input", ""))[:200]
                        summary_parts.append(f"[{block['name']}] {inp}")
                count += 1
                if count >= 5:
                    break
            if entry.get("type") == "result":
                errors = entry.get("errors", [])
                if errors:
                    summary_parts.append(f"[errors] {errors[0][:300]}")

        summary_parts.reverse()
        return "\n".join(summary_parts)
    except Exception as e:
        return f"Could not parse trajectory: {e}"


def worker(task_queue, result_list, args, shadow_dir, model_cfg, *, impl=None):
    """Worker process: load each task first, then provision its exact VM.

    ``impl`` is the eval module owning the generation-specific pieces —
    load_task, build_env_spec, build_env_kwargs, DesktopEnv, run_single_task.
    It defaults to this module; agent_s4.eval.osworld_v2 passes itself so
    its overrides, and any test that monkeypatches them, are what run.
    """
    impl = impl if impl is not None else sys.modules[__name__]
    # Register in the child as well as main() so cleanup also works if the
    # multiprocessing start method is ever changed from fork to spawn.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    # Same reason: the worker is the process that actually calls RunInstances,
    # and an instance launched without the Owner tag can never be terminated
    # (see agent_s4/vm/owner_tag.py). Under fork it is already installed.
    install_owner_tag_hook()
    env = None
    current_env_spec = None
    try:
        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break

            # The third field tracks infrastructure retries.  Accept the old
            # two-field shape for direct callers and backward-compatible tests.
            if len(item) == 2:
                example_id, task_path = item
                item = (example_id, task_path, 0)
            else:
                example_id, task_path, _attempt = item
            proc_name = current_process().name

            example_result_dir = os.path.join(
                args.result_dir,
                args.action_space,
                args.observation_type,
                args.model,
                example_id,
            )
            os.makedirs(example_result_dir, exist_ok=True)

            # Queue carries the path, not the task — see the comment in main().
            try:
                example = impl.load_task(task_path, args.test_config_base_dir)
            except HarnessFailure as e:
                logger.error("[%s] Harness failure on %s: %s", proc_name, example_id, e)
                failure = _fail_result(
                    example_id,
                    str(e),
                    e.failure_type,
                )
                try_persist_task_failure(example_result_dir, failure, stage="task_load")
                result_list.append(failure)
                continue
            except Exception as e:
                logger.error(
                    "[%s] Could not load task %s: %s", proc_name, example_id, e
                )
                failure = _fail_result(
                    example_id,
                    f"task load failed: {type(e).__name__}: {e}",
                    "task_load_failure",
                )
                try_persist_task_failure(example_result_dir, failure, stage="task_load")
                result_list.append(failure)
                continue
            # V2 task constructors may generate credentials/instructions
            # dynamically (notably GitLab tasks 026/041). The instruction
            # must come from this same worker-local instance whose setup will
            # run, never from the separate fail-fast instance in main().
            instruction = example["instruction"]
            logger.info("[%s] Running %s: %s", proc_name, example_id, instruction[:100])

            try:
                spec = impl.build_env_spec(example, args)
                with open(
                    os.path.join(example_result_dir, "env_spec.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(asdict(spec), f, indent=2, sort_keys=True)
                if env is None or spec != current_env_spec:
                    if env is not None:
                        env.close()
                    logger.info(
                        "[%s] VM spec for %s: %s", proc_name, example_id, asdict(spec)
                    )
                    env = impl.DesktopEnv(**impl.build_env_kwargs(spec, args))
                    current_env_spec = spec
            except Exception as e:
                logger.error(
                    "[%s] VM allocation failed for %s: %s", proc_name, example_id, e
                )
                failure = _fail_result(
                    example_id,
                    f"VM allocation failed: {type(e).__name__}: {e}",
                    "vm_allocation_failure",
                )
                retry_or_record_infra_failure(
                    task_queue,
                    result_list,
                    item,
                    failure,
                    args,
                    example_result_dir,
                    stage="vm_allocation",
                    proc_name=proc_name,
                )
                env = None
                current_env_spec = None
                continue

            try:
                result = impl.run_single_task(
                    env,
                    example,
                    instruction,
                    example_result_dir,
                    args,
                    model_cfg,
                    shadow_dir,
                )
                if result.get("valid_for_fitness", True):
                    result_list.append(result)
                    clear_task_failure(example_result_dir)
                else:
                    try:
                        env.close()
                    except Exception as close_error:
                        logger.warning(
                            "[%s] VM close after invalid result failed: %s",
                            proc_name,
                            close_error,
                        )
                    finally:
                        env = None
                        current_env_spec = None
                    retry_or_record_infra_failure(
                        task_queue,
                        result_list,
                        item,
                        result,
                        args,
                        example_result_dir,
                        stage=result.get("failure_type", "task_execution"),
                        proc_name=proc_name,
                    )
                status = (
                    "PASS"
                    if is_binary_success(result["score"], args.benchmark)
                    else "FAIL"
                )
                logger.info(
                    "[%s] %s %s (score=%.2f, %.0fs)",
                    proc_name,
                    status,
                    example_id,
                    result["score"],
                    result.get("elapsed", 0),
                )
            except HarnessFailure as e:
                logger.error("[%s] Harness failure on %s: %s", proc_name, example_id, e)
                failure = _fail_result(
                    example_id,
                    str(e),
                    e.failure_type,
                )
                try:
                    env.close()
                except Exception as close_error:
                    logger.warning(
                        "[%s] VM close after harness failure failed: %s",
                        proc_name,
                        close_error,
                    )
                finally:
                    env = None
                    current_env_spec = None
                retry_or_record_infra_failure(
                    task_queue,
                    result_list,
                    item,
                    failure,
                    args,
                    example_result_dir,
                    stage="agent",
                    proc_name=proc_name,
                )
            except Exception as e:
                import traceback

                logger.error("[%s] Exception on %s: %s", proc_name, example_id, e)
                logger.error(traceback.format_exc())
                failure = _fail_result(
                    example_id,
                    f"task execution failed: {type(e).__name__}: {e}",
                    "evaluator_failure",
                )
                try:
                    env.close()
                except Exception as close_error:
                    logger.warning(
                        "[%s] VM close after evaluator failure failed: %s",
                        proc_name,
                        close_error,
                    )
                finally:
                    env = None
                    current_env_spec = None
                retry_or_record_infra_failure(
                    task_queue,
                    result_list,
                    item,
                    failure,
                    args,
                    example_result_dir,
                    stage="task_execution",
                    proc_name=proc_name,
                )

    finally:
        if env:
            try:
                env.close()
            except Exception:
                pass


def print_results_summary(results, *, benchmark: str = "osworld"):
    results = list(results)
    if not results:
        print("\nNo results.")
        return {}

    invalid = [r for r in results if not r.get("valid_for_fitness", True)]
    valid = [r for r in results if r.get("valid_for_fitness", True)]
    passed = [r for r in valid if is_binary_success(r.get("score", 0), benchmark)]
    failed = [r for r in valid if not is_binary_success(r.get("score", 0), benchmark)]

    print("\n" + "=" * 70)
    print(
        f"RESULTS: {len(passed)}/{len(valid)} valid tasks passed "
        f"({len(passed)/len(valid):.1%})"
        if valid
        else "RESULTS: no valid task results"
    )
    print("=" * 70)

    if passed:
        print(f"\n  PASSED ({len(passed)}):")
        for r in passed:
            print(f"    {r['task_id']} ({r.get('elapsed', 0):.0f}s)")

    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for r in failed:
            err = r.get("error", "")
            summary = r.get("failure_summary", "")
            print(f"\n    {r['task_id']}:")
            if err:
                print(f"      Error: {err}")
            if summary:
                for line in summary.split("\n")[:5]:
                    print(f"      {line}")

    if invalid:
        print(f"\n  INFRASTRUCTURE/CONFIG FAILURES ({len(invalid)}):")
        for r in invalid:
            print(
                f"    {r['task_id']}: {r.get('failure_type', 'unknown')} — {r.get('error', '')}"
            )

    print("=" * 70)

    return {
        "total": len(results),
        "passed": len(passed),
        "failed": len(failed),
        "infrastructure_failed": len(invalid),
        "pass_rate": len(passed) / len(valid) if valid else 0,
        "mean_score": (
            sum(float(r.get("score", 0)) for r in valid) / len(valid) if valid else 0
        ),
        "results": results,
    }


def _first_env(names) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def resolve_default_model() -> str | None:
    """Pick the default --model from .env's OpenAI-compatible deployment.

    A model name is only used if its key is also present -- otherwise the
    default would be something that cannot authenticate, and the failure would
    land after a VM is already running. None means --model is required.

    OPENAI_MODEL may be a deployment path beginning with "/", and that leading
    slash is significant (the endpoint rejects the name without it), so it is
    concatenated verbatim after the "openai/" provider prefix rather than
    joined with a separator.
    """
    model = _first_env(("OPENAI_MODEL", "QWEN_MODEL"))
    if model and _first_env(("OPENAI_API_KEY", "QWEN_API_KEY")):
        return f"openai/{model}"
    return None


def add_common_args(parser):
    """CLI surface shared by both OSWorld generations."""
    parser.add_argument(
        "--task_ids", type=str, nargs="+", default=None, help="Task IDs to test"
    )
    parser.add_argument(
        "--all", action="store_true", default=False, help="Run all tasks"
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of parallel VMs"
    )
    parser.add_argument(
        "--infra-retries",
        type=int,
        default=int(os.environ.get("OSWORLD_INFRA_RETRIES", "0")),
        help="Retries per task after infrastructure/configuration failure; "
        "each retry provisions a fresh VM (default: "
        "OSWORLD_INFRA_RETRIES, else 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=resolve_default_model(),
        help="Free-form provider/model slug for the task agent "
        "(e.g. openai/<deployment>, "
        "openrouter/qwen/qwen3.5-397b-a17b). Defaults to "
        ".env's OPENAI_MODEL as openai/<deployment>.",
    )
    parser.add_argument(
        "--harness",
        type=str,
        default="prime_agent",
        choices=["prime_agent", "opencode"],
        help="Task-agent harness: Prime Agent (default) or the pinned OpenCode.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TASK_TIMEOUT_SECONDS,
        help=(
            "Outer timeout per task in seconds (default: "
            f"{DEFAULT_TASK_TIMEOUT_SECONDS})"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum completed model steps",
    )
    parser.add_argument(
        "--no-time-limit",
        action="store_true",
        default=False,
        help="Disable the per-task wall-clock limit",
    )

    parser.add_argument(
        "--shadow_dir",
        type=str,
        default="shadow_interface",
        help="Path to shadow interface directory",
    )
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument(
        "--test_all_meta_path",
        type=str,
        default="evaluation_examples/test_builder.json",
    )
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples"
    )
    parser.add_argument(
        "--benchmark",
        choices=["auto", "osworld", "osworld_v2", "scienceboard"],
        default="auto",
        help="Benchmark this run belongs to. The eval module already "
        "fixes the generation; this is checked against it so an "
        "osworld_v2 run cannot be dispatched to the v1 module.",
    )

    # VM config
    parser.add_argument("--provider_name", type=str, default="aws")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--client_password", type=str, default="")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--snapshot_name", type=str, default="init_state")
    parser.add_argument(
        "--use-public-ip",
        dest="use_public_ip",
        action="store_true",
        help="Reach each task VM at its public address instead of its private "
        "one. Required when the orchestrator runs outside the benchmark "
        "VPC: the provider hands back a 172.31.x.x address by default, "
        "which an off-VPC host cannot route, so every task hangs in "
        "vm_allocation until it times out. Leave off inside the VPC -- "
        "private addressing is faster and stays off the public internet. "
        "The VMs' security group must also admit this host on the "
        "control-server port; see "
        "docs/result/OSWORLD_V2_ONPREM_SECURITY_GROUP_20260903_ZH.md.",
    )
    parser.add_argument(
        "--volume_size",
        type=int,
        default=None,
        help="Default root volume GB when a task has no override",
    )
    parser.add_argument(
        "--instance-type-override",
        dest="instance_type_override",
        default=os.environ.get("OSWORLD_INSTANCE_TYPE_OVERRIDE") or None,
        help=(
            "OSWorld V2 only: override every selected task's declared AWS "
            "instance type without modifying the signed task corpus "
            "(default: OSWORLD_INSTANCE_TYPE_OVERRIDE, else the task's own)"
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Task cache root; defaults to a result-scoped directory",
    )
    proxy_group = parser.add_mutually_exclusive_group()
    proxy_group.add_argument("--enable-proxy", dest="enable_proxy", action="store_true")
    proxy_group.add_argument(
        "--disable-proxy", dest="enable_proxy", action="store_false"
    )
    parser.set_defaults(enable_proxy=None)
    parser.add_argument("--action_space", type=str, default="shadow_eval")
    parser.add_argument("--observation_type", type=str, default="screenshot")

    # Auth
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Anthropic API key (for anthropic/... models)",
    )
    parser.add_argument(
        "--openrouter_api_key",
        type=str,
        default=None,
        help="OpenRouter API key (for openrouter/... models)",
    )
    parser.add_argument(
        "--parasail_api_key",
        type=str,
        default=None,
        help="Parasail API key (for parasail/... models)",
    )

    parser.add_argument(
        "--no_tool",
        action="store_true",
        default=False,
        help="Run without deploying shadow interface (baseline)",
    )
    parser.add_argument(
        "--prompt_addendum",
        type=str,
        default="",
        help="Optional text appended to the eval-agent system "
        "prompt (TOOL_SYSTEM_PROMPT). Empty by default — "
        "evolution-time fitness eval should leave this "
        "unset; only pass it for post-hoc prompt experiments.",
    )
    parser.add_argument(
        "--or_provider",
        type=str,
        default=None,
        nargs="+",
        help=(
            "Pin OpenRouter routing to a specific provider (or ordered list). "
            "Forwarded to the Prime Agent model requests as "
            '{"provider": {"only": [<list>], "allow_fallbacks": false}}. '
            "Example: --or_provider 'Amazon Bedrock'  →  pin to Bedrock. "
            "Multiple values become an order list. No effect when not routing through OR."
        ),
    )
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser


def _resolve_opencode_config(parser, args):
    """Select the OpenCode harness and fail on a bad config before any VM."""
    args.model_cfg = OPENCODE_HARNESS_CONFIG["opencode"]
    # The model slug's first segment names the upstream provider. Each has its
    # own API-key channel; opencode.json's {env:..._API_KEY} interpolation
    # resolves to an empty string when the variable is unset, which would
    # surface as a misleading auth error at request time.
    provider_prefix = args.model.split("/", 1)[0]
    if provider_prefix == "openrouter":
        if not (args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")):
            parser.error(
                "--model openrouter/... requires OPENROUTER_API_KEY in env or "
                "--openrouter_api_key"
            )
    elif provider_prefix == "parasail":
        if not (args.parasail_api_key or os.environ.get("PARASAIL_API_KEY")):
            parser.error(
                "--model parasail/... requires PARASAIL_API_KEY in env or "
                "--parasail_api_key"
            )
    elif provider_prefix == "openai":
        if not any(
            os.environ.get(name)
            for name in (
                "SHADOW_OPENCODE_VM_BASE_URL",
                "SHADOW_OPENCODE_BASE_URL",
                "SHADOW_EVOL_VM_OPENAI_BASE_URL",
                "OPENAI_BASE_URL",
            )
        ):
            parser.error(
                "--model openai/... requires a configured OpenAI-compatible base URL"
            )
        if not os.environ.get("OPENAI_API_KEY"):
            parser.error("--model openai/... requires OPENAI_API_KEY")
    else:
        parser.error(
            f"--harness opencode --model {args.model!r}: unknown provider prefix "
            f"{provider_prefix!r}. Expected one of: openai, openrouter, parasail."
        )
    from agent_s4.harness.opencode.evaluator import (
        _resolve_runtime_cache,
        vm_config_env,
    )
    from agent_s4.harness.opencode.config import build_config, runner_settings
    from agent_s4.harness.opencode.source import validate_runtime_cache

    try:
        build_config(
            args.model,
            "",
            load_opencode_context(args.shadow_dir),
            max_steps=args.max_steps,
            env=vm_config_env(args.model),
        )
        runner_settings()
        if os.environ.get("SHADOW_EVOL_DRYRUN") != "1":
            validate_runtime_cache(_resolve_runtime_cache())
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


def resolve_model_config(parser, args):
    """Attach args.model_cfg and normalise the shared post-parse defaults."""
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be a positive integer")
    if args.infra_retries < 0:
        parser.error("--infra-retries must be non-negative")
    if not args.model:
        parser.error(
            "--model is required (or set OPENAI_MODEL and OPENAI_API_KEY in .env)"
        )
    # The pinned VM runtime archive is checked here rather than at first use:
    # a drifted pin should stop the run before any VM is provisioned, not
    # after. SHADOW_EVOL_DRYRUN skips it so offline dry runs need no archive.
    if args.harness == "opencode":
        _resolve_opencode_config(parser, args)
    else:
        if os.environ.get("SHADOW_EVOL_DRYRUN") != "1":
            from agent_s4.harness.prime.evaluator import (
                _resolve_runtime_cache,
            )
            from agent_s4.harness.prime.source import (
                validate_prime_agent_runtime_cache,
            )

            try:
                validate_prime_agent_runtime_cache(_resolve_runtime_cache())
            except RuntimeError as exc:
                parser.error(str(exc))
        args.model_cfg = PRIME_AGENT_HARNESS_CONFIG["prime_agent"]

    if args.no_tool and args.action_space == "shadow_eval":
        args.action_space = "shadow_eval_no_tool"

    if not os.path.isabs(args.shadow_dir):
        args.shadow_dir = os.path.abspath(args.shadow_dir)

    return args


def config():
    parser = argparse.ArgumentParser(
        description="Run shadow interface evaluation — deploy tool to VMs and test tasks"
    )
    add_common_args(parser)
    args = parser.parse_args()
    if args.benchmark == "scienceboard":
        parser.error(
            "--benchmark scienceboard must run through "
            "python -m agent_s4.eval.scienceboard_prime"
        )
    if args.benchmark == "osworld_v2":
        parser.error(
            "--benchmark osworld_v2 must run through "
            "python -m agent_s4.eval.osworld_v2; this module implements "
            "OSWorld v1 semantics only."
        )
    return resolve_model_config(parser, args)


def setup_logging(args):
    root_logger = logging.getLogger()
    log_level = getattr(logging, args.log_level.upper())
    root_logger.setLevel(log_level)

    datetime_str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
    os.makedirs("logs", exist_ok=True)

    fh = logging.FileHandler(f"logs/shadow-eval-{datetime_str}.log", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.DEBUG)
    sh.setLevel(log_level)

    fmt = logging.Formatter(
        "\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s "
        "\x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
    )
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    sh.addFilter(logging.Filter("desktopenv"))
    root_logger.addHandler(fh)
    root_logger.addHandler(sh)


def _raise_keyboard_interrupt(signum, frame):
    """Let worker finally-blocks close VMs when the outer harness sends TERM."""
    raise KeyboardInterrupt


def _shutdown_workers(processes, graceful_timeout=30, terminate_timeout=5):
    """Wait for worker VM cleanup, then escalate without orphaning workers.

    The outer evaluator sends SIGTERM to the complete process group, so each
    worker is already unwinding through ``worker``'s ``finally: env.close()``
    when the parent reaches this helper.  Do not immediately terminate those
    workers a second time: that could interrupt the remote close operation.
    """
    deadline = time.monotonic() + graceful_timeout
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=remaining)

    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()

    deadline = time.monotonic() + terminate_timeout
    for process in alive:
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=remaining)

    stubborn = [process for process in alive if process.is_alive()]
    for process in stubborn:
        process.kill()
    for process in stubborn:
        process.join()


def main(*, impl=None):
    """Process fan-out shared by both generations. See worker() for ``impl``."""
    impl = impl if impl is not None else sys.modules[__name__]
    args = impl.config()
    setup_logging(args)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    owner = install_owner_tag_hook()
    if owner:
        logger.info("EC2 instances will be tagged Owner=%s (terminable)", owner)
    else:
        message = (
            f"{OWNER_TAG_ENV} is unset: task VMs would launch with no Owner "
            "tag. Set it to any value: every launched instance is tagged "
            "Owner=<value>, which is how stop_evolution.sh --terminate-vms "
            "finds them (and what IAM policies that deny terminating "
            "untagged instances require)."
        )
        if args.provider_name == "aws":
            raise RuntimeError(message)
        logger.warning(message)

    model_cfg = args.model_cfg
    shadow_dir = args.shadow_dir

    if not args.no_tool and not os.path.isdir(shadow_dir):
        print(f"ERROR: Shadow interface dir not found: {shadow_dir}")
        sys.exit(1)

    # Build task index: task_id -> config file path
    task_index = impl.build_task_index(args.test_config_base_dir)
    logger.info("Task index: %d tasks found", len(task_index))

    # Resolve which tasks to run
    if args.task_ids:
        run_ids = args.task_ids
        if len(run_ids) != len(set(run_ids)):
            duplicates = sorted({tid for tid in run_ids if run_ids.count(tid) > 1})
            print(f"ERROR: Duplicate task IDs are not allowed: {duplicates}")
            sys.exit(1)
        missing = [tid for tid in run_ids if tid not in task_index]
        if missing:
            print(f"ERROR: Task IDs not found: {missing}")
            sys.exit(1)
    elif args.all:
        run_ids = impl.load_all_task_ids(args.test_all_meta_path)
        if (
            not run_ids
            or not all(isinstance(tid, str) and tid for tid in run_ids)
            or len(run_ids) != len(set(run_ids))
        ):
            print("ERROR: --all manifest must contain a non-empty unique task-ID list")
            sys.exit(1)
    else:
        print("ERROR: Provide --task_ids or --all")
        sys.exit(1)

    # Load task configs. Each is loaded here to fail fast (a broken task
    # definition or a missing WEBSITE_HOST_SUFFIX/GITLAB_* should surface
    # before we pay for VMs) but only its *path* goes on the queue: a V2
    # BaseTask comes from a module loaded by file path, which does not
    # pickle reliably across the multiprocessing queue.
    tasks = []
    for tid in run_ids:
        config_file = task_index[tid]
        try:
            example = impl.load_task(config_file, args.test_config_base_dir)
        except Exception as e:
            print(
                f"ERROR: could not load task {tid} from {config_file}: "
                f"{type(e).__name__}: {e}"
            )
            sys.exit(1)
        tasks.append((tid, config_file))

    if not tasks:
        print("ERROR: No tasks to run")
        sys.exit(1)

    tool_status = (
        "WITHOUT shadow interface" if args.no_tool else "WITH shadow interface"
    )
    print(
        f"Running {len(tasks)} tasks across {args.num_envs} VMs "
        f"with model={args.model} ({model_cfg['model_flag']}) {tool_status}"
    )

    with Manager() as manager:
        result_list = manager.list()
        task_queue = manager.Queue()
        for example_id, task_path in tasks:
            task_queue.put((example_id, task_path, 0))

        processes = []
        num_workers = min(args.num_envs, len(tasks))
        for i in range(num_workers):
            p = Process(
                target=impl.worker,
                args=(task_queue, result_list, args, shadow_dir, model_cfg),
                name=f"VM-{i + 1}",
            )
            # A daemon worker is killed abruptly as its parent exits, which
            # can skip DesktopEnv.close() and leave its EC2 instance running.
            p.daemon = False
            p.start()
            processes.append(p)
            if i < num_workers - 1:
                time.sleep(3)  # stagger VM allocation to avoid EC2 rate limits

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\nInterrupted. Cleaning up...")
            _shutdown_workers(processes)
            sys.exit(1)

        summary = print_results_summary(result_list, benchmark=args.benchmark)

    summary_path = os.path.join(args.result_dir, "latest_summary.json")
    os.makedirs(args.result_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary written to {summary_path}")
    if summary.get("infrastructure_failed", 0) or summary.get("total") != len(run_ids):
        logger.error(
            "Incomplete/invalid eval summary; exiting non-zero so callers "
            "cannot treat partial infrastructure work as fitness"
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
