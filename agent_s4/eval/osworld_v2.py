"""
Shadow Interface evaluation runner — OSWorld V2 (xlang-ai/OSWorld-V2).

Run as `python -m agent_s4.eval.osworld_v2 ...` with an OSWorld V2 clone as
cwd, so `desktop_env` resolves to V2's rather than the editable-installed v1
fork. The task definitions come from OSWorldV2_Base/evaluation_examples, which
is the only checkout holding the gated task_class files.

Everything generation-neutral — prompts, agent-command and episode plumbing,
result persistence, the worker/main fan-out, the shared CLI surface — is
imported from agent_s4.eval.osworld. This module holds what V2 actually
changes:

  - tasks are Python BaseTask subclasses under task_class/, loaded through
    V2's own task_loader rather than json.load
  - a task declares its own VM: platform, image, instance_type, volume_size,
    proxy, and the vnc/recording toggles
  - evaluate() may return a dict of partial-credit and safety detail, not
    just a float
  - a task may be multi-phase, with per-phase setup, instruction and
    evaluation, and gates that stop the run early
  - a task may carry a user simulator the agent can question through an
    ASK_USER protocol

Usage:
    python -m agent_s4.eval.osworld_v2 \
        --task_ids 017 --num_envs 1 --harness prime_agent \
        --model openai/<deployment> \
        --test_config_base_dir <OSWorldV2_Base>/evaluation_examples
"""

import argparse
import json
import logging
import os
import re
import sys
import time

# Bound here so osworld.worker(impl=this module) constructs V2's DesktopEnv,
# and so tests can substitute it on this module rather than on osworld.
from desktop_env.desktop_env import DesktopEnv

from agent_s4.eval import grader_image_limit
from agent_s4.eval import osworld
from agent_s4.eval.osworld import (
    EnvSpec,
    HarnessFailure,
    add_common_args,
    build_env_kwargs as _base_env_kwargs,
    finalize_task_result,
    persist_evaluation_result,
    prepare_task_vm,
    resolve_cache_dir,
    resolve_model_config,
    _run_agent_episode,
    _task_get,
)

logger = logging.getLogger("desktopenv.shadow_eval")

grader_image_limit.apply()

DEFAULT_MAX_USER_TURNS = 5

# The V2 AMI ships a 29 GB root of which snapd and flatpak alone occupy ~13 GB,
# leaving under 3 GB free before we add anything. That is not enough for the
# agent runtime, its dependencies and a screen recording, and a full disk
# corrupts the agent's exit-code marker rather than failing loudly. Only 18 of
# the 108 tasks declare a volume_size, so the remaining 90 would otherwise boot
# onto a root that is already full.
DEFAULT_VOLUME_SIZE_GB = 60
# Below this the runtime fits but leaves little room for task artefacts.
MIN_COMFORTABLE_VOLUME_SIZE_GB = 40

HITL_SYSTEM_PROMPT_SUFFIX = """\
## Asking the simulated user

This task provides a benchmark user simulator. If a required fact is genuinely
ambiguous, finish your current episode with exactly one line in this form:

    ASK_USER: <one concise question>

The harness will return the simulated user's answer and start a continuation
episode on the same VM. Do not search the filesystem or task configuration for
hidden user knowledge.
"""


# ---------------------------------------------------------------------------
# Task-declared VM specification
# ---------------------------------------------------------------------------


def _platform_to_os_type(platform: str | None) -> str:
    return (
        "Windows" if str(platform or "").strip().lower().startswith("win") else "Ubuntu"
    )


def _resolve_volume_size(example, args) -> int | None:
    """Root volume GB for this task: task declaration, then CLI, then default.

    A task's own declaration stays authoritative -- upstream sized it -- so a
    small value is warned about rather than silently raised.
    """
    task_volume = _task_get(example, "volume_size")
    if task_volume is not None:
        source, volume_size = "task", int(task_volume)
    else:
        cli_volume = getattr(args, "volume_size", None)
        if cli_volume is not None:
            source, volume_size = "--volume_size", int(cli_volume)
        else:
            source, volume_size = "harness default", DEFAULT_VOLUME_SIZE_GB
    if volume_size < MIN_COMFORTABLE_VOLUME_SIZE_GB:
        logger.warning(
            "Root volume is %dGB (from %s); the V2 AMI uses ~26GB before the "
            "agent runtime is installed, so this may run out of disk",
            volume_size,
            source,
        )
    return volume_size


def build_env_spec(example, args) -> EnvSpec:
    """Resolve the VM spec *after* loading a V2 task.

    A V2 task class may override the platform, the image, the instance type
    and the root volume, and declares whether it needs the proxy. The VM must
    therefore be built per task rather than once per worker.
    """
    screen_size = (args.screen_width, args.screen_height)
    os_type = _platform_to_os_type(_task_get(example, "platform"))
    snapshot_name = getattr(args, "snapshot_name", "init_state")
    if args.provider_name == "aws":
        image = _task_get(example, "image")
        if image:
            snapshot_name = image
        else:
            from desktop_env.providers.aws.manager import get_default_ami_id

            snapshot_name = get_default_ami_id(
                region=args.region,
                screen_size=screen_size,
                os_type=os_type,
            )

    # The official runner enables the proxy and lets each task's own `proxy`
    # field decide whether to route through it, so default to on.
    proxy_override = getattr(args, "enable_proxy", None)
    enable_proxy = True if proxy_override is None else bool(proxy_override)
    volume_size = _resolve_volume_size(example, args)
    instance_type = getattr(args, "instance_type_override", None) or _task_get(
        example, "instance_type"
    )
    return EnvSpec(
        os_type=os_type,
        snapshot_name=snapshot_name,
        instance_type=instance_type,
        volume_size=volume_size,
        enable_proxy=enable_proxy,
        force_disable_vnc=bool(_task_get(example, "disable_vnc", False)),
        force_disable_recording=bool(_task_get(example, "disable_recording", False)),
        cache_dir=resolve_cache_dir(example, args),
        is_v2=True,
    )


def build_env_kwargs(spec: EnvSpec, args) -> dict:
    kwargs = _base_env_kwargs(spec, args)
    # These kwargs exist only in OSWorld V2's DesktopEnv.
    kwargs.update(
        instance_type=spec.instance_type,
        volume_size=spec.volume_size,
        force_disable_vnc=spec.force_disable_vnc,
        force_disable_recording=spec.force_disable_recording,
        # The provider returns the private address unless asked otherwise, so
        # an orchestrator outside the VPC has to opt in or it never reaches a
        # single VM: the connections sit in SYN-SENT, setup_controller never
        # reports ready, and every task fails in vm_allocation after burning
        # its retries. Default stays private for in-VPC runs.
        use_public_ip=bool(getattr(args, "use_public_ip", False)),
    )
    return kwargs


# ---------------------------------------------------------------------------
# Task discovery and loading
# ---------------------------------------------------------------------------


def build_task_index(test_config_base_dir):
    """Build task_id -> definition-file path over both V2 task corpora.

    Two corpora live under one evaluation_examples/ tree:
      - examples/<domain>/<uuid>.json — the v1-style corpus, which V2 also
        ships in a larger, patched form.
      - task_class/task_<NNN>.py      — V2-native tasks, one BaseTask
        subclass per file. The id is the filename suffix ('task_001.py' ->
        '001'), which matches the ids listed in test_v2.json.

    IDs never collide (UUIDs vs. zero-padded ints), so one flat index serves
    both; load_task() dispatches on the file extension.
    """
    index = osworld.build_task_index(test_config_base_dir)

    task_class_dir = os.path.join(test_config_base_dir, "task_class")
    if os.path.isdir(task_class_dir):
        for fname in sorted(os.listdir(task_class_dir)):
            # task_<id>.py only — skips __init__.py and generated_task_utils.py
            if not (fname.startswith("task_") and fname.endswith(".py")):
                continue
            task_id = fname[len("task_") : -len(".py")]
            index[task_id] = os.path.join(task_class_dir, fname)
    from agent_s4.synthetic_tasks import extend_task_index

    extend_task_index(index)
    return index


_TASK_LOADER = None


def _task_loader(test_config_base_dir):
    """Import OSWorld V2's task_loader module, cached.

    It sits at the benchmark repo root (the parent of evaluation_examples),
    and we load it by path rather than plain `import task_loader` because the
    eval subprocess's cwd differs by caller — the orchestrator runs it inside
    an island clone, the held-out scripts from the repo root.
    """
    global _TASK_LOADER
    if _TASK_LOADER is None:
        import importlib.util

        repo_root = os.path.dirname(os.path.abspath(test_config_base_dir))
        loader_path = os.path.join(repo_root, "task_loader.py")
        if not os.path.isfile(loader_path):
            raise RuntimeError(
                f"{loader_path} not found — V2-native (.py) tasks need "
                f"OSWorld V2's task_loader at the benchmark repo root."
            )
        spec = importlib.util.spec_from_file_location(
            "osworld_task_loader", loader_path
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["osworld_task_loader"] = mod
        try:
            spec.loader.exec_module(mod)
        except ModuleNotFoundError as e:
            if "task_base" not in str(e):
                raise
            # `desktop_env` resolved to the v1 fork, which has no task_base.
            # The eval imports desktop_env by name, so its cwd picks the
            # generation — both evaluator harnesses run it with cwd=clone_dir.
            import desktop_env

            raise RuntimeError(
                f"V2-native tasks need OSWorld V2's desktop_env, but "
                f"desktop_env resolved to {desktop_env.__file__} (no "
                f"task_base module). Run the eval with an OSWorld V2 clone as "
                f"cwd — its clones are OSWorldV2_Evol_<island>/."
            ) from e
        _TASK_LOADER = mod
    return _TASK_LOADER


def load_task(task_path, test_config_base_dir):
    """Load one task definition: a V2 BaseTask instance, or a v1 JSON dict.

    BaseTask subclasses dict, so callers treat both the same — `task["id"]`,
    `task.get(...)`, and `DesktopEnv.reset(task_config=...)` all work
    unchanged. V2 task modules read env vars at import time
    (WEBSITE_HOST_SUFFIX for website tasks, GITLAB_* for the GitLab ones), so
    a missing one surfaces here as an ImportError-ish failure.
    """
    if task_path.endswith(".py"):
        return _task_loader(test_config_base_dir).load_task_from_file(task_path)
    from agent_s4.synthetic_tasks import prepare_loaded_task

    return prepare_loaded_task(
        osworld.load_task(task_path, test_config_base_dir), task_path
    )


def load_all_task_ids(test_all_meta_path):
    """Load all task IDs from a manifest.

    Handles both shapes: v1's domain -> [ids] mapping (test_all.json), and
    V2's {"tasks": [ids]} (test_v2.json).
    """
    with open(test_all_meta_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return list(data)
    if set(data) == {"tasks"}:
        return list(data["tasks"])
    all_ids = []
    for task_ids in data.values():
        all_ids.extend(task_ids)
    return all_ids


# ---------------------------------------------------------------------------
# Instructions and the human-in-the-loop protocol
# ---------------------------------------------------------------------------


def _instruction_for_task(example, instruction: str) -> str:
    task_date = _task_get(example, "task_current_date")
    date_context = (
        f"\n\nBenchmark task date: {task_date}. Treat this as the current date for this task."
        if task_date
        else ""
    )
    context_suffix = osworld.INSTRUCTION_CONTEXT_SUFFIX
    if _task_get(example, "user_simulator"):
        context_suffix = context_suffix.replace(
            " Do not ask user any follow-up questions.",
            "",
        )
    return instruction + date_context + context_suffix


def _extract_ask_user(payload: bytes | str | None) -> str | None:
    """Extract the last `ASK_USER:` sentinel from an agent JSONL stream."""
    if not payload:
        return None
    text = (
        payload.decode("utf-8", errors="replace")
        if isinstance(payload, bytes)
        else payload
    )
    candidate_texts = []
    # split("\n"): splitlines() also breaks on U+0085 inside JSON strings.
    for line in text.split("\n"):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            candidate_texts.append(line)
            continue
        stack = [event]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, str):
                candidate_texts.append(value)
    for candidate in reversed(candidate_texts):
        match = re.search(r"(?:^|\n)\s*ASK_USER:\s*(.+?)\s*$", candidate, re.I | re.S)
        if match:
            return match.group(1).strip()
    return None


LENGTH_CONTINUATION_PROMPT = (
    "Your previous reply was cut off because it reached the output-token "
    "limit, and it called no tool, so none of that reasoning was acted on. "
    "The task is NOT finished. Do not restate the analysis. Continue the task "
    "now on the same VM by calling a tool; keep any further reasoning short "
    "and check your conclusions by acting and inspecting the result."
)


def _length_truncated_without_tool(payload: bytes | str | None) -> bool:
    """True when the episode's final assistant message hit the output cap
    (stopReason=length) without calling a tool.

    Prime's loop ends on any reply that carries no tool call, so such an
    episode stops mid-thought while looking like a normal finish.
    """
    if not payload:
        return False
    text = (
        payload.decode("utf-8", errors="replace")
        if isinstance(payload, bytes)
        else payload
    )
    last = None
    # split("\n"), not splitlines(): tool output can hold U+0085, which
    # splitlines() treats as a line break inside a JSON string.
    for line in text.split("\n"):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        # The final truncated reply is not always followed by a message_end;
        # turn_end carries the same assistant message.
        if not isinstance(event, dict) or event.get("type") not in {
            "message_end",
            "turn_end",
        }:
            continue
        message = event.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant":
            last = message
    if last is None or last.get("stopReason") != "length":
        return False
    return not any(
        isinstance(part, dict) and part.get("type") == "toolCall"
        for part in last.get("content") or []
    )


def _max_length_continuations() -> int:
    """Opt-in via SHADOW_EVOL_LENGTH_CONTINUATIONS; 0 keeps the old behaviour."""
    raw = os.environ.get("SHADOW_EVOL_LENGTH_CONTINUATIONS", "").strip()
    value = int(raw) if raw else 0
    if value < 0:
        raise ValueError(f"SHADOW_EVOL_LENGTH_CONTINUATIONS must be >= 0, got {raw!r}")
    return value


def _make_ask_user_bridge(env, args, example_result_dir, hitl_history):
    """Build the ASK_USER follow-up hook for _run_agent_episode.

    Returns continuation text carrying the simulated user's answer, or None
    when the agent asked nothing and the episode is done. With
    SHADOW_EVOL_LENGTH_CONTINUATIONS > 0 it also resumes an episode whose
    last reply was cut off at the output cap, up to that many times.
    """
    simulator = getattr(env, "user_simulator", None)
    max_user_turns = getattr(args, "max_user_turns", DEFAULT_MAX_USER_TURNS)
    max_length_continuations = _max_length_continuations()
    length_continuations = []

    def ask_user(trajectory, attempt, phase_index):
        if len(
            length_continuations
        ) < max_length_continuations and _length_truncated_without_tool(trajectory):
            record = {
                "phase_index": phase_index,
                "attempt": attempt,
                "continuation": len(length_continuations) + 1,
                "limit": max_length_continuations,
            }
            length_continuations.append(record)
            with open(
                os.path.join(example_result_dir, "length_continuations.jsonl"),
                "a",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(record) + "\n")
            logger.warning(
                "Episode ended on a length-truncated reply with no tool call; "
                "resuming (%d/%d)",
                record["continuation"],
                max_length_continuations,
            )
            return LENGTH_CONTINUATION_PROMPT
        question = _extract_ask_user(trajectory)
        if not question:
            return None
        if simulator is None:
            raise HarnessFailure(
                "agent requested user input but this task has no user_simulator",
                "agent_protocol_failure",
            )
        if attempt > max_user_turns:
            raise HarnessFailure(
                f"agent exceeded max user-simulator turns ({max_user_turns})",
                "agent_protocol_failure",
            )
        try:
            answer = simulator.respond(question)
        except Exception as exc:
            raise HarnessFailure(
                f"user simulator failed: {type(exc).__name__}: {exc}",
                "user_simulator_failure",
            ) from exc
        record = {
            "phase_index": phase_index,
            "turn": attempt,
            "question": question,
            "answer": answer,
        }
        hitl_history.append(record)
        with open(
            os.path.join(example_result_dir, "hitl.jsonl"), "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return (
            "\n\nYou previously asked the simulated user:\n"
            f"{question}\n\nThe simulated user answered:\n{answer}\n\n"
            "Continue the task on the same VM. Ask again only if another "
            "required fact remains genuinely ambiguous."
        )

    return ask_user


def _apply_hitl_prompt(system_prompt: str) -> str:
    return (
        system_prompt.replace(
            "- NEVER ask the user for file paths, URLs, or clarification. FIND what you need.",
            "- Ask a question only through the benchmark ASK_USER protocol below.",
        )
        + "\n\n"
        + HITL_SYSTEM_PROMPT_SUFFIX
    )


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------


def _run_phases(
    env,
    example,
    phases,
    args,
    model_cfg,
    example_result_dir,
    *,
    system_prompt,
    artifact_chunks,
    ask_user,
    budget=None,
):
    """Run a multi-phase task in order on one VM. Returns (score, details).

    ``budget`` is the task's wall clock, shared across the phases: a task's
    cap has to bound the task, not each of its phases. A phase that starts
    with the budget already spent runs no agent, and is still evaluated and
    scored -- its setup ran, so the desktop state it produced is real.
    """
    total_score = 0.0
    phase_results = []
    use_proxy = bool(_task_get(example, "proxy", False) and env.enable_proxy)

    for phase_index, phase in enumerate(phases, start=1):
        phase_instruction = _instruction_for_task(example, phase["instruction"])
        if phase_index > 1:
            phase["setup"](env.setup_controller, use_proxy=use_proxy)
            env.is_environment_used = True
            pause = phase.get("pause_after_setup_seconds", 5)
            if pause:
                time.sleep(pause)
        env.instruction = phase["instruction"]
        _run_agent_episode(
            env,
            args,
            model_cfg,
            phase_instruction,
            system_prompt,
            example_result_dir,
            phase_index=phase_index,
            artifact_chunks=artifact_chunks,
            ask_user=ask_user,
            budget=budget,
        )
        env.action_history = ["DONE"]
        env.is_environment_used = True
        time.sleep(20)
        raw_phase_result = phase["evaluate"](env)
        phase_score, phase_details = persist_evaluation_result(
            raw_phase_result,
            example_result_dir,
            suffix=f"_phase_{phase_index}",
        )
        total_score += phase_score
        phase_results.append(
            {
                "phase_index": phase_index,
                "phase_name": phase.get("name", f"Phase {phase_index}"),
                "instruction": phase["instruction"],
                "score": phase_score,
                "details": phase_details,
            }
        )
        gate_min = phase.get("gate_min_score")
        if gate_min is not None and phase_score < float(gate_min):
            break
        if phase.get("gate") and phase_score <= 0.0:
            break

    score = round(max(0.0, min(1.0, total_score)), 4)
    persist_evaluation_result(score, example_result_dir)
    with open(
        os.path.join(example_result_dir, "phase_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(phase_results, f, indent=2, ensure_ascii=False, default=str)
    result_details = {"score": score, "phases": phase_results}
    with open(
        os.path.join(example_result_dir, "result.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(result_details, f, indent=2, ensure_ascii=False, default=str)
    return score, result_details


def run_single_task(
    env, example, instruction, example_result_dir, args, model_cfg, shadow_dir
):
    """Run a single V2 task: reset VM, deploy tool, run agent(s), evaluate."""
    start_time = time.time()
    example_id = example.get("id", "unknown")
    base_instruction = _instruction_for_task(example, instruction)

    staged = prepare_task_vm(env, example, args, model_cfg, shadow_dir)
    if isinstance(staged, dict):
        return staged
    system_prompt, recording_enabled = staged

    if getattr(env, "user_simulator", None) is not None:
        system_prompt = _apply_hitl_prompt(system_prompt)

    phases_getter = getattr(example, "get_phases", None)
    phases = phases_getter() if callable(phases_getter) else []
    if phases and not isinstance(phases, list):
        raise TypeError("get_phases() must return a list")

    artifact_chunks = []
    hitl_history = []
    ask_user = _make_ask_user_bridge(env, args, example_result_dir, hitl_history)

    # The task's clock starts here, not at the top of run_single_task: every
    # line above is VM allocation, reset and interface deployment, whose cost
    # belongs to the infrastructure rather than to the agent's budget.
    budget = osworld.TaskBudget(
        None if getattr(args, "no_time_limit", False) else args.timeout
    ).start()

    if phases:
        score, result_details = _run_phases(
            env,
            example,
            phases,
            args,
            model_cfg,
            example_result_dir,
            system_prompt=system_prompt,
            artifact_chunks=artifact_chunks,
            ask_user=ask_user,
            budget=budget,
        )
    else:
        _run_agent_episode(
            env,
            args,
            model_cfg,
            base_instruction,
            system_prompt,
            example_result_dir,
            artifact_chunks=artifact_chunks,
            ask_user=ask_user,
            budget=budget,
        )
        infeasible_data = env.controller.get_file("/tmp/INFEASIBLE")
        env.action_history = ["FAIL"] if infeasible_data is not None else ["DONE"]
        env.is_environment_used = True
        time.sleep(20)
        raw_result = env.evaluate()
        if example.get("verification") == "dummy_constant_zero":
            raw_result = {
                "score": float(raw_result),
                "verification": "dummy_constant_zero",
                "score_semantics": "UNGRADED: constant zero; use trajectories, not scores",
            }
        score, result_details = persist_evaluation_result(
            raw_result, example_result_dir
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
        phase_count=len(phases) if phases else 1,
        hitl_turns=len(hitl_history),
        recording_enabled=recording_enabled,
    )


# ---------------------------------------------------------------------------
# CLI and process fan-out
# ---------------------------------------------------------------------------


def config():
    parser = argparse.ArgumentParser(
        description="Run shadow interface evaluation against OSWorld V2 tasks"
    )
    add_common_args(parser)
    parser.add_argument(
        "--max-user-turns",
        type=int,
        default=DEFAULT_MAX_USER_TURNS,
        help="Maximum ASK_USER continuations for HITL tasks",
    )
    args = parser.parse_args()
    if args.benchmark in {"osworld", "scienceboard"}:
        parser.error(
            f"--benchmark {args.benchmark} must run through python -m "
            + {
                "osworld": "agent_s4.eval.osworld",
                "scienceboard": "agent_s4.eval.scienceboard_prime",
            }[args.benchmark]
            + "; this module implements OSWorld V2 semantics only."
        )
    return resolve_model_config(parser, args)


def worker(task_queue, result_list, args, shadow_dir, model_cfg):
    return osworld.worker(
        task_queue,
        result_list,
        args,
        shadow_dir,
        model_cfg,
        impl=sys.modules[__name__],
    )


def main():
    return osworld.main(impl=sys.modules[__name__])


if __name__ == "__main__":
    main()
