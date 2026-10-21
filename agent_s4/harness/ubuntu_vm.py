"""UbuntuVMEvaluator — subprocess wrapper around the shared eval worker.

The class drives the task agent (Prime Agent or OpenCode) inside VMs
provisioned per task (OSWorld V2 via its DesktopEnv, ScienceBoard via its AMI;
ScienceBoard is Prime-only). The installer and build_command come from
agent_s4.harness.prime.evaluator / .opencode; this class is
benchmark-bound (VM + scoring), not model-bound.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path

from agent_s4.benchmark_spec import BENCHMARKS
from agent_s4.harness.prime.openrouter import _parse_bool_env
from agent_s4.harness.base import EvaluatorHarness, EvalResults
from agent_s4.scoring import canonicalize_full_credit, is_binary_success
from agent_s4.vm.owner_tag import OWNER_TAG_ENV

logger = logging.getLogger("agent_s4.harness.ubuntu_vm")

_EVAL_SHUTDOWN_GRACE_SECONDS = 45
# Each OSWorld generation has its own eval module: v1 semantics in
# eval/osworld.py, V2's task classes, per-task VM specs, multi-phase runs and
# HITL protocol in eval/osworld_v2.py.
EVAL_MODULE_BY_BENCHMARK = {
    "osworld": "agent_s4.eval.osworld",
    "osworld_v2": "agent_s4.eval.osworld_v2",
    "scienceboard": "agent_s4.eval.scienceboard_prime",
}
# Benchmarks whose eval module launches tagged instances that this harness
# must sweep by run name (AWS_INSTANCE_NAME) after the batch.
_RUN_NAME_TAGGED = {"osworld_v2", "scienceboard"}

_CLEANUP_RECORD_RE = re.compile(r"^vm_cleanup\.(shadow-evol-[A-Za-z0-9-]+)\.json$")


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Durably replace a lifecycle record without a truncated-JSON window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _run_eval_subprocess(cmd, *, cwd, capture_output, text, timeout, env=None):
    """Run eval in its own process group and terminate the whole group on timeout."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=text,
        start_new_session=True,
        env=env,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            # The child SIGTERM handler unwinds every worker through
            # ``env.close()``. Give remote VM cleanup time before SIGKILL.
            proc.wait(timeout=_EVAL_SHUTDOWN_GRACE_SECONDS)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _active_v2_instance_ids(client, run_name: str) -> list[str]:
    return _v2_instance_ids_by_state(
        client,
        run_name,
        ("pending", "running", "stopping", "stopped"),
    )


def _v2_instance_ids_by_state(
    client,
    run_name: str,
    states: tuple[str, ...],
) -> list[str]:
    filters = [
        {"Name": "tag:Name", "Values": [run_name]},
        {"Name": "instance-state-name", "Values": list(states)},
    ]
    owner = os.environ.get(OWNER_TAG_ENV, "").strip()
    if owner:
        filters.append({"Name": "tag:Owner", "Values": [owner]})
    instance_ids = []
    next_token = None
    while True:
        kwargs = {"Filters": filters}
        if next_token:
            kwargs["NextToken"] = next_token
        response = client.describe_instances(**kwargs)
        instance_ids.extend(
            instance["InstanceId"]
            for reservation in response.get("Reservations", [])
            for instance in reservation.get("Instances", [])
        )
        next_token = response.get("NextToken")
        if not next_token:
            return instance_ids


def _cleanup_v2_instances(
    region: str, run_name: str, *, attempts=4, retry_delay=2.0
) -> list[str]:
    """Clean up and re-check every active EC2 instance tagged for one eval.

    The default remains termination. Restricted scheduler roles may explicitly
    select ``SHADOW_EVOL_V2_CLEANUP_ACTION=stop``; that removes compute cost but
    deliberately leaves the instance and its EBS volumes allocated.
    """
    import boto3

    client = boto3.client("ec2", region_name=region)
    cleanup_action = os.environ.get(
        "SHADOW_EVOL_V2_CLEANUP_ACTION",
        "terminate",
    )
    if cleanup_action not in {"terminate", "stop"}:
        raise ValueError(
            "SHADOW_EVOL_V2_CLEANUP_ACTION must be 'terminate' or 'stop', "
            f"got {cleanup_action!r}"
        )
    if cleanup_action == "stop":
        active = _active_v2_instance_ids(client, run_name)
        if not active:
            return []
        pending = _v2_instance_ids_by_state(client, run_name, ("pending",))
        if pending:
            client.get_waiter("instance_running").wait(
                InstanceIds=pending,
                WaiterConfig={"Delay": 5, "MaxAttempts": 60},
            )
        running = _v2_instance_ids_by_state(client, run_name, ("running",))
        if running:
            client.stop_instances(InstanceIds=running)
        client.get_waiter("instance_stopped").wait(
            InstanceIds=active,
            WaiterConfig={"Delay": 5, "MaxAttempts": 120},
        )
        remaining = _v2_instance_ids_by_state(
            client,
            run_name,
            ("pending", "running", "stopping"),
        )
        if remaining:
            raise RuntimeError(
                f"tagged instances remain compute-active after stop: {remaining}"
            )
        logger.warning(
            "V2 cleanup stopped %d tagged instance(s): %s. EBS volumes remain.",
            len(active),
            active,
        )
        return sorted(active)

    terminated = set()
    for attempt in range(attempts):
        active = _active_v2_instance_ids(client, run_name)
        if active:
            client.terminate_instances(InstanceIds=active)
            terminated.update(active)
            logger.warning(
                "V2 cleanup attempt %d terminated %d tagged instance(s): %s",
                attempt + 1,
                len(active),
                active,
            )
        if attempt < attempts - 1:
            time.sleep(retry_delay)
    remaining = _active_v2_instance_ids(client, run_name)
    if remaining:
        raise RuntimeError(f"tagged instances remain active after cleanup: {remaining}")
    return sorted(terminated)


def _recover_v2_cleanup_records(result_dir: Path, region: str) -> None:
    """Sweep pending/failed tagged runs left by a prior host crash."""
    for path in sorted(result_dir.glob("vm_cleanup*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            # The unique AWS Name tag is encoded in the filename. A crash in
            # the old non-atomic writer must not permanently disable recovery.
            match = _CLEANUP_RECORD_RE.fullmatch(path.name)
            if match is None:
                continue
            record = {
                "run_name": match.group(1),
                "region": region,
                "cleanup_status": "malformed",
            }
        if record.get("cleanup_status") == "complete" or not record.get("run_name"):
            continue
        terminated = _cleanup_v2_instances(region, record["run_name"])
        id_field = (
            "stopped_instance_ids"
            if os.environ.get("SHADOW_EVOL_V2_CLEANUP_ACTION") == "stop"
            else "terminated_instance_ids"
        )
        record.update(cleanup_status="complete", recovered_on_next_run=True)
        record[id_field] = terminated
        _atomic_write_json(path, record)


def _validate_summary_aggregates(
    summary: dict,
    task_ids: list[str],
    *,
    benchmark: str = "osworld",
) -> None:
    required = {"pass_rate", "total", "passed", "failed", "results"}
    missing = sorted(required - set(summary))
    if missing:
        raise RuntimeError(f"Eval summary is missing required keys: {missing}")
    results = summary["results"]
    if not isinstance(results, list):
        raise RuntimeError("Eval summary results must be a list")
    if summary["total"] != len(task_ids):
        raise RuntimeError(
            f"Eval returned {summary['total']} result(s) for "
            f"{len(task_ids)} requested task(s); refusing partial fitness"
        )
    returned_ids = [result.get("task_id") for result in results]
    if Counter(returned_ids) != Counter(task_ids):
        raise RuntimeError(
            "Eval summary task IDs do not match the requested tasks; "
            f"requested={task_ids}, returned={returned_ids}. Refusing "
            "stale, duplicate, or cross-run fitness"
        )

    invalid = [
        result for result in results if not result.get("valid_for_fitness", True)
    ]
    valid = [result for result in results if result.get("valid_for_fitness", True)]

    def passed(result):
        if "score" in result:
            return is_binary_success(result["score"], benchmark)
        if "passed" in result:
            return bool(result["passed"])
        raise RuntimeError(
            f"Eval result lacks score/passed for task {result.get('task_id')!r}"
        )

    computed_passed = sum(passed(result) for result in valid)
    computed_failed = len(valid) - computed_passed
    expected = (
        computed_passed,
        computed_failed,
        len(invalid),
        len(results),
    )
    reported = (
        summary["passed"],
        summary["failed"],
        summary.get("infrastructure_failed", 0),
        summary["total"],
    )
    if reported != expected:
        raise RuntimeError(
            "Eval summary aggregates contradict per-task results: "
            f"reported={reported}, computed={expected}"
        )
    expected_rate = computed_passed / len(valid) if valid else 0.0
    if abs(float(summary["pass_rate"]) - expected_rate) > 1e-9:
        raise RuntimeError(
            f"Eval summary pass_rate={summary['pass_rate']} contradicts "
            f"per-task pass_rate={expected_rate}"
        )


def _eval_results_from_summary(
    summary: dict,
    task_ids: list[str],
    *,
    benchmark: str = "osworld",
) -> EvalResults:
    """Validate one complete summary and convert it to the harness result."""
    _validate_summary_aggregates(summary, task_ids, benchmark=benchmark)
    if summary.get("infrastructure_failed", 0):
        failures = [
            result
            for result in summary.get("results", [])
            if not result.get("valid_for_fitness", True)
        ]
        raise RuntimeError(
            f"Eval had {summary['infrastructure_failed']} infrastructure/config "
            f"failure(s); refusing to record them as interface fitness: {failures}"
        )
    return EvalResults(
        pass_rate=summary["pass_rate"],
        total=summary["total"],
        passed=summary["passed"],
        failed=summary["failed"],
        results=summary.get("results", []),
        timed_out=False,
    )


def _is_total_washout(summary: dict) -> bool:
    """True when a summary records no genuine outcome at all.

    Deliberately narrow: a single real result -- even a zero that the grader
    actually produced -- makes the cache trustworthy again. Only a clean sweep
    of imputed zeros is treated as evidence that the evaluation never really
    happened.
    """
    results = summary.get("results")
    if not results:
        return False
    return all(entry.get("imputed_zero") for entry in results)


def _load_complete_cached_summary(
    summary_path: Path,
    task_ids: list[str],
    *,
    benchmark: str = "osworld",
) -> EvalResults | None:
    """Return a trustworthy completed cache, or ``None`` if it is unusable."""
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text())
        if not summary:
            return None
        if _is_total_washout(summary):
            # Every task an imputed zero and not one genuine result. A real
            # evaluation essentially never looks like this; a broken model
            # endpoint does, and an evaluator orphaned by a dying driver
            # writes exactly this before exiting. Reusing it silently hands
            # the variant a fabricated 0.0 that then drives selection, so
            # re-evaluate instead.
            logger.warning(
                "Ignoring cached eval summary %s: all %d task(s) are imputed "
                "zeros with no genuine result, which is not a trustworthy "
                "completion",
                summary_path,
                len(summary.get("results") or []),
            )
            return None
        return _eval_results_from_summary(
            summary,
            task_ids,
            benchmark=benchmark,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError, RuntimeError) as exc:
        logger.warning(
            "Ignoring unusable cached eval summary %s: %s", summary_path, exc
        )
        return None


def _recover_summary_from_task_results(
    result_dir: Path,
    task_ids: list[str],
    model: str,
    *,
    benchmark: str = "osworld",
) -> dict | None:
    """Rebuild a lost summary when every authoritative task leaf is complete.

    This is the crash window where the raw evaluator finished all tasks but its
    parent driver died before ingesting ``latest_summary.json``. A task is
    recoverable only when ``result.txt`` exists and any accompanying
    ``failure.json`` is already valid for fitness (for example, an exhausted
    retry recorded as an imputed zero). One missing task or unresolved infra
    failure makes the whole cache ineligible and forces a normal evaluation.
    """
    model_parts = [part for part in model.split("/") if part]
    leaf_root = result_dir / "shadow_eval" / "screenshot" / Path(*model_parts)
    results = []
    for task_id in task_ids:
        task_dir = leaf_root / task_id
        score_path = task_dir / "result.txt"
        try:
            score = float(score_path.read_text().strip())
        except (OSError, ValueError):
            return None
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            return None
        if benchmark == "osworld_v2":
            score = canonicalize_full_credit(score)

        result = {
            "task_id": task_id,
            "score": score,
            "elapsed": 0.0,
            "valid_for_fitness": True,
        }
        failure_path = task_dir / "failure.json"
        if failure_path.is_file():
            try:
                failure = json.loads(failure_path.read_text())
            except (OSError, json.JSONDecodeError, TypeError):
                return None
            if not isinstance(failure, dict) or not failure.get(
                "valid_for_fitness", True
            ):
                return None
            try:
                failure_score = float(failure.get("score", score))
            except (TypeError, ValueError):
                return None
            if not math.isclose(failure_score, score, rel_tol=0.0, abs_tol=1e-12):
                return None
            for key in (
                "attempt",
                "max_attempts",
                "imputed_zero",
                "failure_type",
                "error",
                "failure_summary",
            ):
                if key in failure:
                    result[key] = failure[key]
        results.append(result)

    passed = sum(is_binary_success(result["score"], benchmark) for result in results)
    summary = {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "infrastructure_failed": 0,
        "pass_rate": passed / len(results) if results else 0.0,
        "mean_score": (
            sum(result["score"] for result in results) / len(results)
            if results
            else 0.0
        ),
        "results": results,
        "recovered_from_task_results": True,
    }
    _validate_summary_aggregates(summary, task_ids, benchmark=benchmark)
    return summary


class UbuntuVMEvaluator(EvaluatorHarness):
    """Evaluator harness for agents running inside Ubuntu eval VMs.

    Shells out to the eval module for the requested benchmark, which
    provisions VMs and drives Prime Agent. Reads
    latest_summary.json after the
    subprocess completes and returns aggregated results.

    `benchmark` selects two things: the *_Base task tree, and which OSWorld
    generation's eval module runs (EVAL_MODULE_BY_BENCHMARK).
    """

    def run_tasks(
        self,
        *,
        clone_dir: Path,
        shadow_dir: Path,
        result_dir: Path,
        task_ids: list[str],
        model: str,
        region: str,
        timeout: int,
        max_steps: int | None = None,
        no_time_limit: bool = False,
        harness: str | None = None,
        benchmark: str = "osworld",
    ) -> EvalResults:
        if harness not in (None, "prime_agent", "opencode"):
            raise ValueError(
                "UbuntuVMEvaluator supports prime_agent and opencode, got "
                f"{harness!r}"
            )
        if benchmark == "scienceboard" and harness == "opencode":
            raise ValueError("ScienceBoard runs support only harness='prime_agent'")
        if harness == "prime_agent" and os.environ.get("SHADOW_EVOL_DRYRUN") != "1":
            # Validate the pin before provisioning: a missing runtime archive
            # would otherwise fail once per VM, after the instances are
            # already running and billable.
            if benchmark not in EVAL_MODULE_BY_BENCHMARK:
                raise ValueError(
                    "UbuntuVMEvaluator has no eval module for benchmark "
                    f"{benchmark!r}"
                )
            from agent_s4.harness.prime.evaluator import (
                _resolve_runtime_cache,
            )
            from agent_s4.harness.prime.source import (
                validate_prime_agent_runtime_cache,
            )

            validate_prime_agent_runtime_cache(_resolve_runtime_cache())

        if harness == "opencode" and os.environ.get("SHADOW_EVOL_DRYRUN") != "1":
            from agent_s4.harness.opencode.evaluator import (
                _resolve_runtime_cache,
            )
            from agent_s4.harness.opencode.source import (
                validate_runtime_cache,
            )

            validate_runtime_cache(_resolve_runtime_cache())

        os.makedirs(result_dir, exist_ok=True)
        result_dir = Path(result_dir)
        summary_path = result_dir / "latest_summary.json"

        # Recover leaked V2 instances before accepting any cached completion.
        # A raw evaluator can finish after its parent driver dies, leaving both
        # a valid result cache and a pending cleanup record.
        if benchmark in _RUN_NAME_TAGGED:
            _recover_v2_cleanup_records(result_dir, region)

        cached = _load_complete_cached_summary(
            summary_path,
            task_ids,
            benchmark=benchmark,
        )
        if cached is not None:
            logger.warning(
                "Resume: reusing complete eval summary %s; skipping VM launch",
                summary_path,
            )
            return cached

        recovered_summary = _recover_summary_from_task_results(
            result_dir,
            task_ids,
            model,
            benchmark=benchmark,
        )
        if recovered_summary is not None:
            _atomic_write_json(summary_path, recovered_summary)
            logger.warning(
                "Resume: rebuilt %s from %d complete per-task result(s); "
                "skipping VM launch",
                summary_path,
                len(task_ids),
            )
            return _eval_results_from_summary(
                recovered_summary,
                task_ids,
                benchmark=benchmark,
            )

        # An invalid or partial summary must never survive into a fresh retry.
        summary_path.unlink(missing_ok=True)

        # Task JSONs come from the benchmark's read-only *_Base tree, not the
        # island clone — OSWorld_Base for osworld, OSWorldV2_Base for
        # osworld_v2. Defaults to osworld so older callers are unaffected.
        base_name = BENCHMARKS[benchmark]["base"]
        test_config_base_dir = (
            clone_dir.parent / base_name / "tasks" / "VM"
            if benchmark == "scienceboard"
            else clone_dir.parent / base_name / "evaluation_examples"
        )

        requested_num_envs = os.environ.get("SHADOW_EVOL_EVAL_NUM_ENVS")
        if requested_num_envs is None:
            num_envs = len(task_ids)
        else:
            try:
                num_envs = int(requested_num_envs)
            except ValueError as exc:
                raise ValueError(
                    "SHADOW_EVOL_EVAL_NUM_ENVS must be a positive integer, got "
                    f"{requested_num_envs!r}"
                ) from exc
            if num_envs <= 0:
                raise ValueError(
                    "SHADOW_EVOL_EVAL_NUM_ENVS must be a positive integer, got "
                    f"{requested_num_envs!r}"
                )
            num_envs = min(num_envs, len(task_ids))

        cmd = [
            sys.executable,
            "-m",
            EVAL_MODULE_BY_BENCHMARK[benchmark],
            "--model",
            model,
            "--num_envs",
            str(num_envs),
            "--task_ids",
            *task_ids,
            "--shadow_dir",
            str(shadow_dir),
            "--result_dir",
            str(result_dir),
            "--region",
            region,
            "--timeout",
            str(timeout),
            "--test_config_base_dir",
            str(test_config_base_dir),
            "--benchmark",
            benchmark,
        ]
        if harness is not None:
            cmd.extend(["--harness", harness])
        if max_steps is not None:
            cmd.extend(["--max-steps", str(max_steps)])
        if no_time_limit:
            cmd.append("--no-time-limit")
        if benchmark == "osworld_v2":
            cmd.append("--enable-proxy")
        if benchmark in _RUN_NAME_TAGGED:
            # An orchestrator outside the benchmark VPC cannot route the
            # private address the provider returns by default, so it has to
            # ask for the public one. Strict parse: a typo here costs a whole
            # eval batch, because the failure is every task hanging in
            # vm_allocation rather than an error.
            if _parse_bool_env("SHADOW_EVOL_USE_PUBLIC_IP", "false"):
                cmd.append("--use-public-ip")

        run_name = None
        subprocess_env = None
        cleanup_record_path = Path(result_dir) / "vm_cleanup.json"
        if benchmark in _RUN_NAME_TAGGED:
            run_name = f"shadow-evol-{uuid.uuid4().hex}"
            cleanup_record_path = Path(result_dir) / f"vm_cleanup.{run_name}.json"
            subprocess_env = os.environ.copy()
            # Both V2's initial manager and reset/revert provider attach this
            # tag at run_instances time, including the pre-return leak window.
            subprocess_env["AWS_INSTANCE_NAME"] = run_name
            _atomic_write_json(
                cleanup_record_path,
                {
                    "run_name": run_name,
                    "region": region,
                    "cleanup_status": "pending",
                    "cleanup_action": os.environ.get(
                        "SHADOW_EVOL_V2_CLEANUP_ACTION",
                        "terminate",
                    ),
                },
            )

        def cleanup_v2_instances():
            if run_name is None:
                return
            try:
                terminated = _cleanup_v2_instances(region, run_name)
            except Exception as exc:
                _atomic_write_json(
                    cleanup_record_path,
                    {
                        "run_name": run_name,
                        "region": region,
                        "cleanup_status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise RuntimeError(
                    f"Could not verify cleanup for V2 eval {run_name}: {exc}"
                ) from exc
            id_field = (
                "stopped_instance_ids"
                if os.environ.get("SHADOW_EVOL_V2_CLEANUP_ACTION") == "stop"
                else "terminated_instance_ids"
            )
            cleanup_record = {
                "run_name": run_name,
                "region": region,
                "cleanup_status": "complete",
                "cleanup_action": os.environ.get(
                    "SHADOW_EVOL_V2_CLEANUP_ACTION",
                    "terminate",
                ),
            }
            cleanup_record[id_field] = terminated
            _atomic_write_json(cleanup_record_path, cleanup_record)

        logger.info(
            "Evaluating %d task(s) via agent_s4.eval.osworld (model=%s, harness=%s)",
            len(task_ids),
            model,
            harness or "<legacy>",
        )

        # A no-time-limit agent run must not be re-capped at the enclosing
        # 40-task subprocess. Infrastructure setup/upload calls retain their
        # own bounded timeouts, but completed agent work may take arbitrarily
        # long. Timed runs keep the historical batch watchdog formula.
        #
        # That formula assumes the batch is roughly one wave of environments,
        # so it stops holding once a long per-task cap meets a task list
        # several times SHADOW_EVOL_EVAL_NUM_ENVS: 40 tasks through 20
        # environments at a 2h cap needs over 4h, which timeout*2 does not
        # cover, and the batch would be killed mid-episode and scored as an
        # eval timeout rather than at the per-task deadline. The override
        # states the batch bound explicitly; "none" disables it, matching
        # no_time_limit's outer behaviour while keeping the per-task cap.
        outer_timeout = (
            None if no_time_limit else timeout * 2 + 400 * max(1, len(task_ids) // 25)
        )
        batch_override = os.environ.get("SHADOW_EVOL_EVAL_BATCH_TIMEOUT", "").strip()
        if batch_override and not no_time_limit:
            if batch_override.lower() == "none":
                outer_timeout = None
            else:
                try:
                    outer_timeout = int(batch_override)
                except ValueError as exc:
                    raise ValueError(
                        "SHADOW_EVOL_EVAL_BATCH_TIMEOUT must be a positive "
                        f"integer or 'none', got {batch_override!r}"
                    ) from exc
                if outer_timeout <= 0:
                    raise ValueError(
                        "SHADOW_EVOL_EVAL_BATCH_TIMEOUT must be a positive "
                        f"integer or 'none', got {batch_override!r}"
                    )
            logger.info(
                "Batch watchdog pinned by SHADOW_EVOL_EVAL_BATCH_TIMEOUT: %s",
                outer_timeout if outer_timeout is not None else "disabled",
            )

        try:
            proc = _run_eval_subprocess(
                cmd,
                cwd=str(clone_dir),
                capture_output=True,
                text=True,
                timeout=outer_timeout,
                env=subprocess_env,
            )
            output = proc.stdout
            if len(output) > 2000:
                output = "...\n" + output[-2000:]
            print(output, flush=True)
            # Keep the stderr tail: worker subprocesses die there (a provider
            # import that raises, a bad kwarg), and without it a failed eval
            # surfaces only as missing results with no cause attached.
            stderr_tail = (proc.stderr or "")[-2000:]
            if proc.returncode != 0:
                logger.error(
                    "Eval subprocess exited %d; stderr tail:\n%s",
                    proc.returncode,
                    stderr_tail or "<empty>",
                )
        except subprocess.TimeoutExpired:
            assert outer_timeout is not None
            logger.error(
                "Eval harness timed out after %ds (per-task cap=%ds) — "
                "returning timed_out=True so the caller can salvage partial results",
                outer_timeout,
                timeout,
            )
            cleanup_v2_instances()
            return EvalResults(
                pass_rate=0,
                total=len(task_ids),
                passed=0,
                failed=len(task_ids),
                results=[],
                timed_out=True,
            )
        except BaseException:
            cleanup_v2_instances()
            raise

        cleanup_v2_instances()

        if summary_path.exists():
            with summary_path.open() as f:
                d = json.load(f)
            if not d:
                # An empty summary means the subprocess ran to completion but
                # no worker produced a single result — every VM died before
                # its first task. That is infrastructure, not a bad interface,
                # so refuse rather than record a 0 the population would then
                # be selected against. (A missing summary keeps its existing
                # zeroed-results contract below.)
                raise RuntimeError(
                    f"Eval produced an empty summary at {summary_path}: all "
                    f"{len(task_ids)} task(s) went unrun, so every worker died "
                    f"before starting. Subprocess exited {proc.returncode}; "
                    f"stderr tail:\n{stderr_tail or '<empty>'}"
                )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Eval subprocess exited {proc.returncode} despite writing "
                    f"a summary; refusing ambiguous fitness. stderr tail:\n"
                    f"{stderr_tail or '<empty>'}"
                )
            return _eval_results_from_summary(
                d,
                task_ids,
                benchmark=benchmark,
            )

        raise RuntimeError(
            f"Eval subprocess exited {proc.returncode} without "
            f"latest_summary.json; stderr tail:\n{stderr_tail or '<empty>'}"
        )
