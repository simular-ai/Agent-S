"""Re-run a live V2 eval batch's infrastructure-failed tasks, in place.

Variant scores (seed and children alike) are read from each task's
result.txt only after the eval batch exits. A task re-run into the same
result directory while that batch is still running therefore replaces its
infrastructure zero in the recorded fitness, with no driver restart.

The re-run clones the live batch's argv and environment, so model, timeouts,
step cap, harness and shadow_interface are identical. It gets its own
AWS_INSTANCE_NAME, because the batch's cleanup terminates every instance
carrying the batch's tag, and the script cleans its own tag when done.

The re-run writes to a staging directory and the old task directories stay
in place until it finishes. Only then, and only while the batch is still
running, are they swapped (old ones moved under _aborted/, never deleted).
A batch that finishes first therefore always finds a complete cache and
records the old result: removing a task up front made the orchestrator
refuse the incomplete cache and exit, which mid-step can re-dispatch the
whole step.

  python scripts/ops/rerun_infra_failures.py --result-dir eval_cache_.../<variant> [--dry-run]

It refuses when the batch is no longer running: its scores are then already
recorded, and a re-run would change the files but not the DB.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # Agent-S root

# Failures of the environment around the agent, never of the agent itself.
INFRA_TYPES = {
    "vm_allocation_failure", "agent_install_failure", "task_setup_failure",
    "task_load_failure", "infra_failure", "evaluator_failure",
    "user_simulator_failure", "agent_auth_failure",
}
# agent_execution_failure is infrastructure only when the endpoint broke.
TRANSPORT_ERROR = re.compile(
    r"connection|timed? ?out|timeout|ECONNRESET|reset by peer|\b5\d\d\b|\b429\b|"
    r"overloaded|service unavailable|bad gateway", re.I)
MODEL_ERROR = re.compile(r"No valid tool call", re.I)


def batch_process(result_dir: Path) -> int | None:
    """PID of the running osworld_v2 batch writing result_dir (not a re-run)."""
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().decode().split("\0")[:-1]
            env = (proc / "environ").read_bytes().decode()
        except OSError:
            continue
        if "agent_s4.eval.osworld_v2" not in argv or "--result_dir" not in argv:
            continue
        if Path(argv[argv.index("--result_dir") + 1]).resolve() != result_dir:
            continue
        if "AWS_INSTANCE_NAME=shadow-evol-rerun" in env:
            continue
        # Pool workers share argv; the batch leader is the one whose parent is not.
        try:
            parent = (proc / "stat").read_text().rsplit(")", 1)[1].split()[1]
            parent_argv = (Path("/proc") / parent / "cmdline").read_bytes().decode()
        except OSError:
            parent_argv = ""
        if "agent_s4.eval.osworld_v2" in parent_argv:
            continue
        return int(proc.name)
    return None


def classify(task_dir: Path) -> tuple[bool, str]:
    failure = task_dir / "failure.json"
    if not failure.exists():
        return False, "no failure"
    record = json.loads(failure.read_text())
    kind = record.get("failure_type")
    if kind in INFRA_TYPES:
        return True, f"{kind}: {record.get('error', '')[:100]}"
    if kind == "agent_execution_failure":
        trajectory = task_dir / "claude_trajectory.jsonl"
        tail = trajectory.read_text(errors="replace")[-20000:] if trajectory.exists() else ""
        errors = [line for line in tail.splitlines() if '"type":"error"' in line.replace(" ", "")]
        message = errors[-1] if errors else record.get("error", "")
        if MODEL_ERROR.search(message):
            return False, f"{kind}: model tool-call error (not infrastructure)"
        if TRANSPORT_ERROR.search(message):
            return True, f"{kind}: transport error: {message[:120]}"
        return False, f"{kind}: unclassified, left alone: {message[:120]}"
    return False, f"{kind}: agent-side, left alone"


MAX_RERUNS = 2  # per task and variant; a persistent fault must not loop


def rerun_history(result_dir: Path) -> tuple[set[str], dict[str, int]]:
    """Tasks with a re-run in flight, and how often each was re-run."""
    running, attempts = set(), {}
    for record in (ROOT / "runs" / "infra_reruns").glob("*.json"):
        data = json.loads(record.read_text())
        if data.get("result_dir") != str(result_dir):
            continue
        for task in data["tasks"]:
            attempts[task] = attempts.get(task, 0) + 1
        if data.get("status") == "running":
            running.update(data["tasks"])
    return running, attempts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    pid = batch_process(result_dir)
    argv = (Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")[:-1]) if pid else None
    if argv is None:
        print(f"REFUSED: no running batch writes {result_dir}; its scores are already recorded.")
        return 2
    model = argv[argv.index("--model") + 1]
    cache = result_dir / "shadow_eval" / "screenshot" / model
    busy, attempts = rerun_history(result_dir)
    selected = []
    for task_dir in sorted(p for p in cache.iterdir() if p.is_dir()):
        if task_dir.name in busy or not (task_dir / "result.txt").exists():
            continue
        rerun, reason = classify(task_dir)
        if rerun and attempts.get(task_dir.name, 0) >= MAX_RERUNS:
            rerun, reason = False, f"{reason} [already re-run {attempts[task_dir.name]}x; left alone]"
        if reason != "no failure":
            print(f"{'RERUN' if rerun else 'skip '} {task_dir.name}: {reason}")
        if rerun:
            selected.append(task_dir.name)
    if not selected or args.dry_run:
        print(f"{len(selected)} task(s) selected{' (dry run)' if args.dry_run else ''}.")
        return 0

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = f"{result_dir.name}_{stamp}"
    backup = ROOT / "_aborted" / f"infra_rerun_{label}"
    staging = result_dir.parent / f"_infra_rerun_{label}"
    env = dict(kv.split("=", 1) for kv in Path(f"/proc/{pid}/environ").read_bytes().decode().split("\0") if "=" in kv)
    run_name = f"shadow-evol-rerun-{uuid.uuid4().hex[:16]}"
    env["AWS_INSTANCE_NAME"] = run_name
    start = argv.index("--task_ids")
    stop = next(i for i in range(start + 1, len(argv)) if argv[i].startswith("--"))
    command = argv[:start] + ["--task_ids", *selected] + argv[stop:]
    command[command.index("--num_envs") + 1] = str(len(selected))
    command[command.index("--result_dir") + 1] = str(staging)
    records = ROOT / "runs" / "infra_reruns"
    records.mkdir(parents=True, exist_ok=True)
    record_path, log_path = records / f"{label}.json", records / f"{label}.log"
    record = {"result_dir": str(result_dir), "tasks": selected, "run_name": run_name,
              "staging": str(staging), "backup": str(backup), "batch_pid": pid,
              "started": stamp, "status": "running"}
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, cwd=os.readlink(f"/proc/{pid}/cwd"), env=env,
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    print(f"re-running {selected} as pid {proc.pid}, tag {run_name}\n  log {log_path}\n  record {record_path}")
    code = proc.wait()
    # Clean up with the batch's credentials: this script's own environment
    # (an operator shell) need not carry AWS_PROFILE.
    for key in ("AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_CONFIG_FILE",
                "AWS_SHARED_CREDENTIALS_FILE"):
        if key in env:
            os.environ[key] = env[key]
    from agent_s4.harness.ubuntu_vm import _cleanup_v2_instances
    try:
        terminated = _cleanup_v2_instances(env.get("AWS_REGION", "us-east-1"), run_name)
    except Exception as exc:  # the record must still leave "running"
        terminated = f"cleanup failed: {type(exc).__name__}: {exc}"
    staged = staging / "shadow_eval" / "screenshot" / model
    scores = {t: (staged / t / "result.txt").read_text().strip()
              if (staged / t / "result.txt").exists() else None for t in selected}
    swapped, kept = [], []
    if batch_process(result_dir) == pid:
        backup.mkdir(parents=True, exist_ok=True)
        for task in selected:
            if scores[task] is None:
                kept.append(task)  # re-run produced nothing: keep the original
                continue
            shutil.move(str(cache / task), str(backup / task))
            shutil.move(str(staged / task), str(cache / task))
            swapped.append(task)
    else:
        kept = list(selected)
        print("WARNING: the batch finished before this re-run; it recorded the old results. "
              f"Re-run results stay in {staging}.")
    record.update(status="done", exit_code=code, terminated=terminated, scores=scores,
                  swapped=swapped, kept_original=kept)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
