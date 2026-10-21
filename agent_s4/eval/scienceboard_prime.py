"""ScienceBoard evaluation through the shared OSWorld worker.

Run as ``python -m agent_s4.eval.scienceboard_prime ...``. This module
plugs ScienceBoard into ``osworld.main(impl=...)`` the same way ``osworld_v2``
does, so Prime Agent, the per-task wall-clock budget, infra
retries, artifact collection and summaries are the code V2 already runs.

What is ScienceBoard-specific lives here:

  - the task index: official tasks under ScienceBoard_Base/tasks/VM (ids
    ``<App>:<stem>``) plus the ungraded synthetic training bank named by
    ``SCIENCEBOARD_SYNTHETIC_ROOT`` (ids ``sb_<app>_<NNN>``)
  - ``ScienceBoardEnv``: one fresh EC2 per task from the ScienceBoard AMI,
    exposing the ``controller`` / ``reset`` / ``close`` surface the shared
    worker drives, with the per-app ``sci`` VMManager wired onto it
  - scoring: official tasks run ScienceBoard's own ``Task.eval`` after
    bridging the agent's /tmp/AGENT_ANSWER and /tmp/INFEASIBLE sentinels into
    a stop primitive; synthetic tasks have no evaluator and record the same
    ``dummy_constant_zero`` placeholder the V2 synthetic bank uses
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from agent_s4.eval import osworld
from agent_s4.eval.osworld import (
    INSTRUCTION_CONTEXT_SUFFIX,
    add_common_args,
    finalize_task_result,
    persist_evaluation_result,
    prepare_task_vm,
    resolve_model_config,
    _run_agent_episode,
)

logger = logging.getLogger("desktopenv.shadow_eval.scienceboard")

from agent_s4.paths import PROJECT_ROOT

SYNTHETIC_ROOT_ENV = "SCIENCEBOARD_SYNTHETIC_ROOT"

# A ScienceBoard AMI in your own account (see osworld_setup/s4/README.md).
SB_AMI_ID = os.environ.get("SB_AMI_ID", "")
# t3.medium (4 GB) runs out of memory once VS Code starts Lean over Mathlib.
SB_INSTANCE_TYPE = os.environ.get("SB_INSTANCE_TYPE", "t3.large")
# sci.Prompts.VM_PASSWORD -- the AMI's sudo password for user "user".
SB_CLIENT_PASSWORD = os.environ.get("SB_CLIENT_PASSWORD", "osworld-public-evaluation")
# The scientific apps bind their REST ports well after the :5000 daemon is up;
# ScienceBoard's own task waits (2.5-5s) assume a warm desktop. Below this the
# next init step races the app and fails with a JSON decode error.
LAUNCH_WAIT_FLOOR_SECONDS = 60
DAEMON_BOOT_DEADLINE_SECONDS = 420

SCIENCEBOARD_PROMPT_ADDENDUM = """

## ScienceBoard task semantics

- If the task asks for a specific answer (e.g. a number, a name, a value), write your final answer to /tmp/AGENT_ANSWER as plain text on a single line, then exit.
- If you determine the task is impossible to complete, create /tmp/INFEASIBLE and exit.
- Most tasks have no answer expectation; for those, just complete the operations and exit normally.
- The scientific application the task concerns is already open. Keep it running: the benchmark inspects its live state after you exit.
"""

UNGRADED = {
    "verification": "dummy_constant_zero",
    "score_semantics": "UNGRADED: constant zero; use trajectories, not scores",
}


# ---------------------------------------------------------------------------
# Task index
# ---------------------------------------------------------------------------


def _validate_synthetic_package(root: Path) -> list[dict]:
    """Check the input bundle against its own manifest before any VM exists."""
    manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text())
    for relative, expected in manifest.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"ScienceBoard synthetic package is missing {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(
                f"ScienceBoard synthetic package hash mismatch: {relative}"
            )
    entries = json.loads((root / "index.json").read_text())
    if len(entries) != 50 or any(e.get("verification") != "none" for e in entries):
        raise ValueError("Expected the 50-task ungraded ScienceBoard synthetic bank")
    return entries


def build_task_index(test_config_base_dir):
    """Map task id -> definition path for official and synthetic tasks."""
    root = Path(test_config_base_dir)
    if not root.is_dir():
        raise SystemExit(f"ScienceBoard tasks root not found: {root}")
    index = {}
    for app_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(app_dir.glob("*.json")):
            index[f"{app_dir.name}:{path.stem}"] = str(path)
    configured = os.environ.get(SYNTHETIC_ROOT_ENV)
    if configured:
        synthetic_root = Path(configured).resolve()
        for entry in _validate_synthetic_package(synthetic_root):
            if entry["id"] in index:
                raise ValueError(f"Synthetic task id collides: {entry['id']}")
            index[entry["id"]] = str(synthetic_root / entry["path"] / "task.json")
    return index


# The guest recorder is ffmpeg x11grab -> libx264 at 1080p30. On the 2-vCPU
# ScienceBoard instance it saturates the CPU for the whole episode, competing
# with the agent and the scientific application, and after SIGINT it spends
# tens of minutes draining its backlog -- /end_recording has no timeout, so the
# task's VM and worker stay pinned that long. Trajectories are what the Builder
# reads; recordings are not worth either cost.
DISABLE_RECORDING = True


def load_task(task_path, test_config_base_dir):
    del test_config_base_dir
    path = Path(task_path)
    spec = json.loads(path.read_text())
    if "primary_application" in spec:
        return {
            "id": spec["id"],
            "instruction": spec["instruction"],
            "type": spec["primary_application"],
            "steps": spec["steps"],
            "sb_kind": "synthetic",
            "sb_dir": str(path.parent),
            "disable_recording": DISABLE_RECORDING,
        }
    task = dict(spec)
    task["id"] = f"{path.parent.name}:{path.stem}"
    task["sb_kind"] = "official"
    task["sb_path"] = str(path)
    task["disable_recording"] = DISABLE_RECORDING
    return task


def load_all_task_ids(test_all_meta_path):
    return osworld.load_all_task_ids(test_all_meta_path)


# ---------------------------------------------------------------------------
# One EC2 per task
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvSpec:
    # The task id makes every spec distinct, so the shared worker never reuses
    # a VM: ScienceBoard has no in-place snapshot revert on AWS.
    task_id: str
    app: str
    ami_id: str
    instance_type: str


def build_env_spec(example, args) -> EnvSpec:
    del args
    if not SB_AMI_ID:
        raise RuntimeError("SB_AMI_ID is not set: point it at a ScienceBoard AMI")
    return EnvSpec(
        task_id=example["id"],
        app=example["type"],
        ami_id=SB_AMI_ID,
        instance_type=SB_INSTANCE_TYPE,
    )


def build_env_kwargs(spec: EnvSpec, args) -> dict:
    return {
        "spec": spec,
        "region": args.region,
        "use_public_ip": bool(getattr(args, "use_public_ip", False)),
    }


class _SciEnvShim:
    """What sci's VManager expects of OSWorld's DesktopEnv, on our instance.

    sci's AWS revert terminates and relaunches through OSWorld's provider (no
    Owner tag, private addressing). The instance here is already fresh, so
    revert reduces to waiting for the daemon, which sci then does itself.
    """

    def __init__(self, env: "ScienceBoardEnv"):
        self.controller = env.controller
        self.vm_ip = env.vm_ip
        self.snapshot_name = env.spec.ami_id

    def _revert_to_snapshot(self):
        return None

    def _start_emulator(self):
        return None

    def close(self):
        return None


def _ensure_sci_importable() -> None:
    root = str(
        Path(
            os.environ.get(
                "SCIENCEBOARD_PATH",
                PROJECT_ROOT / "ScienceBoard_Base",
            )
        ).resolve()
    )
    if root not in sys.path:
        sys.path.insert(0, root)


def _floor_launch_waits(initialize: list[dict]) -> list[dict]:
    out = []
    for item in initialize:
        item = dict(item)
        if item.get("func") == "launch":
            item["wait"] = max(item.get("wait", 0), LAUNCH_WAIT_FLOOR_SECONDS)
        out.append(item)
    return out


class ScienceBoardEnv:
    def __init__(
        self,
        *,
        spec: EnvSpec,
        region: str,
        use_public_ip: bool,
        volume_size: int | None = None,
    ):
        import boto3
        import requests
        from desktop_env.controllers.python import PythonController

        self.spec = spec
        self.region = region
        self.client_password = SB_CLIENT_PASSWORD
        self.action_history = []
        self.is_environment_used = False
        self.manager = None
        self.task = None
        self._tmp_dir = None
        self._closed = False

        subnet = os.environ.get("AWS_SUBNET_ID")
        group = os.environ.get("AWS_SECURITY_GROUP_ID")
        if not subnet or not group:
            raise RuntimeError("AWS_SUBNET_ID and AWS_SECURITY_GROUP_ID are required")
        tags = [
            {
                "Key": "Name",
                "Value": os.environ.get(
                    "AWS_INSTANCE_NAME", f"shadow-evol-sb-{spec.task_id}"
                ),
            },
            {"Key": "sb_task", "Value": spec.task_id},
        ]
        self._ec2 = boto3.client("ec2", region_name=region)
        volume_options = {}
        self.root_volume_gib = None
        if volume_size is not None:
            if volume_size < 1:
                raise ValueError("volume_size must be positive")
            image = self._ec2.describe_images(ImageIds=[spec.ami_id])["Images"][0]
            root_name = image["RootDeviceName"]
            root_ebs = next(
                mapping["Ebs"]
                for mapping in image["BlockDeviceMappings"]
                if mapping["DeviceName"] == root_name
            )
            # An EBS snapshot cannot be restored onto a smaller volume.
            self.root_volume_gib = max(volume_size, root_ebs["VolumeSize"])
            volume_options["BlockDeviceMappings"] = [
                {
                    "DeviceName": root_name,
                    "Ebs": {
                        "VolumeSize": self.root_volume_gib,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    },
                }
            ]
        response = self._ec2.run_instances(
            ImageId=spec.ami_id,
            InstanceType=spec.instance_type,
            MinCount=1,
            MaxCount=1,
            NetworkInterfaces=[
                {
                    "SubnetId": subnet,
                    "Groups": [group],
                    "AssociatePublicIpAddress": True,
                    "DeviceIndex": 0,
                }
            ],
            TagSpecifications=[{"ResourceType": "instance", "Tags": tags}],
            **volume_options,
        )
        self.instance_id = response["Instances"][0]["InstanceId"]
        try:
            self._ec2.get_waiter("instance_running").wait(
                InstanceIds=[self.instance_id],
                WaiterConfig={"Delay": 5, "MaxAttempts": 60},
            )
            instance = self._ec2.describe_instances(
                InstanceIds=[self.instance_id],
            )[
                "Reservations"
            ][0]["Instances"][0]
            self.vm_ip = (
                instance.get("PublicIpAddress")
                if use_public_ip
                else instance["PrivateIpAddress"]
            )
            if not self.vm_ip:
                raise RuntimeError(f"{self.instance_id} has no reachable address")
            deadline = time.time() + DAEMON_BOOT_DEADLINE_SECONDS
            while True:
                try:
                    if (
                        requests.get(
                            f"http://{self.vm_ip}:5000/platform", timeout=5
                        ).status_code
                        == 200
                    ):
                        break
                except requests.RequestException:
                    pass
                if time.time() > deadline:
                    raise RuntimeError(
                        f"daemon on {self.vm_ip}:5000 never came up for {self.instance_id}"
                    )
                time.sleep(5)
        except BaseException:
            self._terminate()
            raise
        logger.info(
            "ScienceBoard VM %s up at %s for %s",
            self.instance_id,
            self.vm_ip,
            spec.task_id,
        )
        self.controller = PythonController(vm_ip=self.vm_ip, server_port=5000)

    # -- task setup ---------------------------------------------------------

    def _enter_manager(self, app: str):
        _ensure_sci_importable()
        from sci import Presets
        import sci.vm.vmanager as vmanager

        modules = Presets.spawn_modules()
        if app not in modules:
            raise RuntimeError(
                f"unknown ScienceBoard app {app!r}; known: {sorted(modules)}"
            )
        manager = modules[app].VMManager(
            version="0.1",
            vm_path=self.instance_id,
            headless=True,
            port=8000,
            provider_name="aws",
            region=self.region,
            ami_id=self.spec.ami_id,
        )
        shim = _SciEnvShim(self)
        vmanager.ENVS[manager.key] = lambda: shim
        manager.__enter__()
        self.manager = manager
        return modules[app]

    def reset(self, task_config=None):
        if task_config is None:
            # Fresh instance per task; nothing to revert before staging.
            return None
        if self.manager is not None:
            raise RuntimeError("ScienceBoardEnv supports one task per instance")
        module = self._enter_manager(task_config["type"])
        if task_config["sb_kind"] == "synthetic":
            _ensure_sci_importable()
            from agent_s4.scienceboard_synthetic_runtime import TrainingTask

            task = TrainingTask(task_config["sb_dir"])
            task.spec["initialize"] = _floor_launch_waits(task.spec["initialize"])
            # Reset exactly as official VTask._init does. reset.sh is a VMware
            # nmcli network bounce that exits 1 on AWS (no ethernet profile);
            # VTask only checks the HTTP status, while TrainingTask's own
            # reset would treat that exit code as a setup failure.
            if not self.manager.revert("sci_bench"):
                raise RuntimeError("ScienceBoard revert (daemon wait) failed")
            response = self.manager._request(
                "POST/setup/execute",
                {
                    "json": {
                        "command": "/bin/bash /home/user/server/reset.sh",
                        "shell": True,
                    }
                },
            )
            if response.status_code != 200:
                raise RuntimeError(f"reset.sh request failed: {response.status_code}")
            # The adapter installs each material as one base64 argv element,
            # which exceeds Linux's 128 KiB per-argument limit for larger
            # inputs (sb_chimerax_006's 160 KB .dx -> /setup/execute 500).
            # Ship them with the chunked uploader instead, then let prepare()
            # run only the task's own initialization.
            from agent_s4.vm.utils import (
                _execute_shell_on_vm,
                upload_file_to_vm,
            )

            for name in task.spec["materials"]:
                relative = PurePosixPath(name).relative_to("materials")
                dest = f"{task.guest_root}/{relative}"
                made = _execute_shell_on_vm(
                    self.controller,
                    f"mkdir -p {shlex.quote(str(PurePosixPath(dest).parent))}",
                    timeout=30,
                )
                if not made or made.get("returncode") != 0:
                    raise RuntimeError(f"could not create the directory for {dest}")
                if not upload_file_to_vm(self.controller, task.directory / name, dest):
                    raise RuntimeError(f"material upload failed: {name}")
            task.spec["materials"] = []
            task.prepare(self.manager, reset=False)
            self.task = task
            return None

        from sci import OBS
        from sci.base.community import Community

        class _StubCommunity(Community):
            """Satisfies Task's isinstance check; agent_s4 drives its own agent."""

            def __call__(self, *args, **kwargs):
                raise RuntimeError("the stub community never drives an agent")

        self._tmp_dir = tempfile.mkdtemp(prefix="sb-task-")
        config = {
            k: v
            for k, v in task_config.items()
            if k not in {"id", "sb_kind", "sb_path", "disable_recording"}
        }
        config["initialize"] = _floor_launch_waits(config.get("initialize", []))
        path = os.path.join(self._tmp_dir, "task.json")
        with open(path, "w") as stream:
            json.dump(config, stream, indent=2)
        task = module.VMTask(
            config_path=path,
            manager=self.manager,
            community=_StubCommunity(),
            obs_types={OBS.screenshot},
        )
        if not task.init():
            raise RuntimeError("ScienceBoard task.init() failed")
        self.task = task
        return None

    # -- scoring ------------------------------------------------------------

    def score_official(self, example_result_dir: str) -> dict:
        from sci.base.agent import Primitive
        from sci.base.log import Log

        infeasible = self.controller.get_file("/tmp/INFEASIBLE")
        answer_blob = self.controller.get_file("/tmp/AGENT_ANSWER")
        answer = (answer_blob or b"").decode("utf-8", errors="replace").strip()
        if infeasible is not None:
            stop_type, stop_args = Primitive.FAIL, []
        elif answer:
            stop_type, stop_args = Primitive.ANS, [answer]
        else:
            stop_type, stop_args = Primitive.TIMEOUT, []

        # Task.eval consumes matched stop items; snapshot what it checks first.
        eval_items = [dict(item) for item in self.task.evaluate]
        # Task.eval's result handler asserts an open log file. Give the proxy a
        # persistent backing Log, then trigger it into this task's directory
        # (sci writes result.out, distinct from our result.txt).
        if self.task.vlog._log is None:
            self.task.vlog.set(Log(disabled=True, global_vlog=False))
        self.task.vlog.trigger(example_result_dir)

        eval_exception = None
        try:
            passed = bool(self.task.eval(stop_type, stop_args))
        except Exception:
            eval_exception = traceback.format_exc()
            passed = False
        post_eval_state_keys = post_eval_state_error = None
        if not passed:
            try:
                states = self.manager.states_dump()
                if isinstance(states, dict):
                    post_eval_state_keys = sorted(states)[:80]
            except Exception as exc:
                post_eval_state_error = f"{type(exc).__name__}: {exc}"
        details = {
            "score": 1.0 if passed else 0.0,
            "passed": passed,
            "stop_used": stop_type.__name__,
            "stop_args": list(stop_args),
            "eval_items_in_config": eval_items,
            "eval_exception": eval_exception,
            "post_eval_state_error": post_eval_state_error,
            "post_eval_state_keys": post_eval_state_keys,
            "task_app": self.spec.app,
        }
        with open(
            os.path.join(example_result_dir, "scienceboard_evaluation.json"), "w"
        ) as f:
            json.dump(details, f, indent=2, default=str)
        return details

    # -- teardown -----------------------------------------------------------

    def _terminate(self):
        try:
            self._ec2.terminate_instances(InstanceIds=[self.instance_id])
        except Exception as exc:
            logger.error("Could not terminate %s: %s", self.instance_id, exc)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.manager is not None:
            try:
                self.manager.__exit__(None, None, None)
            except Exception as exc:
                logger.warning("sci manager exit failed: %s", exc)
        if self._tmp_dir:
            import shutil

            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._terminate()


DesktopEnv = ScienceBoardEnv


# ---------------------------------------------------------------------------
# Task lifecycle
# ---------------------------------------------------------------------------


def run_single_task(
    env, example, instruction, example_result_dir, args, model_cfg, shadow_dir
):
    start_time = time.time()
    example_id = example["id"]
    base_instruction = instruction + INSTRUCTION_CONTEXT_SUFFIX

    staged = prepare_task_vm(env, example, args, model_cfg, shadow_dir)
    if isinstance(staged, dict):
        return staged
    system_prompt, recording_enabled = staged

    # The clock starts after VM allocation, task setup and interface staging,
    # exactly as in V2: those are infrastructure time, not agent work.
    budget = osworld.TaskBudget(
        None if getattr(args, "no_time_limit", False) else args.timeout
    ).start()
    artifact_chunks = []
    _run_agent_episode(
        env,
        args,
        model_cfg,
        base_instruction,
        system_prompt,
        example_result_dir,
        artifact_chunks=artifact_chunks,
        budget=budget,
    )
    env.is_environment_used = True
    time.sleep(20)
    if example["sb_kind"] == "synthetic":
        raw_result = {"score": 0.0, **UNGRADED, "task_app": example["type"]}
    else:
        raw_result = env.score_official(example_result_dir)
    score, result_details = persist_evaluation_result(raw_result, example_result_dir)

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


# ---------------------------------------------------------------------------
# CLI and process fan-out
# ---------------------------------------------------------------------------


def config():
    parser = argparse.ArgumentParser(
        description="Run shadow interface evaluation against ScienceBoard tasks"
    )
    add_common_args(parser)
    args = parser.parse_args()
    if args.benchmark not in {"auto", "scienceboard"}:
        parser.error("this module implements ScienceBoard semantics only")
    args.benchmark = "scienceboard"
    if not args.prompt_addendum:
        args.prompt_addendum = SCIENCEBOARD_PROMPT_ADDENDUM
    if not args.client_password:
        args.client_password = SB_CLIENT_PASSWORD
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
