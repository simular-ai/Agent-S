"""Scratch VM lifecycle for the builder agent.

Provisions an AWS OSWorld VM before each mutation-builder invocation, polls
for readiness, deploys the parent's shadow_interface, and tears the VM
down after the builder exits. On any provisioning failure the helper
returns None so the builder can still run (soft-degrade)."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import shutil
import socket
import sys
import threading
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("scratch_vm")

# Test override only. Production resolves DesktopEnv lazily from the selected
# benchmark checkout so an editable-installed V1 package cannot contaminate a
# V2 scratch VM. Existing tests patch this symbol with a fake constructor.
DesktopEnv = None
_DESKTOP_IMPORT_LOCK = threading.Lock()
_V2_PROVISION_LOCK = threading.Lock()
_ACTIVE_V2_SCRATCH_RUNS: set[str] = set()
_ACTIVE_V2_SCRATCH_LOCK = threading.Lock()
_V2_SCRATCH_RECORD_RE = re.compile(
    r"^vm_cleanup\.(shadow-evol-scratch-[0-9a-f]+)\.json$"
)


@dataclass(frozen=True)
class V2ScratchVmSpec:
    release: str
    code_tag: str
    code_commit: str
    ami_id: str
    screen_size: tuple[int, int]
    instance_type: str
    volume_size: int | None
    client_password: str
    enable_proxy: bool


def _module_within(module_file: str, root: Path) -> bool:
    try:
        Path(module_file).resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _resolve_desktop_env(clone_dir: str | None, benchmark: str):
    """Resolve DesktopEnv from the benchmark checkout, never by cwd accident."""
    if DesktopEnv is not None:
        return DesktopEnv
    if benchmark not in {"osworld", "osworld_v2", "scienceboard"}:
        raise ValueError(f"unsupported scratch benchmark: {benchmark!r}")
    if benchmark == "scienceboard":
        return importlib.import_module("desktop_env.desktop_env").DesktopEnv
    if clone_dir is None:
        raise ValueError(f"clone_dir is required for {benchmark} scratch VM")
    checkout = Path(clone_dir).resolve()
    if benchmark == "osworld_v2":
        # ProcessPool workers are warm across mutations and islands. Import
        # from the immutable template every time so the first island's
        # absolute module path cannot poison later islands in sys.modules.
        checkout = checkout.parent / "OSWorldV2_Evol"
    expected_package = checkout / "desktop_env"
    if not expected_package.is_dir():
        raise RuntimeError(
            f"{benchmark} checkout lacks desktop_env package: {expected_package}"
        )

    with _DESKTOP_IMPORT_LOCK:
        loaded = sys.modules.get("desktop_env.desktop_env")
        if loaded is not None:
            if not _module_within(getattr(loaded, "__file__", ""), checkout):
                raise RuntimeError(
                    f"desktop_env is already loaded from {getattr(loaded, '__file__', None)}; "
                    f"refusing to provision {benchmark} scratch from {checkout}"
                )
            return loaded.DesktopEnv
        sys.path.insert(0, str(checkout))
        try:
            module = importlib.import_module("desktop_env.desktop_env")
        finally:
            try:
                sys.path.remove(str(checkout))
            except ValueError:
                pass
        if not _module_within(getattr(module, "__file__", ""), checkout):
            raise RuntimeError(
                f"resolved desktop_env from {getattr(module, '__file__', None)}, "
                f"expected checkout {checkout}"
            )
        return module.DesktopEnv


def _load_v2_scratch_spec(clone_dir: str, region: str) -> V2ScratchVmSpec:
    root = Path(clone_dir).resolve().parent
    pin_path = root / "config" / "osworld_v2_release_pin.json"
    try:
        pin = json.loads(pin_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OSWorld V2 scratch release pin is invalid: {exc}") from exc
    release_path = (
        root / "OSWorldV2_Base" / "benchmark_releases" / f"{pin.get('release')}.json"
    )
    try:
        release = json.loads(release_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OSWorld V2 release manifest is invalid: {exc}") from exc
    screen = tuple(pin.get("scratch_vm", {}).get("screen_size", (1920, 1080)))
    if len(screen) != 2:
        raise RuntimeError(f"invalid V2 scratch screen_size in {pin_path}: {screen!r}")
    screen_key = f"{screen[0]}x{screen[1]}"
    pinned_ami = pin.get("aws_ami", {}).get(region, {}).get(screen_key)
    release_ami = (
        release.get("provider_images", {})
        .get("aws", {})
        .get("ubuntu", {})
        .get(region, {})
        .get(screen_key, {})
        .get("ami_id")
    )
    if not pinned_ami or pinned_ami != release_ami:
        raise RuntimeError(
            f"V2 scratch AMI is unpinned or disagrees with release: "
            f"pin={pinned_ami!r}, release={release_ami!r}, region={region}"
        )
    release_code = release.get("osworld_code", {})
    if release_code.get("tag") != pin.get("code_tag") or release_code.get(
        "repository"
    ) != pin.get("code_repository"):
        raise RuntimeError("V2 scratch release code identity disagrees with parent pin")
    scratch = pin.get("scratch_vm", {})
    return V2ScratchVmSpec(
        release=pin["release"],
        code_tag=pin["code_tag"],
        code_commit=pin["code_commit"],
        ami_id=pinned_ami,
        screen_size=(int(screen[0]), int(screen[1])),
        instance_type=str(scratch["instance_type"]),
        volume_size=scratch.get("volume_size"),
        client_password=str(scratch["client_password"]),
        enable_proxy=bool(scratch.get("enable_proxy", False)),
    )


@dataclass
class ScratchVmHandle:
    env: (
        object  # DesktopEnv instance; typed `object` to avoid the import at module load
    )
    vm_ip: str
    server_port: int
    instance_id: str
    client_password: str
    clone_dir: str
    # OSWorld exposes browser/media control ports for its SetupController.
    # ScienceBoard has none (the SB builder launches the scientific apps itself
    # via execute_python_command), so these stay None on the SB path and
    # scratch_vm_env omits them.
    chromium_port: Optional[int] = None
    vnc_port: Optional[int] = None
    vlc_port: Optional[int] = None
    # Set only on the ScienceBoard VManager-fallback attach path. When present,
    # teardown closes via the manager's __exit__ (which also clears the
    # per-process sci ENVS entry); otherwise teardown calls env.close().
    vmanager: object = None
    benchmark: str = "osworld"
    release: str | None = None
    ami_id: str | None = None
    run_name: str | None = None
    region: str | None = None
    cleanup_record_path: str | None = None


def _v2_scratch_cleanup_path(clone_dir: str, run_name: str) -> Path:
    root = Path(clone_dir).resolve().parent
    return root / "scratch_cleanup_osworld_v2" / f"vm_cleanup.{run_name}.json"


def _process_identity(pid: int | None = None) -> dict:
    pid = os.getpid() if pid is None else pid
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        stat = Path(f"/proc/{pid}/stat").read_text()
        # Everything after the final ')' starts at field 3 (state); starttime
        # is proc-stat field 22, hence zero-based offset 19 in this suffix.
        start_ticks = stat.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        boot_id = "unknown"
        start_ticks = "unknown"
    return {
        "owner_pid": pid,
        "owner_host": socket.gethostname(),
        "owner_boot_id": boot_id,
        "owner_start_ticks": start_ticks,
    }


def _record_owner_is_live(record: dict) -> bool:
    if record.get("owner_host") not in (None, socket.gethostname()):
        # A shared workspace may be mounted on another live host. Without a
        # cross-host lease service, fail safe and leave that record alone.
        return True
    identity = _process_identity(record.get("owner_pid"))
    expected = {
        key: record.get(key)
        for key in (
            "owner_pid",
            "owner_host",
            "owner_boot_id",
            "owner_start_ticks",
        )
    }
    if expected != identity:
        return False
    if record.get("owner_pid") == os.getpid():
        with _ACTIVE_V2_SCRATCH_LOCK:
            return record.get("run_name") in _ACTIVE_V2_SCRATCH_RUNS
    return True


def _write_v2_scratch_record(
    path: Path, *, run_name: str, region: str, status: str, **extra
) -> None:
    from agent_s4.harness.ubuntu_vm import _atomic_write_json

    _atomic_write_json(
        path,
        {
            "run_name": run_name,
            "region": region,
            "cleanup_status": status,
            **_process_identity(),
            **extra,
        },
    )


def _sweep_v2_scratch_run(path: Path, *, run_name: str, region: str) -> None:
    from agent_s4.harness.ubuntu_vm import _cleanup_v2_instances

    try:
        terminated = _cleanup_v2_instances(region, run_name)
    except BaseException as exc:
        _write_v2_scratch_record(
            path,
            run_name=run_name,
            region=region,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    else:
        _write_v2_scratch_record(
            path,
            run_name=run_name,
            region=region,
            status="complete",
            terminated_instance_ids=terminated,
        )
    finally:
        with _ACTIVE_V2_SCRATCH_LOCK:
            _ACTIVE_V2_SCRATCH_RUNS.discard(run_name)


def _recover_v2_scratch_records(cleanup_dir: Path, region: str) -> None:
    """Sweep incomplete runs whose exact local owner process is no longer live."""
    import fcntl

    cleanup_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cleanup_dir / ".recovery.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for path in sorted(cleanup_dir.glob("vm_cleanup.*.json")):
            try:
                record = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                match = _V2_SCRATCH_RECORD_RE.fullmatch(path.name)
                if match is None:
                    continue
                record = {"run_name": match.group(1), "cleanup_status": "malformed"}
            run_name = record.get("run_name")
            if (
                not run_name
                or record.get("cleanup_status") == "complete"
                or _record_owner_is_live(record)
            ):
                continue
            _sweep_v2_scratch_run(path, run_name=run_name, region=region)


def scratch_vm_env(handle: ScratchVmHandle) -> dict:
    """Build the env-var dict injected into the builder subprocess.

    All values are stringified — subprocess env vars are strings. The OSWorld
    browser/media ports are emitted only when set; on the ScienceBoard path
    they are None and omitted, so the SB builder-side module reads just the
    five common SCRATCH_VM_* vars (plus SCRATCH_VM_PYTHON, which only tells
    the Builder how to run it)."""
    env = {
        "SCRATCH_VM_IP": handle.vm_ip,
        "SCRATCH_VM_SERVER_PORT": str(handle.server_port),
        "SCRATCH_VM_INSTANCE_ID": handle.instance_id,
        "SCRATCH_VM_CLIENT_PASSWORD": handle.client_password,
        "SCRATCH_VM_CLONE_DIR": handle.clone_dir,
        # An interpreter that can import the bridge's desktop_env dependencies
        # and that Landlock lets the Builder execute. A Builder's own Python
        # (e.g. Prime's IPython kernel) or the `python` on PATH may be neither.
        "SCRATCH_VM_PYTHON": sys.executable,
    }
    if handle.chromium_port is not None:
        env["SCRATCH_VM_CHROMIUM_PORT"] = str(handle.chromium_port)
    if handle.vnc_port is not None:
        env["SCRATCH_VM_VNC_PORT"] = str(handle.vnc_port)
    if handle.vlc_port is not None:
        env["SCRATCH_VM_VLC_PORT"] = str(handle.vlc_port)
    return env


def stage_builder_scratch_bridge(clone_dir: str) -> Path:
    """Refresh the standalone bridge inside the Builder's allowed workspace.

    The bridge's redeploy helper needs agent_s4/vm/utils.py, which Landlock
    hides with the rest of the parent project, so it is staged alongside as
    ``scratch_vm_utils.py`` (it depends only on the stdlib and requests)."""
    here = Path(__file__).resolve().parent
    workspace = Path(clone_dir).resolve()
    for source, name in (
        (here / "vm" / "utils.py", "scratch_vm_utils.py"),
        (here / "builder_scratch_vm.py", "scratch_vm.py"),
    ):
        destination = workspace / name
        temporary = destination.with_name(f".{name}.staging")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    return workspace / "scratch_vm.py"


def _wait_for_vm_server(
    vm_ip: str, server_port: int, *, timeout_s: int = 300, poll_interval_s: float = 3.0
) -> bool:
    """Poll http://{vm_ip}:{server_port}/screenshot until HTTP 200 or timeout.

    DesktopEnv.__init__ returns as soon as EC2 says instance_running, but the
    OSWorld agent-side server inside the VM boots 30-90s later. Returns True
    on success, False on timeout."""
    url = f"http://{vm_ip}:{server_port}/screenshot"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, socket.timeout, ConnectionRefusedError, OSError):
            pass
        time.sleep(poll_interval_s)
    return False


def teardown_scratch_vm(handle: Optional[ScratchVmHandle]) -> None:
    """Terminate the VM. Safe to call with None. Swallows exceptions so a
    teardown failure never propagates out of the builder's finally block.

    On the ScienceBoard VManager-fallback path the manager owns the env, so we
    close through manager.__exit__ (env.close() + ENVS cleanup); otherwise we
    close the DesktopEnv directly."""
    if handle is None:
        return
    vmanager = getattr(handle, "vmanager", None)
    close_error = None
    try:
        if vmanager is not None:
            vmanager.__exit__(None, None, None)
        else:
            handle.env.close()
    except Exception as e:
        close_error = e
        logger.warning("teardown_scratch_vm: close failed: %s", e)
    if (
        handle.benchmark == "osworld_v2"
        and handle.run_name
        and handle.region
        and handle.cleanup_record_path
    ):
        # A provider close error is not accepted as proof of termination.
        _sweep_v2_scratch_run(
            Path(handle.cleanup_record_path),
            run_name=handle.run_name,
            region=handle.region,
        )
    elif close_error is not None:
        # Preserve the historical soft teardown behavior outside formal V2.
        return


def _tag_instance(instance_id: str, *, region: str) -> None:
    """Tag the EC2 instance with scratch-vm=true for orphan discovery."""
    import boto3

    ec2 = boto3.client("ec2", region_name=region)
    ec2.create_tags(
        Resources=[instance_id],
        Tags=[
            {"Key": "scratch-vm", "Value": "true"},
            {"Key": "orchestrator-pid", "Value": str(os.getpid())},
        ],
    )


def _extract_aws_instance_id(env) -> str:
    """Return the EC2 instance ID for an AWS-provider DesktopEnv.

    OSWorld's AWS provider stores the instance ID in env.path_to_vm
    (the attribute name is generic across providers — for VMware/virtualbox
    it's a filesystem path). Asserting provider_name here makes the
    assumption explicit; if this module is ever used against a different
    provider, the failure is a loud AssertionError rather than a silent
    ec2.create_tags failure on an i-prefixed-path that never existed.
    """
    provider = getattr(env, "provider_name", None)
    if provider != "aws":
        raise AssertionError(
            f"scratch_vm expects AWS-provider DesktopEnv, got provider_name={provider!r}"
        )
    return env.path_to_vm


def _upload_parent_shadow_interface(env, clone_dir: str) -> None:
    """Deploy the parent's shadow_interface/ (checked out at clone_dir) to the VM."""
    from agent_s4.vm.utils import upload_shadow_interface

    shadow_dir = str(Path(clone_dir) / "shadow_interface")
    ok = upload_shadow_interface(env.controller, shadow_dir)
    if not ok:
        raise RuntimeError(f"upload_shadow_interface returned False for {shadow_dir}")


def provision_scratch_vm(
    clone_dir: str,
    *,
    region: str = "us-east-1",
    readiness_timeout_s: int = 300,
    with_shadow_setup: bool = True,
    benchmark: str = "osworld",
) -> Optional[ScratchVmHandle]:
    """Provision a scratch VM, wait for its HTTP server, deploy the parent's
    shadow_interface. Returns None on ANY failure (soft-degrade).

    benchmark="scienceboard" boots the ScienceBoard AMI (the 6 scientific apps
    pre-installed) on the eval's network and attaches the same OSWorld
    controller; benchmark="osworld" (default) provisions a bare OSWorld VM.

    with_shadow_setup=False skips the shadow_interface upload + setup.sh run
    — useful for the asset-cache bootstrap, which needs a bare Ubuntu VM
    plus the agent install, nothing else."""
    if benchmark not in {"osworld", "osworld_v2", "scienceboard"}:
        raise ValueError(f"unsupported scratch benchmark: {benchmark!r}")
    if benchmark == "scienceboard":
        return _provision_sb_scratch_vm(
            clone_dir,
            region=region,
            readiness_timeout_s=readiness_timeout_s,
            with_shadow_setup=with_shadow_setup,
        )
    env = None
    run_name = None
    cleanup_record_path = None
    try:
        desktop_env_cls = _resolve_desktop_env(clone_dir, benchmark)
        kwargs = dict(
            provider_name="aws",
            region=region,
            os_type="Ubuntu",
            headless=True,
            require_a11y_tree=True,
        )
        v2_spec = None
        if benchmark == "osworld_v2":
            from agent_s4.harness.prime.openrouter import _parse_bool_env
            from agent_s4.vm.owner_tag import (
                OWNER_TAG_ENV,
                install_owner_tag_hook,
            )

            # The account denies TerminateInstances unless the instance's Owner
            # tag matches the caller, and that tag can only be set in
            # RunInstances itself: an untagged scratch VM can never be torn down.
            if install_owner_tag_hook() is None:
                raise RuntimeError(
                    f"{OWNER_TAG_ENV} is unset; refusing a V2 scratch VM that "
                    "could not be terminated"
                )
            v2_spec = _load_v2_scratch_spec(clone_dir, region)
            # Same addressing as the task VMs: an orchestrator outside the
            # benchmark VPC cannot route the private address.
            kwargs["use_public_ip"] = _parse_bool_env(
                "SHADOW_EVOL_USE_PUBLIC_IP",
                "false",
            )
            kwargs.update(
                snapshot_name=v2_spec.ami_id,
                screen_size=v2_spec.screen_size,
                instance_type=v2_spec.instance_type,
                volume_size=v2_spec.volume_size,
                client_password=v2_spec.client_password,
                enable_proxy=v2_spec.enable_proxy,
            )
            run_name = f"shadow-evol-scratch-{uuid.uuid4().hex}"
            cleanup_record_path = _v2_scratch_cleanup_path(clone_dir, run_name)
            _recover_v2_scratch_records(cleanup_record_path.parent, region)
            with _ACTIVE_V2_SCRATCH_LOCK:
                _ACTIVE_V2_SCRATCH_RUNS.add(run_name)
            _write_v2_scratch_record(
                cleanup_record_path,
                run_name=run_name,
                region=region,
                status="pending",
            )
            # AWS_INSTANCE_NAME is process-global. Serialize the short launch
            # window so concurrent scratch threads cannot steal each other's
            # launch-time Name tag.
            with _V2_PROVISION_LOCK:
                previous_name = os.environ.get("AWS_INSTANCE_NAME")
                os.environ["AWS_INSTANCE_NAME"] = run_name
                try:
                    env = desktop_env_cls(**kwargs)
                finally:
                    if previous_name is None:
                        os.environ.pop("AWS_INSTANCE_NAME", None)
                    else:
                        os.environ["AWS_INSTANCE_NAME"] = previous_name
        else:
            env = desktop_env_cls(**kwargs)
    except BaseException as e:
        if run_name and cleanup_record_path:
            _sweep_v2_scratch_run(
                cleanup_record_path,
                run_name=run_name,
                region=region,
            )
        if not isinstance(e, Exception):
            raise
        logger.warning("provision_scratch_vm: DesktopEnv init failed: %s", e)
        return None

    try:
        if not _wait_for_vm_server(
            env.vm_ip, env.server_port, timeout_s=readiness_timeout_s
        ):
            logger.warning(
                "provision_scratch_vm: server on %s:%s did not become ready within %ss",
                env.vm_ip,
                env.server_port,
                readiness_timeout_s,
            )
            env.close()
            if run_name and cleanup_record_path:
                _sweep_v2_scratch_run(
                    cleanup_record_path,
                    run_name=run_name,
                    region=region,
                )
            return None

        instance_id = _extract_aws_instance_id(env)
        _tag_instance(instance_id, region=region)

        if with_shadow_setup:
            _upload_parent_shadow_interface(env, clone_dir)

            # Run setup.sh so the scratch VM matches eval VMs (apt-installs
            # scrot/xdotool/xclip/curl/wmctrl on first run, idempotent after).
            from agent_s4.vm.utils import run_setup_script

            if not run_setup_script(
                env.controller, client_password=env.client_password
            ):
                raise RuntimeError("scratch shadow_interface setup.sh failed")

        return ScratchVmHandle(
            env=env,
            vm_ip=env.vm_ip,
            server_port=env.server_port,
            chromium_port=env.chromium_port,
            vnc_port=env.vnc_port,
            vlc_port=env.vlc_port,
            instance_id=instance_id,
            client_password=env.client_password,
            clone_dir=str(Path(clone_dir).resolve()),
            benchmark=benchmark,
            release=v2_spec.release if v2_spec else None,
            ami_id=v2_spec.ami_id if v2_spec else None,
            run_name=run_name,
            region=region if run_name else None,
            cleanup_record_path=(
                str(cleanup_record_path) if cleanup_record_path else None
            ),
        )
    except BaseException as e:
        logger.warning("provision_scratch_vm: post-boot setup failed: %s", e)
        try:
            env.close()
        except Exception:
            pass
        if run_name and cleanup_record_path:
            _sweep_v2_scratch_run(
                cleanup_record_path,
                run_name=run_name,
                region=region,
            )
        if not isinstance(e, Exception):
            raise
        return None


# --- ScienceBoard scratch VM -------------------------------------------------
#
# A ScienceBoard VM is an OSWorld DesktopEnv attached to a ScienceBoard-AMI
# instance (the 6 scientific apps pre-installed on the OSWorld base image). We
# boot it on the eval's network, then wire the *same* OSWorld controller the
# builder already knows. Unlike OSWorld there is no SetupController / browser
# ports — the builder launches the scientific apps itself via
# controller.execute_python_command (discovering each app's control-server
# launch the way the eval agent does).

# SB AMI user password (matches agent_s4/eval/scienceboard_prime.py).
_SB_CLIENT_PASSWORD = "osworld-public-evaluation"


def _attach_sb_controller(
    instance_id: str, *, region: str, ami_id: str, headless: bool = True
):
    """Wire an OSWorld controller to an already-booted SB-AMI instance.

    Tries the direct DesktopEnv attach first (app-agnostic, cleanest, mirrors
    OSWorld's path); on any failure falls back to the proven VManager.revert
    path the SB eval uses. Both converge on a DesktopEnv whose .controller is
    the OSWorld PythonController.

    Returns (desktop_env, vmanager_or_None). Raises if BOTH paths fail (after
    closing whatever it opened)."""
    from desktop_env.desktop_env import DesktopEnv

    # --- primary: direct DesktopEnv attach to the running instance ----------
    env = None
    try:
        env = DesktopEnv(
            provider_name="aws",
            region=region,
            path_to_vm=instance_id,
            snapshot_name=ami_id,
            action_space="pyautogui",
            headless=headless,
        )
        # _launch_sb_instance already booted the instance; _start_emulator() attaches and wires the controller without
        # creating a new instance (the AWS provider no-ops a running one).
        env._start_emulator()
        _ = env.controller  # surface a wiring failure here, before we return
        logger.info("SB scratch VM: direct DesktopEnv attach on %s", instance_id)
        return env, None
    except Exception as e:
        logger.warning(
            "SB scratch VM: direct DesktopEnv attach failed (%s); "
            "falling back to VManager.revert",
            e,
        )
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    # --- fallback: VManager.revert (the proven eval wiring) -----------------
    from sci.vm.vmanager import VManager

    manager = VManager(
        version="0.1",
        vm_path=instance_id,
        headless=headless,
        provider_name="aws",
        region=region,
        ami_id=ami_id,
    )
    manager.__enter__()  # materialises the lazy DesktopEnv lambda
    manager.revert(ami_id)  # on AWS: _start_emulator() only, wires controller
    logger.info("SB scratch VM: VManager.revert attach on %s", instance_id)
    return manager.env, manager


def _close_sb(env, vmanager) -> None:
    """Best-effort teardown for a partially-provisioned SB scratch VM."""
    try:
        if vmanager is not None:
            vmanager.__exit__(None, None, None)
        elif env is not None:
            env.close()
    except Exception:
        pass


def _launch_sb_instance(
    run_label: str, *, region: str, ami_id: str, extra_tags: dict[str, str]
) -> str:
    """Launch one SB-AMI instance on the same network as the ScienceBoard eval
    (AWS_SUBNET_ID / AWS_SECURITY_GROUP_ID, public IP) and wait until it is
    running. The :5000 daemon wait is the caller's _wait_for_vm_server."""
    import boto3

    subnet = os.environ.get("AWS_SUBNET_ID")
    group = os.environ.get("AWS_SECURITY_GROUP_ID")
    if not subnet or not group:
        raise RuntimeError("AWS_SUBNET_ID and AWS_SECURITY_GROUP_ID are required")
    tags = [
        {"Key": "Name", "Value": run_label},
        {"Key": "RunLabel", "Value": run_label},
    ]
    tags += [{"Key": k, "Value": v} for k, v in extra_tags.items()]
    ec2 = boto3.client("ec2", region_name=region)
    resp = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=os.environ.get("SB_INSTANCE_TYPE", "t3.large"),
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
    )
    instance_id = resp["Instances"][0]["InstanceId"]
    ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
    return instance_id


def _provision_sb_scratch_vm(
    clone_dir: str, *, region: str, readiness_timeout_s: int, with_shadow_setup: bool
) -> Optional[ScratchVmHandle]:
    """Provision a ScienceBoard scratch VM. Returns None on ANY failure
    (soft-degrade), mirroring the OSWorld path's contract."""
    from agent_s4.eval.scienceboard_prime import SB_AMI_ID

    region = region or os.environ.get("SB_REGION", "us-east-1")
    run_label = f"scratch-{Path(clone_dir).name}"
    # Tag for orphan discovery (scripts/ops/list_orphan_scratch_vms.py filters
    # on scratch-vm=true). Applied at LAUNCH via RunInstances TagSpecifications,
    # not a standalone ec2:CreateTags call — the SB IAM role can tag-on-launch
    # but not CreateTags, so a post-launch _tag_instance would fail UnauthorizedOperation.
    scratch_tags = {"scratch-vm": "true", "orchestrator-pid": str(os.getpid())}

    try:
        instance_id = _launch_sb_instance(
            run_label, region=region, ami_id=SB_AMI_ID, extra_tags=scratch_tags
        )
    except Exception as e:
        logger.warning("_provision_sb_scratch_vm: launch failed: %s", e)
        return None

    try:
        env, vmanager = _attach_sb_controller(
            instance_id,
            region=region,
            ami_id=SB_AMI_ID,
        )
    except Exception as e:
        logger.warning(
            "_provision_sb_scratch_vm: controller attach failed "
            "(%s); instance %s may need manual reaping",
            e,
            instance_id,
        )
        return None

    try:
        if not _wait_for_vm_server(
            env.vm_ip, env.server_port, timeout_s=readiness_timeout_s
        ):
            logger.warning(
                "_provision_sb_scratch_vm: server on %s:%s did not become "
                "ready within %ss",
                env.vm_ip,
                env.server_port,
                readiness_timeout_s,
            )
            _close_sb(env, vmanager)
            return None

        # (orphan-discovery tags were applied at launch — see scratch_tags above)
        if with_shadow_setup:
            _upload_parent_shadow_interface(env, clone_dir)
            from agent_s4.vm.utils import run_setup_script

            run_setup_script(env.controller, client_password=_SB_CLIENT_PASSWORD)

        return ScratchVmHandle(
            env=env,
            vm_ip=env.vm_ip,
            server_port=env.server_port,
            instance_id=instance_id,
            client_password=_SB_CLIENT_PASSWORD,
            clone_dir=str(Path(clone_dir).resolve()),
            vmanager=vmanager,
        )
    except Exception as e:
        logger.warning("_provision_sb_scratch_vm: post-boot setup failed: %s", e)
        _close_sb(env, vmanager)
        return None
