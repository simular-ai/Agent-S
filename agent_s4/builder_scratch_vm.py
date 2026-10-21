"""Builder-side bridge to the scratch VM provisioned by the orchestrator.

This module is imported by the host-side Builder subprocess. It never creates
or destroys a VM: the orchestrator owns that lifecycle and injects the exact
connection parameters through ``SCRATCH_VM_*`` environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path


class ScratchVmUnavailable(RuntimeError):
    """Raised when a Builder is launched without a complete scratch handle."""


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ScratchVmUnavailable(
            f"{name} is missing; this Builder has no usable scratch VM"
        )
    return value


def _required_port(name: str) -> int:
    value = _required(name)
    try:
        port = int(value)
    except ValueError as exc:
        raise ScratchVmUnavailable(f"{name} is not an integer: {value!r}") from exc
    if not 1 <= port <= 65535:
        raise ScratchVmUnavailable(f"{name} is outside the TCP port range: {port}")
    return port


_VM_IP = _required("SCRATCH_VM_IP")
_SERVER_PORT = _required_port("SCRATCH_VM_SERVER_PORT")
_INSTANCE_ID = _required("SCRATCH_VM_INSTANCE_ID")
_CLIENT_PASSWORD = _required("SCRATCH_VM_CLIENT_PASSWORD")
_CLONE_DIR = Path(_required("SCRATCH_VM_CLONE_DIR")).resolve()
_controller = None
_setup_controller = None


def get_controller():
    """Return a cached PythonController connected to the scratch desktop."""
    global _controller
    if _controller is None:
        from desktop_env.controllers.python import PythonController

        _controller = PythonController(vm_ip=_VM_IP, server_port=_SERVER_PORT)
    return _controller


def _import_setup_controller():
    """Import SetupController without its import-time ``load_dotenv()``.

    desktop_env's controllers call ``dotenv.load_dotenv()`` at import, which
    walks up past the workspace to the host project's ``.env``. That file holds
    host secrets and Landlock denies it, so the import would fail with
    PermissionError; skip the load instead of widening the sandbox.
    """
    import dotenv

    original = dotenv.load_dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False
    try:
        from desktop_env.controllers.setup import SetupController
    finally:
        dotenv.load_dotenv = original
    return SetupController


def get_setup_controller():
    """Return OSWorld's cached SetupController.

    ScienceBoard scratch VMs intentionally omit browser/media ports; calling
    this function on such a handle fails loudly instead of constructing a
    partially configured controller.
    """
    global _setup_controller
    if _setup_controller is None:
        chromium_port = _required_port("SCRATCH_VM_CHROMIUM_PORT")
        vlc_port = _required_port("SCRATCH_VM_VLC_PORT")
        SetupController = _import_setup_controller()
        _setup_controller = SetupController(
            vm_ip=_VM_IP,
            server_port=_SERVER_PORT,
            chromium_port=chromium_port,
            vlc_port=vlc_port,
            cache_dir=str(_CLONE_DIR / ".scratch_vm_cache"),
            client_password=_CLIENT_PASSWORD,
            screen_width=1920,
            screen_height=1080,
        )
    return _setup_controller


def redeploy_interface() -> None:
    """Upload the Builder's current ``shadow_interface/`` and run setup.sh."""
    shadow_dir = _CLONE_DIR / "shadow_interface"
    if not shadow_dir.is_dir():
        raise ScratchVmUnavailable(
            f"shadow_interface directory is missing from {_CLONE_DIR}"
        )
    # Staged beside this file by the orchestrator: the parent project's
    # agent_s4 package is outside the Builder sandbox.
    from scratch_vm_utils import run_setup_script, upload_shadow_interface

    controller = get_controller()
    if not upload_shadow_interface(controller, str(shadow_dir)):
        raise RuntimeError("scratch VM interface upload failed")
    if not run_setup_script(controller, client_password=_CLIENT_PASSWORD):
        raise RuntimeError("scratch VM interface setup failed")


def scratch_vm_identity() -> dict[str, str | int]:
    """Return non-secret connection identity for Builder notes/debugging."""
    return {
        "instance_id": _INSTANCE_ID,
        "vm_ip": _VM_IP,
        "server_port": _SERVER_PORT,
        "clone_dir": str(_CLONE_DIR),
    }
