from __future__ import annotations

import importlib
import sys

import pytest


def _import_bridge(monkeypatch, clone, *, osworld_ports=True):
    env = {
        "SCRATCH_VM_IP": "10.0.0.7",
        "SCRATCH_VM_SERVER_PORT": "5000",
        "SCRATCH_VM_INSTANCE_ID": "i-test",
        "SCRATCH_VM_CLIENT_PASSWORD": "pw",
        "SCRATCH_VM_CLONE_DIR": str(clone),
    }
    if osworld_ports:
        env.update({
            "SCRATCH_VM_CHROMIUM_PORT": "9222",
            "SCRATCH_VM_VNC_PORT": "8006",
            "SCRATCH_VM_VLC_PORT": "8080",
        })
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("agent_s4.builder_scratch_vm", None)
    return importlib.import_module("agent_s4.builder_scratch_vm")


def test_bridge_missing_environment_fails_at_import(monkeypatch):
    for key in (
        "SCRATCH_VM_IP", "SCRATCH_VM_SERVER_PORT", "SCRATCH_VM_INSTANCE_ID",
        "SCRATCH_VM_CLIENT_PASSWORD", "SCRATCH_VM_CLONE_DIR",
        "SCRATCH_VM_CHROMIUM_PORT", "SCRATCH_VM_VNC_PORT", "SCRATCH_VM_VLC_PORT",
    ):
        monkeypatch.delenv(key, raising=False)
    sys.modules.pop("agent_s4.builder_scratch_vm", None)
    with pytest.raises(RuntimeError, match="SCRATCH_VM_IP"):
        importlib.import_module("agent_s4.builder_scratch_vm")


def test_bridge_controller_setup_and_redeploy_contract(tmp_path, monkeypatch):
    clone = tmp_path / "clone"
    (clone / "shadow_interface").mkdir(parents=True)
    bridge = _import_bridge(monkeypatch, clone)

    controller = object()
    setup_controller = object()
    controller_calls = []
    setup_calls = []
    monkeypatch.setattr(
        "desktop_env.controllers.python.PythonController",
        lambda **kwargs: controller_calls.append(kwargs) or controller,
    )
    monkeypatch.setattr(
        "desktop_env.controllers.setup.SetupController",
        lambda **kwargs: setup_calls.append(kwargs) or setup_controller,
    )
    uploaded = []
    setup = []
    # The orchestrator stages agent_s4/vm/utils.py beside the bridge.
    import agent_s4.vm.utils
    monkeypatch.setitem(sys.modules, "scratch_vm_utils", agent_s4.vm.utils)
    monkeypatch.setattr(
        "agent_s4.vm.utils.upload_shadow_interface",
        lambda ctl, path: uploaded.append((ctl, path)) or True,
    )
    monkeypatch.setattr(
        "agent_s4.vm.utils.run_setup_script",
        lambda ctl, client_password: setup.append((ctl, client_password)) or True,
    )

    assert bridge.get_controller() is controller
    assert bridge.get_controller() is controller
    assert controller_calls == [{"vm_ip": "10.0.0.7", "server_port": 5000}]
    assert bridge.get_setup_controller() is setup_controller
    assert setup_calls[0]["chromium_port"] == 9222
    assert setup_calls[0]["vlc_port"] == 8080
    bridge.redeploy_interface()
    assert uploaded == [(controller, str(clone / "shadow_interface"))]
    assert setup == [(controller, "pw")]
    assert bridge.scratch_vm_identity()["instance_id"] == "i-test"


def test_scienceboard_bridge_has_no_setup_controller(tmp_path, monkeypatch):
    clone = tmp_path / "clone"
    (clone / "shadow_interface").mkdir(parents=True)
    bridge = _import_bridge(monkeypatch, clone, osworld_ports=False)
    with pytest.raises(bridge.ScratchVmUnavailable, match="CHROMIUM_PORT"):
        bridge.get_setup_controller()
