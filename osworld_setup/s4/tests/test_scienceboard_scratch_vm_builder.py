"""Contract test for the framework-owned builder scratch bridge.

The real module reads the SCRATCH_VM_* env vars at import time and exposes
get_controller()/redeploy_interface() to the builder. We load it by path with a
fake desktop_env injected so the test needs no real VM and no network — it locks
the contract: read the five common vars, build PythonController(vm_ip,
server_port), and raise ScratchVmUnavailable when a required var is missing.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
from agent_s4.paths import PACKAGE_ROOT

REPO_ROOT = Path(__file__).resolve().parents[1]
SB_SCRATCH_VM = PACKAGE_ROOT / "builder_scratch_vm.py"

_REQUIRED_ENV = {
    "SCRATCH_VM_IP": "10.1.2.3",
    "SCRATCH_VM_SERVER_PORT": "5000",
    "SCRATCH_VM_INSTANCE_ID": "i-sbtest",
    "SCRATCH_VM_CLIENT_PASSWORD": "osworld-public-evaluation",
    "SCRATCH_VM_CLONE_DIR": "/tmp/clone",
}


def _install_fake_desktop_env(monkeypatch, captured):
    """Inject a fake desktop_env.controllers.python.PythonController."""
    class _FakePythonController:
        def __init__(self, vm_ip, server_port):
            captured["vm_ip"] = vm_ip
            captured["server_port"] = server_port

    pkg = types.ModuleType("desktop_env")
    ctrl_pkg = types.ModuleType("desktop_env.controllers")
    py_mod = types.ModuleType("desktop_env.controllers.python")
    py_mod.PythonController = _FakePythonController
    monkeypatch.setitem(sys.modules, "desktop_env", pkg)
    monkeypatch.setitem(sys.modules, "desktop_env.controllers", ctrl_pkg)
    monkeypatch.setitem(sys.modules, "desktop_env.controllers.python", py_mod)


def _load_module(name):
    spec = importlib.util.spec_from_file_location(name, SB_SCRATCH_VM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # reads env vars here
    return mod


def test_sb_builder_get_controller_uses_ip_and_port(monkeypatch):
    captured = {}
    for k, v in _REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    _install_fake_desktop_env(monkeypatch, captured)

    mod = _load_module("_sb_scratch_vm_ok")
    c = mod.get_controller()

    assert captured == {"vm_ip": "10.1.2.3", "server_port": 5000}
    assert c is mod.get_controller()   # cached singleton


def test_sb_builder_setup_controller_fails_without_osworld_ports(monkeypatch):
    """The shared bridge fails loudly when an SB handle omits OSWorld ports."""
    for k, v in _REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    _install_fake_desktop_env(monkeypatch, {})

    mod = _load_module("_sb_scratch_vm_nosetup")
    assert hasattr(mod, "get_controller")
    assert hasattr(mod, "redeploy_interface")
    with pytest.raises(RuntimeError, match="SCRATCH_VM_CHROMIUM_PORT"):
        mod.get_setup_controller()


def test_sb_builder_missing_env_raises(monkeypatch):
    """A missing required SCRATCH_VM_* var makes import fail loudly with
    ScratchVmUnavailable (builder then falls back to eval_results/)."""
    for k in _REQUIRED_ENV:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError):   # ScratchVmUnavailable subclasses RuntimeError
        _load_module("_sb_scratch_vm_missing")
