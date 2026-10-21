"""Tests for agent_s4.scratch_vm — scratch VM lifecycle helpers."""
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _owner_tag_hook(monkeypatch):
    """V2 scratch provisioning refuses to launch an untaggable VM."""
    monkeypatch.setattr(
        "agent_s4.vm.owner_tag.install_owner_tag_hook", lambda: "tester",
    )


class _OkResponse:
    """Minimal context-manager substitute for urllib.request.urlopen's return.
    MagicMock can't easily fake __enter__/__exit__ via instance attributes,
    because Python looks up dunder methods on the type, not the instance."""
    status = 200
    def __enter__(self): return self
    def __exit__(self, *a): return None


def test_scratch_vm_env_contains_all_required_keys():
    """scratch_vm_env returns exactly the SCRATCH_VM_* env vars the
    builder-side scratch_vm module requires, plus the interpreter to run it."""
    import sys
    from agent_s4.scratch_vm import ScratchVmHandle, scratch_vm_env

    handle = ScratchVmHandle(
        env=MagicMock(),
        vm_ip="10.0.0.5",
        server_port=5000,
        chromium_port=9222,
        vnc_port=8006,
        vlc_port=8080,
        instance_id="i-abc123",
        client_password="pw",
        clone_dir="/clone",
    )
    out = scratch_vm_env(handle)
    assert out == {
        "SCRATCH_VM_IP": "10.0.0.5",
        "SCRATCH_VM_SERVER_PORT": "5000",
        "SCRATCH_VM_CHROMIUM_PORT": "9222",
        "SCRATCH_VM_VNC_PORT": "8006",
        "SCRATCH_VM_VLC_PORT": "8080",
        "SCRATCH_VM_INSTANCE_ID": "i-abc123",
        "SCRATCH_VM_CLIENT_PASSWORD": "pw",
        "SCRATCH_VM_CLONE_DIR": "/clone",
        "SCRATCH_VM_PYTHON": sys.executable,
    }


def test_wait_for_vm_server_returns_true_on_first_http_200():
    """Returns True immediately when the first probe succeeds."""
    from agent_s4.scratch_vm import _wait_for_vm_server

    with patch("urllib.request.urlopen", return_value=_OkResponse()) as mock_urlopen:
        ok = _wait_for_vm_server("1.2.3.4", 5000, timeout_s=10, poll_interval_s=0.01)
    assert ok is True
    mock_urlopen.assert_called()


def test_wait_for_vm_server_retries_on_connection_error_then_succeeds():
    """Retries through ConnectionRefusedError / URLError and succeeds on later attempt."""
    from agent_s4.scratch_vm import _wait_for_vm_server
    import urllib.error

    side_effects = [
        ConnectionRefusedError("first"),
        urllib.error.URLError("second"),
        _OkResponse(),
    ]
    with patch("urllib.request.urlopen", side_effect=side_effects):
        ok = _wait_for_vm_server("1.2.3.4", 5000, timeout_s=5, poll_interval_s=0.01)
    assert ok is True


def test_wait_for_vm_server_returns_false_on_timeout():
    """After timeout_s of nothing-works, returns False."""
    from agent_s4.scratch_vm import _wait_for_vm_server

    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("always")):
        ok = _wait_for_vm_server("1.2.3.4", 5000, timeout_s=0.3, poll_interval_s=0.1)
    assert ok is False


def test_teardown_scratch_vm_no_op_on_none():
    """Passing None does nothing and returns without error."""
    from agent_s4.scratch_vm import teardown_scratch_vm

    # Must not raise
    assert teardown_scratch_vm(None) is None


def test_teardown_scratch_vm_calls_env_close():
    """When given a handle, calls .env.close() on it."""
    from agent_s4.scratch_vm import ScratchVmHandle, teardown_scratch_vm

    env_mock = MagicMock()
    handle = ScratchVmHandle(
        env=env_mock, vm_ip="x", server_port=5000,
        chromium_port=9222, vnc_port=8006, vlc_port=8080,
        instance_id="i-x", client_password="pw", clone_dir="/c",
    )
    teardown_scratch_vm(handle)
    env_mock.close.assert_called_once_with()


def test_teardown_scratch_vm_swallows_exceptions():
    """If env.close() raises, teardown logs the error and returns normally."""
    from agent_s4.scratch_vm import ScratchVmHandle, teardown_scratch_vm

    env_mock = MagicMock()
    env_mock.close.side_effect = RuntimeError("aws flake")
    handle = ScratchVmHandle(
        env=env_mock, vm_ip="x", server_port=5000,
        chromium_port=9222, vnc_port=8006, vlc_port=8080,
        instance_id="i-x", client_password="pw", clone_dir="/c",
    )
    # Must not raise
    assert teardown_scratch_vm(handle) is None


def _fake_desktop_env(vm_ip="10.0.0.5", port=5000, instance_id="i-abc",
                      provider_name="aws"):
    """Build a DesktopEnv-like mock ready for provision_scratch_vm to consume.

    provider_name must match what _extract_aws_instance_id asserts against —
    default 'aws' keeps tests aligned with the scratch_vm AWS-only contract.
    """
    env = MagicMock()
    env.provider_name = provider_name
    env.vm_ip = vm_ip
    env.server_port = port
    env.chromium_port = 9222
    env.vnc_port = 8006
    env.vlc_port = 8080
    env.client_password = "pw"
    env.path_to_vm = instance_id
    return env


def test_extract_aws_instance_id_rejects_non_aws_provider():
    """Regression guard: _extract_aws_instance_id must fail loudly on
    non-AWS providers so path_to_vm (a filesystem path on vmware/virtualbox)
    is never passed into ec2.create_tags as an instance ID."""
    from agent_s4.scratch_vm import _extract_aws_instance_id

    env = MagicMock()
    env.provider_name = "vmware"
    env.path_to_vm = "/var/lib/vmware/instance.vmx"
    import pytest
    with pytest.raises(AssertionError, match="vmware"):
        _extract_aws_instance_id(env)


def test_provision_scratch_vm_returns_handle_on_success():
    """Happy path: DesktopEnv boots, server becomes ready, upload succeeds,
    tagging succeeds → returns a populated ScratchVmHandle."""
    from agent_s4.scratch_vm import provision_scratch_vm

    env = _fake_desktop_env()
    with patch("agent_s4.scratch_vm.DesktopEnv", return_value=env), \
         patch("agent_s4.scratch_vm._wait_for_vm_server", return_value=True), \
         patch("agent_s4.scratch_vm._tag_instance"), \
         patch("agent_s4.scratch_vm._upload_parent_shadow_interface"), \
         patch("agent_s4.vm.utils.run_setup_script", return_value=True):
        handle = provision_scratch_vm("/clone", region="us-east-1")

    assert handle is not None
    assert handle.vm_ip == "10.0.0.5"
    assert handle.instance_id == "i-abc"
    # clone_dir is resolved via Path — compare with .resolve() to match impl
    from pathlib import Path
    assert handle.clone_dir == str(Path("/clone").resolve())


def test_provision_scratch_vm_returns_none_on_readiness_timeout():
    """If _wait_for_vm_server returns False, we tear down the partial VM
    and return None."""
    from agent_s4.scratch_vm import provision_scratch_vm

    env = _fake_desktop_env()
    with patch("agent_s4.scratch_vm.DesktopEnv", return_value=env), \
         patch("agent_s4.scratch_vm._wait_for_vm_server", return_value=False):
        handle = provision_scratch_vm("/clone")

    assert handle is None
    env.close.assert_called_once_with()


def test_provision_scratch_vm_returns_none_on_desktop_env_exception():
    """If DesktopEnv(...) raises, return None (never propagate)."""
    from agent_s4.scratch_vm import provision_scratch_vm

    with patch("agent_s4.scratch_vm.DesktopEnv",
               side_effect=RuntimeError("aws quota exceeded")):
        handle = provision_scratch_vm("/clone")
    assert handle is None


def test_provision_scratch_vm_tags_instance_and_uploads_parent_interface():
    """Verifies _tag_instance and _upload_parent_shadow_interface are called
    with the correct arguments."""
    from agent_s4.scratch_vm import provision_scratch_vm

    env = _fake_desktop_env(instance_id="i-xyz")
    with patch("agent_s4.scratch_vm.DesktopEnv", return_value=env), \
         patch("agent_s4.scratch_vm._wait_for_vm_server", return_value=True), \
         patch("agent_s4.scratch_vm._tag_instance") as tag_mock, \
         patch("agent_s4.scratch_vm._upload_parent_shadow_interface") as upload_mock, \
         patch("agent_s4.vm.utils.run_setup_script", return_value=True):
        provision_scratch_vm("/clone", region="us-west-2")

    tag_mock.assert_called_once_with("i-xyz", region="us-west-2")
    upload_mock.assert_called_once_with(env, "/clone")


def test_provision_osworld_v2_uses_explicit_release_spec(tmp_path):
    from agent_s4 import scratch_vm

    env = _fake_desktop_env()
    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return env

    spec = scratch_vm.V2ScratchVmSpec(
        release="osworld-v2-2026.08.08",
        code_tag="v2026.08.08", code_commit="deadbeef",
        ami_id="ami-v2", screen_size=(1920, 1080),
        instance_type="t3.xlarge", volume_size=None,
        client_password="osworld-public-evaluation", enable_proxy=False,
    )
    with patch("agent_s4.scratch_vm._resolve_desktop_env",
               return_value=constructor), \
         patch("agent_s4.scratch_vm._load_v2_scratch_spec",
               return_value=spec), \
         patch("agent_s4.scratch_vm._wait_for_vm_server", return_value=True), \
         patch("agent_s4.scratch_vm._tag_instance"), \
         patch("agent_s4.scratch_vm._upload_parent_shadow_interface"), \
         patch("agent_s4.vm.utils.run_setup_script", return_value=True):
        handle = scratch_vm.provision_scratch_vm(
            str(tmp_path / "OSWorldV2_Evol_haiku"),
            benchmark="osworld_v2", region="us-east-1",
        )

    assert captured["snapshot_name"] == "ami-v2"
    assert captured["instance_type"] == "t3.xlarge"
    assert captured["screen_size"] == (1920, 1080)
    assert captured["client_password"] == "osworld-public-evaluation"
    assert captured["enable_proxy"] is False
    assert handle.benchmark == "osworld_v2"
    assert handle.release == "osworld-v2-2026.08.08"
    assert handle.ami_id == "ami-v2"
    assert handle.run_name.startswith("shadow-evol-scratch-")
    record = __import__("json").loads(
        __import__("pathlib").Path(handle.cleanup_record_path).read_text()
    )
    assert record["cleanup_status"] == "pending"


def test_provision_osworld_v2_refuses_without_owner_tag(tmp_path, monkeypatch):
    """Without an Owner tag the account can never terminate the VM."""
    from agent_s4 import scratch_vm

    monkeypatch.setattr(
        "agent_s4.vm.owner_tag.install_owner_tag_hook", lambda: None,
    )
    constructor = MagicMock()
    with patch("agent_s4.scratch_vm._resolve_desktop_env",
               return_value=constructor):
        handle = scratch_vm.provision_scratch_vm(
            str(tmp_path / "OSWorldV2_Evol_haiku"), benchmark="osworld_v2",
        )
    assert handle is None
    constructor.assert_not_called()


@pytest.mark.parametrize("flag,expected", [("true", True), (None, False)])
def test_provision_osworld_v2_follows_public_ip_setting(
        tmp_path, monkeypatch, flag, expected):
    from agent_s4 import scratch_vm

    if flag is None:
        monkeypatch.delenv("SHADOW_EVOL_USE_PUBLIC_IP", raising=False)
    else:
        monkeypatch.setenv("SHADOW_EVOL_USE_PUBLIC_IP", flag)
    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return _fake_desktop_env()

    spec = scratch_vm.V2ScratchVmSpec(
        release="r", code_tag="t", code_commit="c", ami_id="ami",
        screen_size=(1920, 1080), instance_type="t3.xlarge",
        volume_size=None, client_password="pw", enable_proxy=False,
    )
    with patch("agent_s4.scratch_vm._resolve_desktop_env",
               return_value=constructor), \
         patch("agent_s4.scratch_vm._load_v2_scratch_spec", return_value=spec), \
         patch("agent_s4.scratch_vm._wait_for_vm_server", return_value=True), \
         patch("agent_s4.scratch_vm._tag_instance"), \
         patch("agent_s4.scratch_vm._upload_parent_shadow_interface"), \
         patch("agent_s4.vm.utils.run_setup_script", return_value=True):
        scratch_vm.provision_scratch_vm(
            str(tmp_path / "OSWorldV2_Evol_haiku"), benchmark="osworld_v2",
        )
    assert captured["use_public_ip"] is expected


def test_provision_scratch_vm_rejects_unknown_benchmark():
    from agent_s4.scratch_vm import provision_scratch_vm
    import pytest
    with pytest.raises(ValueError, match="unsupported scratch benchmark"):
        provision_scratch_vm("/clone", benchmark="unknown")


def test_v2_desktop_env_resolution_rejects_loaded_v1(tmp_path, monkeypatch):
    """A host-imported V1 module must fail closed, never select the V1 AMI."""
    import sys
    from types import SimpleNamespace
    from agent_s4 import scratch_vm

    checkout = tmp_path / "OSWorldV2_Evol"
    (checkout / "desktop_env").mkdir(parents=True)
    loaded_v1 = SimpleNamespace(
        __file__=str(tmp_path / "OSWorld_Evol/desktop_env/desktop_env.py"),
        DesktopEnv=object,
    )
    monkeypatch.setitem(sys.modules, "desktop_env.desktop_env", loaded_v1)
    monkeypatch.setattr(scratch_vm, "DesktopEnv", None)
    with pytest.raises(RuntimeError, match="already loaded"):
        scratch_vm._resolve_desktop_env(str(checkout), "osworld_v2")


def test_v2_desktop_env_resolution_is_stable_across_islands(tmp_path, monkeypatch):
    import sys
    from agent_s4 import scratch_vm

    template = tmp_path / "OSWorldV2_Evol"
    package = template / "desktop_env"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "desktop_env.py").write_text("class DesktopEnv: pass\n")
    monkeypatch.setattr(scratch_vm, "DesktopEnv", None)
    for name in list(sys.modules):
        if name == "desktop_env" or name.startswith("desktop_env."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    first = scratch_vm._resolve_desktop_env(
        str(tmp_path / "OSWorldV2_Evol_haiku"), "osworld_v2",
    )
    second = scratch_vm._resolve_desktop_env(
        str(tmp_path / "OSWorldV2_Evol_sonnet"), "osworld_v2",
    )
    assert first is second


def test_v2_keyboard_interrupt_after_boot_closes_and_sweeps(tmp_path):
    from agent_s4 import scratch_vm

    env = _fake_desktop_env()
    spec = scratch_vm.V2ScratchVmSpec(
        release="r", code_tag="t", code_commit="c", ami_id="ami",
        screen_size=(1920, 1080), instance_type="t3.xlarge",
        volume_size=None, client_password="pw", enable_proxy=False,
    )
    swept = []
    with patch("agent_s4.scratch_vm.DesktopEnv", return_value=env), \
         patch("agent_s4.scratch_vm._load_v2_scratch_spec", return_value=spec), \
         patch("agent_s4.scratch_vm._wait_for_vm_server",
               side_effect=KeyboardInterrupt), \
         patch("agent_s4.harness.ubuntu_vm._cleanup_v2_instances",
               side_effect=lambda region, name: swept.append((region, name)) or []):
        with pytest.raises(KeyboardInterrupt):
            scratch_vm.provision_scratch_vm(
                str(tmp_path / "OSWorldV2_Evol_haiku"), benchmark="osworld_v2",
            )
    env.close.assert_called_once_with()
    assert len(swept) == 1
    record = __import__("json").loads(next(
        (tmp_path / "scratch_cleanup_osworld_v2").glob("vm_cleanup.*.json")
    ).read_text())
    assert record["cleanup_status"] == "complete"


def test_v2_scratch_next_run_recovers_dead_local_owner(tmp_path, monkeypatch):
    import json
    from agent_s4 import scratch_vm

    cleanup_dir = tmp_path / "scratch_cleanup_osworld_v2"
    cleanup_dir.mkdir()
    path = cleanup_dir / "vm_cleanup.shadow-evol-scratch-deadbeef.json"
    path.write_text(json.dumps({
        "run_name": "shadow-evol-scratch-deadbeef",
        "cleanup_status": "pending", "owner_pid": 999999,
        "owner_host": __import__("socket").gethostname(),
        "owner_boot_id": "old", "owner_start_ticks": "old",
    }))
    swept = []
    monkeypatch.setattr(
        scratch_vm, "_sweep_v2_scratch_run",
        lambda record_path, **kwargs: swept.append((record_path, kwargs)),
    )
    scratch_vm._recover_v2_scratch_records(cleanup_dir, "us-east-1")
    assert swept == [(path, {
        "run_name": "shadow-evol-scratch-deadbeef", "region": "us-east-1",
    })]


def test_v2_scratch_recovery_never_sweeps_live_owner(tmp_path, monkeypatch):
    import json
    from agent_s4 import scratch_vm

    cleanup_dir = tmp_path / "scratch_cleanup_osworld_v2"
    cleanup_dir.mkdir()
    path = cleanup_dir / "vm_cleanup.shadow-evol-scratch-cafebabe.json"
    path.write_text(json.dumps({
        "run_name": "shadow-evol-scratch-cafebabe",
        "cleanup_status": "pending",
    }))
    monkeypatch.setattr(scratch_vm, "_record_owner_is_live", lambda record: True)
    monkeypatch.setattr(
        scratch_vm, "_sweep_v2_scratch_run",
        lambda *a, **k: pytest.fail("live owner must not be swept"),
    )
    scratch_vm._recover_v2_scratch_records(cleanup_dir, "us-east-1")


def test_provision_scratch_vm_setup_failure_closes_vm():
    from agent_s4.scratch_vm import provision_scratch_vm

    env = _fake_desktop_env()
    with patch("agent_s4.scratch_vm.DesktopEnv", return_value=env), \
         patch("agent_s4.scratch_vm._wait_for_vm_server", return_value=True), \
         patch("agent_s4.scratch_vm._tag_instance"), \
         patch("agent_s4.scratch_vm._upload_parent_shadow_interface"), \
         patch("agent_s4.vm.utils.run_setup_script", return_value=False):
        handle = provision_scratch_vm("/clone")
    assert handle is None
    env.close.assert_called_once_with()


# --- ScienceBoard scratch VM ----------------------------------------------

def test_scratch_vm_env_scienceboard_omits_osworld_ports():
    """A ScienceBoard handle has no chromium/vnc/vlc ports, so scratch_vm_env
    emits only the five common SCRATCH_VM_* vars the SB builder-side module
    reads, plus the interpreter to run it — never the OSWorld browser/media
    ports."""
    import sys
    from agent_s4.scratch_vm import ScratchVmHandle, scratch_vm_env

    handle = ScratchVmHandle(
        env=MagicMock(),
        vm_ip="10.0.0.9",
        server_port=5000,
        instance_id="i-sb01",
        client_password="osworld-public-evaluation",
        clone_dir="/clone",
        # chromium_port / vnc_port / vlc_port / vmanager default to None
    )
    out = scratch_vm_env(handle)
    assert out == {
        "SCRATCH_VM_IP": "10.0.0.9",
        "SCRATCH_VM_SERVER_PORT": "5000",
        "SCRATCH_VM_INSTANCE_ID": "i-sb01",
        "SCRATCH_VM_CLIENT_PASSWORD": "osworld-public-evaluation",
        "SCRATCH_VM_CLONE_DIR": "/clone",
        "SCRATCH_VM_PYTHON": sys.executable,
    }
    for k in ("SCRATCH_VM_CHROMIUM_PORT", "SCRATCH_VM_VNC_PORT",
              "SCRATCH_VM_VLC_PORT"):
        assert k not in out


def test_provision_scratch_vm_scienceboard_dispatches_to_sb_path():
    """benchmark='scienceboard' routes to _provision_sb_scratch_vm rather than
    the OSWorld DesktopEnv path."""
    from agent_s4 import scratch_vm

    with patch("agent_s4.scratch_vm._provision_sb_scratch_vm",
               return_value="SENTINEL") as sb_mock, \
         patch("agent_s4.scratch_vm.DesktopEnv") as osworld_mock:
        out = scratch_vm.provision_scratch_vm(
            "/clone", region="us-east-1", benchmark="scienceboard",
        )
    assert out == "SENTINEL"
    sb_mock.assert_called_once()
    osworld_mock.assert_not_called()      # OSWorld path untouched


def test_provision_scratch_vm_scienceboard_dryrun_returns_none():
    """Under SHADOW_EVOL_DRYRUN=1 the SB launch short-circuits before any AWS
    call, so provision_scratch_vm(benchmark='scienceboard') soft-degrades to
    None — exercising dispatch + lazy import with zero AWS spend."""
    from agent_s4.scratch_vm import provision_scratch_vm

    with patch.dict("os.environ", {"SHADOW_EVOL_DRYRUN": "1"}):
        handle = provision_scratch_vm(
            "/clone", region="us-east-1", benchmark="scienceboard",
        )
    assert handle is None


# --- launch-time tagging (SB scratch VM tags at launch, not via CreateTags) ---

def _fake_ec2_capturing(captured):
    class _FakeEC2:
        def run_instances(self, **kw):
            captured.update(kw)
            return {"Instances": [{"InstanceId": "i-tagtest"}]}
        def get_waiter(self, name):
            class _W:
                def wait(self, **k): pass
            return _W()
        def describe_instances(self, **kw):
            return {"Reservations": [{"Instances": [{"PrivateIpAddress": "10.9.9.9"}]}]}
    return _FakeEC2()


def test_launch_sb_instance_applies_extra_tags_at_launch(monkeypatch):
    """extra_tags ride along in RunInstances' TagSpecifications, so the scratch
    VM gets its scratch-vm/orchestrator-pid tags WITHOUT a standalone
    ec2:CreateTags call (which the SB IAM role isn't granted)."""
    import boto3
    captured = {}
    monkeypatch.setattr(boto3, "client", lambda *a, **k: _fake_ec2_capturing(captured))
    monkeypatch.setenv("AWS_SUBNET_ID", "subnet-test")
    monkeypatch.setenv("AWS_SECURITY_GROUP_ID", "sg-test")

    from agent_s4.scratch_vm import _launch_sb_instance
    iid = _launch_sb_instance(
        "scratch-clone-x", region="us-east-1", ami_id="ami-test",
        extra_tags={"scratch-vm": "true", "orchestrator-pid": "4242"},
    )
    assert iid == "i-tagtest"
    assert captured["ImageId"] == "ami-test"
    nic = captured["NetworkInterfaces"][0]
    assert nic["SubnetId"] == "subnet-test" and nic["Groups"] == ["sg-test"]
    tags = {t["Key"]: t["Value"] for t in captured["TagSpecifications"][0]["Tags"]}
    assert tags["scratch-vm"] == "true"
    assert tags["orchestrator-pid"] == "4242"
    assert tags["Name"] == "scratch-clone-x"     # base tags preserved
    assert tags["RunLabel"] == "scratch-clone-x"


def test_launch_sb_instance_requires_network_ids(monkeypatch):
    monkeypatch.delenv("AWS_SUBNET_ID", raising=False)
    monkeypatch.delenv("AWS_SECURITY_GROUP_ID", raising=False)

    from agent_s4.scratch_vm import _launch_sb_instance
    with pytest.raises(RuntimeError, match="AWS_SUBNET_ID"):
        _launch_sb_instance("x", region="us-east-1", ami_id="ami-test", extra_tags={})
