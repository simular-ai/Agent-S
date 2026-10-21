import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from agent_s4 import builder_sandbox
from agent_s4.builder_sandbox import landlock_abi
from agent_s4.paths import PACKAGE_ROOT


pytestmark = pytest.mark.skipif(landlock_abi() == 0, reason="Landlock unavailable")


def _run(workspace: Path, *command: str, env=None):
    return subprocess.run(
        [
            sys.executable, str(Path(builder_sandbox.__file__).resolve()),
            "--workspace", str(workspace),
            "--executable", command[0], "--", *command,
        ],
        cwd=workspace, capture_output=True, text=True, env=env,
    )


def test_landlock_builder_can_write_its_workspace(tmp_path):
    workspace = tmp_path / "OSWorldV2_Evol_haiku"
    shadow = workspace / "shadow_interface"
    shadow.mkdir(parents=True)
    result = _run(
        workspace, "/bin/sh", "-c",
        "printf ok > shadow_interface/allowed.txt",
    )
    assert result.returncode == 0, result.stderr
    assert (shadow / "allowed.txt").read_text() == "ok"


def test_landlock_builder_uses_short_private_tmp_alias(tmp_path):
    """TMPDIR must stay inside the Unix-socket pathname budget.

    Prime Agent appends daemon/worker socket names to TMPDIR, so a long
    workspace path makes bind() truncate silently and Prime then dies on the
    untruncated name. Which short directory hosts the alias depends on the
    host -- a box with no systemd user session has no /run/user/<uid> -- so
    the invariants are the length budget and the resolved target, not the
    location.
    """
    workspace = tmp_path / ("OSWorldV2_Evol_" + "deep_" * 20)
    (workspace / "shadow_interface").mkdir(parents=True)
    result = _run(
        workspace, sys.executable, "-c",
        "import os; print(os.environ['TMPDIR'])",
    )
    assert result.returncode == 0, result.stderr
    tmpdir = result.stdout.strip()
    assert Path(tmpdir).resolve() == (workspace / ".builder_tmp").resolve()
    assert len(tmpdir) <= builder_sandbox.MAX_TMPDIR_LEN, (
        f"TMPDIR {tmpdir!r} is {len(tmpdir)} characters; Prime's socket names "
        "no longer fit"
    )
    assert tmpdir != str(workspace / ".builder_tmp"), (
        "the long workspace path was used verbatim, so no alias was made"
    )


def test_short_tmp_alias_falls_back_when_there_is_no_user_runtime_dir(
    tmp_path, monkeypatch,
):
    """No /run/user/<uid> must still yield a short alias, not the long path.

    The substitute parent has to sit directly under the system temp dir:
    pytest's own tmp_path is already too long to leave room for Prime's socket
    names, which is the very budget being tested.
    """
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    parent = Path(tempfile.mkdtemp(prefix="sbt", dir=tempfile.gettempdir()))
    parent.rmdir()  # the code under test creates it, 0700, and we check that
    monkeypatch.setattr(
        builder_sandbox, "_alias_parent_candidates",
        lambda: [
            (Path("/run/user/does-not-exist"), False),
            (parent, True),
        ],
    )
    workspace = tmp_path / ("OSWorldV2_Evol_" + "deep_" * 20)
    private_tmp = workspace / ".builder_tmp"
    private_tmp.mkdir(parents=True)

    try:
        alias = builder_sandbox._short_tmp_alias(workspace, private_tmp)

        assert alias != private_tmp
        assert alias.resolve() == private_tmp.resolve()
        assert len(str(alias)) <= builder_sandbox.MAX_TMPDIR_LEN
        assert parent.stat().st_mode & 0o077 == 0
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def test_short_tmp_alias_rejects_a_parent_owned_by_someone_else(
    tmp_path, monkeypatch,
):
    """A directory we do not own could redirect the Builder's temp files."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(
        builder_sandbox, "_alias_parent_candidates",
        lambda: [(Path("/"), False)],  # a real directory, owned by root
    )
    workspace = tmp_path / "OSWorldV2_Evol_haiku"
    private_tmp = workspace / ".builder_tmp"
    private_tmp.mkdir(parents=True)

    assert builder_sandbox._short_tmp_alias(workspace, private_tmp) == private_tmp


def test_landlock_builder_cannot_modify_evaluator_checkout_files(tmp_path):
    workspace = tmp_path / "OSWorldV2_Evol_haiku"
    (workspace / "shadow_interface").mkdir(parents=True)
    protected = workspace / "desktop_env.py"
    protected.write_text("original")
    result = _run(
        workspace, "/bin/sh", "-c", "printf corrupt > desktop_env.py",
    )
    assert result.returncode != 0
    assert protected.read_text() == "original"


def test_landlock_builder_cannot_read_sibling_gated_checkout(tmp_path):
    workspace = tmp_path / "OSWorldV2_Evol_haiku"
    workspace.mkdir()
    gated = tmp_path / "OSWorldV2_Base" / "evaluation_examples/task_class"
    gated.mkdir(parents=True)
    secret = gated / "task_001.py"
    secret.write_text("gold evaluator logic")
    result = _run(workspace, "/bin/cat", str(secret))
    assert result.returncode != 0
    assert "gold evaluator logic" not in result.stdout


def test_landlock_builder_cannot_read_parent_database(tmp_path):
    workspace = tmp_path / "OSWorldV2_Evol_haiku"
    workspace.mkdir()
    database = tmp_path / "shadow_evolution_osworld_v2.db"
    database.write_text("private fitness state")
    result = _run(workspace, "/bin/cat", str(database))
    assert result.returncode != 0
    assert "private fitness state" not in result.stdout


def test_landlock_builder_can_import_only_the_framework_scratch_bridge(
        tmp_path, monkeypatch):
    import os
    import shutil

    workspace = tmp_path / "OSWorldV2_Evol_haiku"
    workspace.mkdir()
    bridge_source = PACKAGE_ROOT / "builder_scratch_vm.py"
    shutil.copy2(bridge_source, workspace / "scratch_vm.py")
    env = dict(os.environ)
    env.update({
        "SCRATCH_VM_IP": "127.0.0.1",
        "SCRATCH_VM_SERVER_PORT": "5000",
        "SCRATCH_VM_INSTANCE_ID": "i-test",
        "SCRATCH_VM_CLIENT_PASSWORD": "pw",
        "SCRATCH_VM_CLONE_DIR": str(workspace),
    })
    result = _run(
        workspace, sys.executable, "-c",
        "import scratch_vm; print('bridge-ok')",
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "bridge-ok"


def test_landlock_builder_can_reach_setup_and_redeploy_helpers(tmp_path):
    """The staged bridge must load its helpers without the parent project.

    desktop_env's controllers call load_dotenv() at import, which walks up to
    the host .env; Landlock denies that file, and redeploy_interface's helpers
    live in agent_s4, which Landlock also hides.
    """
    from agent_s4.scratch_vm import stage_builder_scratch_bridge

    (tmp_path / ".env").write_text("HOST_SECRET=do-not-leak\n")
    workspace = tmp_path / "OSWorldV2_Evol_haiku"
    controllers = workspace / "desktop_env" / "controllers"
    controllers.mkdir(parents=True)
    (workspace / "desktop_env" / "__init__.py").touch()
    (controllers / "__init__.py").touch()
    (controllers / "setup.py").write_text(
        "import dotenv\n"
        "dotenv.load_dotenv()\n"
        "class SetupController:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.kwargs = kwargs\n"
    )
    stage_builder_scratch_bridge(str(workspace))
    env = dict(os.environ)
    env.update({
        "SCRATCH_VM_IP": "127.0.0.1",
        "SCRATCH_VM_SERVER_PORT": "5000",
        "SCRATCH_VM_INSTANCE_ID": "i-test",
        "SCRATCH_VM_CLIENT_PASSWORD": "pw",
        "SCRATCH_VM_CLONE_DIR": str(workspace),
        "SCRATCH_VM_CHROMIUM_PORT": "9222",
        "SCRATCH_VM_VLC_PORT": "8080",
    })
    env.pop("HOST_SECRET", None)
    result = _run(
        workspace, sys.executable, "-c",
        "import os, scratch_vm; scratch_vm.get_setup_controller(); "
        "from scratch_vm_utils import run_setup_script, upload_shadow_interface; "
        "print('helpers-ok', os.environ.get('HOST_SECRET'))",
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "helpers-ok None"


def test_v2_builder_subprocess_is_always_wrapped(monkeypatch, tmp_path):
    import json
    from agent_s4.harness.builder_common import run_builder_subprocess

    workspace = tmp_path / "OSWorldV2_Evol_haiku"
    (workspace / "shadow_interface").mkdir(parents=True)
    (workspace / "notebook.jsonl").touch()
    captured = {}

    class Proc:
        returncode = 0

        def wait(self, timeout=None):
            return 0

    def popen(argv, **kwargs):
        captured["argv"] = argv
        kwargs["stdout"].write(json.dumps({
            "type": "result", "total_cost_usd": 0.1,
        }) + "\n")
        kwargs["stdout"].flush()
        return Proc()

    monkeypatch.setattr(subprocess, "Popen", popen)
    result = run_builder_subprocess(
        ["/bin/true"], clone_dir=workspace, logs_dir=tmp_path / "logs",
        log_name="builder", timeout=1, env={}, variant_id="haiku-if-001",
    )
    assert result.crashed is False
    assert Path(captured["argv"][1]).name == "builder_sandbox.py"
    assert "--workspace" in captured["argv"]
