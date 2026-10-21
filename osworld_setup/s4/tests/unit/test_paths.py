"""Shared-root / project-root split used by the ScienceBoard launchers."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_s4 import paths

REPO = Path(__file__).resolve().parents[4]


def test_prompt_path_prefers_project_root(tmp_path, monkeypatch):
    shared, project = tmp_path / "shared", tmp_path / "project"
    for root in (shared, project):
        (root / "prompts").mkdir(parents=True)
    (shared / "prompts" / "crossover.md").write_text("shared")
    (shared / "prompts" / "addendum.md").write_text("shared")
    (project / "prompts" / "addendum.md").write_text("project")
    monkeypatch.setattr(paths, "SHARED_ROOT", shared)
    monkeypatch.setattr(paths, "PROJECT_ROOT", project)
    assert paths.prompt_path("addendum.md").read_text() == "project"
    assert paths.prompt_path("crossover.md").read_text() == "shared"


def test_scienceboard_roots_resolve():
    """With the SB launchers' environment, run files come from
    scienceboard_setup/s4 and the Prime runtime from osworld_setup/s4."""
    sb, shared = REPO / "scienceboard_setup" / "s4", REPO / "osworld_setup" / "s4"
    code = (
        "from agent_s4 import paths\n"
        "from agent_s4.harness.prime import source as s\n"
        "from agent_s4.harness.prime.evaluator import _resolve_runtime_cache\n"
        "print(paths.PROJECT_ROOT); print(paths.SHARED_ROOT)\n"
        "print(paths.prompt_path('scratch_vm_addendum_scienceboard.md'))\n"
        "print(paths.prompt_path('crossover.md'))\n"
        "print(s.VENDORED_RUNTIME); print(_resolve_runtime_cache())\n"
    )
    env = dict(os.environ, SHADOW_EVOL_PROJECT_ROOT=str(sb), PYTHONPATH=str(REPO))
    env.pop("SHADOW_EVOL_SHARED_ROOT", None)
    env.pop("SHADOW_VM_ASSETS_DIR", None)
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True,
        check=True,
    ).stdout.splitlines()
    assert out[0] == str(sb) and out[1] == str(shared)
    assert out[2] == str(sb / "prompts" / "scratch_vm_addendum_scienceboard.md")
    assert out[3] == str(shared / "prompts" / "crossover.md")
    assert out[4] == str(shared / "agent_harness" / "prime-agent-runtime")
    assert out[5].startswith(str(shared / "vm_assets"))


def test_require_shared_root_accepts_the_checkout():
    paths.require_shared_root()


def test_require_shared_root_explains_a_missing_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SHARED_ROOT", tmp_path / "missing")
    with pytest.raises(SystemExit, match="SHADOW_EVOL_SHARED_ROOT"):
        paths.require_shared_root()


def test_cli_run_stops_before_any_work_without_a_checkout(tmp_path, monkeypatch):
    from agent_s4 import cli

    monkeypatch.setattr(paths, "SHARED_ROOT", tmp_path / "missing")
    monkeypatch.setattr(sys, "argv", [
        "agent_s4.cli", "run",
        "--island", "name=o1,evaluator_harness=prime_agent,eval_model=m",
    ])
    monkeypatch.setattr(
        cli, "build_config_from_args",
        lambda args: pytest.fail("must stop before building a config"),
    )
    with pytest.raises(SystemExit, match="setup files not found"):
        cli.main()
