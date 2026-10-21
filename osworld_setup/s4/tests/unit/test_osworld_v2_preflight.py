from __future__ import annotations

import json
from pathlib import Path

from agent_s4 import osworld_v2_preflight as preflight


def _release_pin(release_path, *, task_digest="unused", splits=None,
                 task_count=0, ami="ami-test"):
    splits = splits or {
        "osworld_v2_train_50.json": [],
        "osworld_v2_val_25.json": [],
        "osworld_v2_test.json": [],
    }
    return {
        "schema_version": 1,
        "release": "osworld-v2-2026.08.08",
        "status": "active",
        "code_repository": "xlang-ai/OSWorld-V2",
        "code_tag": "v2026.08.08",
        "code_commit": "deadbeef",
        "release_manifest_sha256": preflight._sha256(release_path),
        "task_set_sha256": task_digest,
        "split_sha256": {
            name: preflight.split_set_digest(values)
            for name, values in splits.items()
        },
        "task_count": task_count,
        "split_sizes": [50, 25, 33],
        "website_suffix": "site.hku.icu",
        "aws_ami": {"us-east-1": {"1920x1080": ami}},
        "asset_min_files": 1,
        "asset_min_bytes": 1,
    }


def test_proxy_preflight_rejects_placeholder_credentials(tmp_path):
    path = tmp_path / "proxy.json"
    path.write_text(json.dumps([{
        "host": "proxy.example", "port": 8080,
        "username": "your_username", "password": "your_password",
    }]))
    assert "placeholder" in preflight._check_proxy(path)


def test_exact_non_proxy_subset_does_not_require_proxy(tmp_path):
    task_dir = tmp_path / "OSWorldV2_Base/evaluation_examples/task_class"
    split_dir = tmp_path / "eval_sets"
    task_dir.mkdir(parents=True)
    split_dir.mkdir()
    (task_dir / "task_034.py").write_text("proxy = False\n")
    (task_dir / "task_037.py").write_text("proxy = True\n")
    (split_dir / "osworld_v2_train_50.json").write_text('["034", "037"]')
    (split_dir / "osworld_v2_val_25.json").write_text("[]")
    (split_dir / "osworld_v2_test.json").write_text("[]")

    assert not preflight.selected_tasks_require_proxy(tmp_path, {"034"})
    assert preflight.selected_tasks_require_proxy(tmp_path, {"037"})
    assert preflight.selected_tasks_require_proxy(tmp_path, {"034", "037"})
    assert preflight.selected_tasks_require_proxy(tmp_path, None)


def test_selected_harness_limits_unrelated_runtime_checks(
        tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "parent_pin_git_errors", lambda root: [])
    monkeypatch.setattr(preflight, "builder_sandbox_errors", lambda: [])
    monkeypatch.setattr(preflight, "load_release_pin", lambda root: {
        "release": "osworld-v2-2026.08.08",
    })
    captured = {}

    def validate_workspace(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(preflight, "validate_workspace", validate_workspace)
    assert preflight.validate_default_workspace(
        tmp_path, required_harnesses=("prime_agent",),
    ) == []
    assert captured["required_harnesses"] == ("prime_agent",)


def _write_hf_asset(root: Path, relative: str, data: bytes, revision: str):
    asset = root / relative
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(data)
    metadata = root / ".cache/huggingface/download" / f"{relative}.metadata"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        f"{revision}\n{preflight._git_blob_sha1(asset)}\n0\n"
    )
    return asset, metadata


def test_asset_snapshot_verifies_revision_and_content_offline(tmp_path):
    revision = "a" * 40
    asset, metadata = _write_hf_asset(
        tmp_path, "task_001/state.json", b"{}", revision,
    )
    assert preflight.asset_snapshot_errors(tmp_path, revision) == []
    asset.write_bytes(b"tampered")
    assert any(
        "checksum mismatch" in error
        for error in preflight.asset_snapshot_errors(tmp_path, revision)
    )
    asset.write_bytes(b"{}")
    metadata.write_text(f"{'b' * 40}\n{preflight._git_blob_sha1(asset)}\n0\n")
    assert any(
        "expected" in error
        for error in preflight.asset_snapshot_errors(tmp_path, revision)
    )


def test_task_source_gate_compiles_without_execution(tmp_path):
    valid = tmp_path / "task_001.py"
    invalid = tmp_path / "task_002.py"
    valid.write_text("raise RuntimeError('must not execute')\n")
    invalid.write_text("def broken(:\n")
    assert preflight.task_source_errors([valid]) == []
    assert "does not compile" in preflight.task_source_errors([invalid])[0]


def test_explicit_empty_user_sim_key_env_does_not_fallback_to_openai():
    env = {
        "OSWORLD_USER_SIM_API_KEY_ENV": "CUSTOM_SIM_KEY",
        "CUSTOM_SIM_KEY": "",
        "OPENAI_API_KEY": "would-otherwise-pass",
    }
    assert preflight._has_user_sim_credentials(env) is False


def test_current_workspace_gate_fails_locally_before_api(
        tmp_path, monkeypatch):
    # Use explicit empty asset/runtime fixtures. The validator is stdlib-only
    # and may invoke local git, but has no network/provider code path.
    assets = tmp_path / "assets"
    assets.mkdir()
    proxy = tmp_path / "proxy.json"
    proxy.write_text(json.dumps([{
        "host": "proxy.example", "port": 8080,
        "username": "real", "password": "secret",
    }]))
    monkeypatch.setenv("GITLAB_URL", "https://gitlab.example")
    monkeypatch.setenv("GITLAB_PRIVATE_TOKEN", "secret")
    monkeypatch.setenv("OSWORLD_USER_SIM_API_KEY", "secret")
    monkeypatch.setenv("OSWORLD_EVAL_MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("OSWORLD_EVAL_MODEL_NAME", "openai/gpt-4o")
    monkeypatch.setenv("OSWORLD_EVAL_MODEL_API_KEY", "secret")

    root = tmp_path / "repo"
    base = root / "OSWorldV2_Base"
    evol = root / "OSWorldV2_Evol"
    (base / "benchmark_releases").mkdir(parents=True)
    (base / "evaluation_examples/task_class").mkdir(parents=True)
    (base / "desktop_env/providers/aws").mkdir(parents=True)
    evol.mkdir(parents=True)
    (root / "eval_sets").mkdir()
    release_path = base / "benchmark_releases/osworld-v2-2026.08.08.json"
    release_path.write_text(json.dumps({
        "release": "osworld-v2-2026.08.08", "status": "active",
        "osworld_code": {
            "repository": "xlang-ai/OSWorld-V2", "tag": "v2026.08.08",
        },
        "task_hash_manifest": {"task_count": 0},
        "provider_images": {"aws": {"ubuntu": {"us-east-1": {
            "1920x1080": {"ami_id": "ami-test"},
        }}}},
    }))
    (base / "desktop_env/providers/aws/manager.py").write_text("AMI = 'ami-test'\n")
    (root / "eval_sets/osworld_v2_train_50.json").write_text(json.dumps([
        f"unknown-{index}" for index in range(50)
    ]))
    (root / "eval_sets/osworld_v2_val_25.json").write_text("[]")
    (root / "eval_sets/osworld_v2_test.json").write_text("[]")
    monkeypatch.setattr(preflight, "_exact_tag", lambda repo: "v2026.08.08")
    monkeypatch.setattr(preflight, "_worktree_clean", lambda repo: False)

    errors = preflight.validate_workspace(
        root=root,
        release_name="osworld-v2-2026.08.08",
        assets_dir=assets,
        website_suffix="web.hku.icu",
        proxy_config=proxy,
        release_pin=_release_pin(release_path),
    )
    assert any("WEBSITE_HOST_SUFFIX" in error for error in errors)
    assert any("asset cache is incomplete" in error for error in errors)
    assert any("Prime Agent VM runtime is missing" in error for error in errors)
    assert any("expected 25" in error for error in errors)
    assert any("IDs do not exactly match" in error for error in errors)
    assert any("tracked or untracked changes" in error for error in errors)


def test_complete_release_fixture_passes_all_local_structural_gates(
        tmp_path, monkeypatch):
    root = tmp_path / "repo"
    base = root / "OSWorldV2_Base"
    evol = root / "OSWorldV2_Evol"
    task_dir = base / "evaluation_examples/task_class"
    manager = base / "desktop_env/providers/aws/manager.py"
    release_dir = base / "benchmark_releases"
    split_dir = root / "eval_sets"
    for directory in (task_dir, manager.parent, release_dir, split_dir, evol):
        directory.mkdir(parents=True, exist_ok=True)

    ids = [f"{index:03d}" for index in range(1, 109)]
    task_paths = []
    for task_id in ids:
        path = task_dir / f"task_{task_id}.py"
        path.write_text(f"# task {task_id}\n")
        task_paths.append(path)
    task_manifest_path = base / "evaluation_examples/test_v2.json"
    task_manifest_path.write_text(json.dumps({"tasks": ids}))
    digest = preflight.task_set_digest(task_paths)
    splits = {
        "osworld_v2_train_50.json": ids[:50],
        "osworld_v2_val_25.json": ids[50:75],
        "osworld_v2_test.json": ids[75:],
    }
    for name, values in splits.items():
        (split_dir / name).write_text(json.dumps(values))
    release = {
        "release": "osworld-v2-2026.08.08", "status": "active",
        "osworld_code": {
            "repository": "xlang-ai/OSWorld-V2", "tag": "v2026.08.08",
        },
        "task_hash_manifest": {"task_count": 108},
        "provider_images": {"aws": {"ubuntu": {"us-east-1": {
            "1920x1080": {"ami_id": "ami-release"},
        }}}},
    }
    release_path = release_dir / "osworld-v2-2026.08.08.json"
    release_path.write_text(json.dumps(release))
    release_pin = _release_pin(
        release_path, task_digest=digest, splits=splits,
        task_count=108, ami="ami-release",
    )
    release_pin["task_manifest_sha256"] = preflight._sha256(task_manifest_path)
    manager.write_text("DEFAULT_AMI = 'ami-release'\n")

    proxy = tmp_path / "proxy.json"
    proxy.write_text(json.dumps([{"host": "proxy.example", "port": 8080}]))
    assets = tmp_path / "assets"
    assets.mkdir()
    monkeypatch.setattr(preflight, "_exact_tag", lambda repo: "v2026.08.08")
    monkeypatch.setattr(preflight, "_worktree_clean", lambda repo: True)
    monkeypatch.setattr(preflight, "_dir_stats", lambda path: (3185, 4 * 1024 ** 3))
    def git_output(repo, *args):
        if args == ("rev-parse", "HEAD"):
            return "deadbeef"
        if args == ("remote", "get-url", "origin"):
            return "https://github.com/xlang-ai/OSWorld-V2.git"
        if args == ("rev-parse", "--absolute-git-dir"):
            return str(repo / ".git")
        raise AssertionError(args)
    monkeypatch.setattr(preflight, "_git_output", git_output)
    for name in (
        "GITLAB_URL", "GITLAB_PRIVATE_TOKEN", "OSWORLD_USER_SIM_API_KEY",
        "OSWORLD_EVAL_MODEL_API_KEY",
    ):
        monkeypatch.setenv(name, "configured")
    monkeypatch.setenv("OSWORLD_EVAL_MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("OSWORLD_EVAL_MODEL_NAME", "openai/gpt-4o")

    assert preflight.validate_workspace(
        root=root, release_name="osworld-v2-2026.08.08",
        assets_dir=assets,
        website_suffix="site.hku.icu", proxy_config=proxy,
        release_pin=release_pin,
        # The Prime VM runtime archive is validated by its own tests.
        required_harnesses=(),
    ) == []

    # OpenCode's VM archive is checked only when the run uses OpenCode.
    def opencode_errors():
        return [
            error
            for error in preflight.validate_workspace(
                root=root,
                release_name="osworld-v2-2026.08.08",
                assets_dir=assets,
                website_suffix="site.hku.icu",
                proxy_config=proxy,
                release_pin=release_pin,
                required_harnesses=("opencode",),
            )
            if "OpenCode" in error
        ]

    assert opencode_errors()
    from agent_s4.harness.opencode.source import OPENCODE_VERSION, VM_BINARY
    import hashlib

    archive = root / "vm_assets/opencode_runtime.tar.gz"
    archive.parent.mkdir()
    archive.write_bytes(b"opencode runtime fixture")
    archive.with_suffix(".gz.metadata.json").write_text(
        json.dumps(
            {
                "opencode_version": OPENCODE_VERSION,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "binary": VM_BINARY,
            }
        )
    )
    assert opencode_errors() == []


def test_parent_pin_rejects_evol_task_leak(tmp_path, monkeypatch):
    """Evaluator gold files must never be copied into a builder-visible tree."""
    root = tmp_path / "repo"
    base = root / "OSWorldV2_Base"
    evol = root / "OSWorldV2_Evol"
    for path in (
        base / "benchmark_releases",
        base / "evaluation_examples/task_class",
        base / "desktop_env/providers/aws",
        evol / "evaluation_examples/task_class",
        root / "eval_sets",
    ):
        path.mkdir(parents=True, exist_ok=True)
    release_path = base / "benchmark_releases/osworld-v2-2026.08.08.json"
    release_path.write_text(json.dumps({
        "release": "osworld-v2-2026.08.08", "status": "active",
        "osworld_code": {
            "repository": "xlang-ai/OSWorld-V2", "tag": "v2026.08.08",
        },
        "task_hash_manifest": {"task_count": 0},
        "provider_images": {"aws": {"ubuntu": {"us-east-1": {
            "1920x1080": {"ami_id": "ami-test"},
        }}}},
    }))
    (base / "desktop_env/providers/aws/manager.py").write_text("ami-test")
    (evol / "evaluation_examples/task_class/task_001.py").write_text("# gold")
    for name, values in {
        "osworld_v2_train_50.json": [f"x{i}" for i in range(50)],
        "osworld_v2_val_25.json": [f"y{i}" for i in range(25)],
        "osworld_v2_test.json": [f"z{i}" for i in range(33)],
    }.items():
        (root / "eval_sets" / name).write_text(json.dumps(values))
    monkeypatch.setattr(preflight, "_exact_tag", lambda repo: "v2026.08.08")
    monkeypatch.setattr(preflight, "_worktree_clean", lambda repo: True)
    monkeypatch.setattr(preflight, "_git_output", lambda repo, *args: {
        ("rev-parse", "HEAD"): "deadbeef",
        ("remote", "get-url", "origin"): "xlang-ai/OSWorld-V2",
        ("rev-parse", "--absolute-git-dir"): str(repo / ".git"),
    }[args])
    errors = preflight.validate_workspace(
        root=root, release_name="osworld-v2-2026.08.08",
        assets_dir=tmp_path / "assets", website_suffix="site.hku.icu",
        proxy_config=tmp_path / "proxy.json",
        release_pin=_release_pin(release_path),
    )
    assert any("Builder-visible" in error and "gated" in error for error in errors)


def test_cli_preflight_checks_builder_landlock_like_the_run_gate(monkeypatch):
    """README documents --preflight-only as covering Builder Landlock. It only
    ever ran inside validate_default_workspace, so the CLI could report a
    clean workspace that the orchestrator would then refuse to run."""
    monkeypatch.setattr(preflight, "builder_sandbox_errors",
                        lambda: ["Landlock is unavailable; sentinel"])
    monkeypatch.setattr(preflight, "parent_pin_git_errors", lambda root: [])
    monkeypatch.setattr(preflight, "validate_workspace", lambda **kwargs: [])
    assert preflight.main([]) == 1


def test_builder_sandbox_errors_reports_unavailable_landlock(monkeypatch):
    import agent_s4.builder_sandbox as sandbox
    monkeypatch.setattr(sandbox, "landlock_abi", lambda: 0)
    assert preflight.builder_sandbox_errors() == [
        "Landlock is unavailable; formal V2 Builders cannot be isolated "
        "from gated tasks/DB/eval cache"
    ]
    monkeypatch.setattr(sandbox, "landlock_abi", lambda: 3)
    assert preflight.builder_sandbox_errors() == []


def test_no_library_module_imports_from_scripts():
    """agent_s4 must be importable from an installed wheel, where scripts/
    is not packaged. The V2 run gate used to live there, so `agent_s4.cli
    run --benchmark osworld_v2` raised ModuleNotFoundError off a wheel."""
    offenders = []
    for path in preflight.PACKAGE_ROOT.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith(("import scripts", "from scripts")):
                rel = path.relative_to(preflight.PROJECT_ROOT)
                offenders.append(f"{rel}:{lineno}: {stripped}")
    assert offenders == []
