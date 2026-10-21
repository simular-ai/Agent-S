#!/usr/bin/env python3
"""Strict, local-only preflight for the pinned OSWorld V2 native suite.

This module deliberately performs no network, AWS, model-provider, or VM
calls. It validates the inputs that can be proven from the workspace and
environment before a costly evaluation is allowed to start.

It lives in the package rather than under scripts/ because the orchestrator
imports validate_default_workspace() as a hard gate before any run. As a
scripts/ module it was unimportable from an installed wheel or console script,
since only agent_s4* is packaged.

Also runnable directly:  python -m agent_s4.osworld_v2_preflight --help
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from agent_s4.paths import PACKAGE_ROOT, PROJECT_ROOT  # noqa: F401

DEFAULT_RELEASE = "osworld-v2-2026.08.08"
PIN_RELATIVE_PATH = Path("config/osworld_v2_release_pin.json")
REQUIRED_V2_TASK_MODULES = ("gitlab", "imageio_ffmpeg")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_set_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _exact_tag(repo: Path) -> str | None:
    proc = subprocess.run(
        ["git", "describe", "--tags", "--exact-match"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _git_output(repo: Path, *args: str) -> str | None:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _repository_slug(url: str | None) -> str | None:
    """Normalize HTTPS/SSH git remotes to an owner/repository slug."""
    if not url:
        return None
    value = url.strip().removesuffix(".git").rstrip("/")
    if "://" in value:
        value = urlparse(value).path
    elif "@" in value and ":" in value:
        value = value.split(":", 1)[1]
    return value.strip("/").removesuffix(".git")


def load_release_pin(root: Path) -> dict:
    pin_path = root / PIN_RELATIVE_PATH
    data = json.loads(pin_path.read_text())
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or data.get("status") != "active"
    ):
        raise ValueError("expected active schema_version=1 object")
    return data


def parent_pin_git_errors(root: Path) -> list[str]:
    """Require the parent-owned release pin to be committed and unchanged."""
    relative = str(PIN_RELATIVE_PATH)
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        return [f"parent release pin is not tracked by Git: {relative}"]
    changed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", relative],
        cwd=root,
    )
    if changed.returncode != 0:
        return [f"parent release pin differs from HEAD: {relative}"]
    return []


def _worktree_clean(repo: Path) -> bool:
    """Return whether Git reports no tracked or visible untracked changes.

    Release-owned ignored caches (the gated tasks/assets supplied out of band)
    remain outside this check and are validated separately below.
    """
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and not proc.stdout.strip()


def _dir_stats(path: Path) -> tuple[int, int]:
    if not path.is_dir():
        return 0, 0
    files = [
        p
        for p in path.rglob("*")
        if p.is_file() and ".cache" not in p.parts and "__pycache__" not in p.parts
    ]
    return len(files), sum(p.stat().st_size for p in files)


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(
        b"blob " + str(len(data)).encode() + b"\0" + data,
        usedforsecurity=False,
    ).hexdigest()


def asset_snapshot_errors(path: Path, expected_revision: str) -> list[str]:
    """Verify a Hugging Face local-dir snapshot without contacting the Hub."""
    errors: list[str] = []
    metadata_root = path / ".cache" / "huggingface" / "download"
    if not metadata_root.is_dir():
        return [f"asset cache lacks Hugging Face local-dir provenance: {metadata_root}"]
    incomplete = sorted(metadata_root.rglob("*.incomplete"))
    if incomplete:
        errors.append(f"asset cache has {len(incomplete)} incomplete download(s)")

    assets = [
        asset
        for asset in path.rglob("*")
        if asset.is_file()
        and ".cache" not in asset.parts
        and "__pycache__" not in asset.parts
    ]
    metadata_paths = sorted(metadata_root.rglob("*.metadata"))
    if len(metadata_paths) != len(assets):
        errors.append(
            f"asset provenance covers {len(metadata_paths)}/{len(assets)} files"
        )

    for asset in assets:
        relative = asset.relative_to(path)
        metadata_path = metadata_root / (relative.as_posix() + ".metadata")
        try:
            lines = metadata_path.read_text().splitlines()
        except OSError as exc:
            errors.append(f"asset lacks metadata: {relative}: {exc}")
            continue
        if len(lines) < 2:
            errors.append(f"asset metadata is malformed: {metadata_path}")
            continue
        revision, etag = lines[:2]
        if revision != expected_revision:
            errors.append(
                f"asset {relative} is from revision {revision!r}, expected "
                f"{expected_revision!r}"
            )
            continue
        if len(etag) == 64:
            actual = _sha256(asset)
        elif len(etag) == 40:
            actual = _git_blob_sha1(asset)
        else:
            errors.append(f"asset metadata has unsupported ETag: {relative}: {etag!r}")
            continue
        if actual != etag:
            errors.append(f"asset content checksum mismatch: {relative}")
    return errors


def task_source_errors(paths: list[Path]) -> list[str]:
    """Compile gated task sources without importing or executing them."""
    errors = []
    for path in paths:
        try:
            compile(path.read_bytes(), str(path), "exec")
        except (OSError, SyntaxError, ValueError) as exc:
            errors.append(f"task source does not compile: {path.name}: {exc}")
    return errors


def task_dependency_errors() -> list[str]:
    """Verify import-time task dependencies in the interpreter doing preflight."""
    missing = [
        module
        for module in REQUIRED_V2_TASK_MODULES
        if importlib.util.find_spec(module) is None
    ]
    return [f"V2 task Python dependencies are missing: {missing}"] if missing else []


def split_set_digest(task_ids: list[str]) -> str:
    payload = "\n".join(sorted(task_ids)) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def _gated_task_paths(repo: Path) -> list[Path]:
    return sorted((repo / "evaluation_examples" / "task_class").glob("task_*.py"))


def _init_marker_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _has_user_sim_credentials(env=os.environ) -> bool:
    key_env = env.get("OSWORLD_USER_SIM_API_KEY_ENV")
    if key_env:
        return bool(env.get(key_env))
    return bool(env.get("OSWORLD_USER_SIM_API_KEY") or env.get("OPENAI_API_KEY"))


def _check_proxy(path: Path) -> str | None:
    try:
        entries = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"proxy config is unreadable/invalid: {path}: {exc}"
    if not isinstance(entries, list) or not entries:
        return f"proxy config must contain a non-empty list: {path}"
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not entry.get("host")
            or not entry.get("port")
        ):
            return f"proxy entry lacks host/port: {path}"
        serialized = json.dumps(entry).lower()
        if "your_username" in serialized or "your_password" in serialized:
            return f"proxy config still contains placeholder credentials: {path}"
    return None


def proxy_dependent_task_ids(root: Path) -> dict[str, list[str]]:
    """Map each canonical split to the task ids that declare ``proxy = True``.

    Used only to make the cost of --allow-placeholder-proxy explicit. Reads the
    task sources textually rather than importing them: importing a V2 task
    module runs its top-level env-var checks, which is exactly what preflight
    exists to report on rather than trigger.
    """
    pattern = re.compile(r"^\s*proxy\s*=\s*True\s*$", re.MULTILINE)
    task_dir = root / "OSWorldV2_Base/evaluation_examples/task_class"
    needs = set()
    for path in sorted(task_dir.glob("task_*.py")):
        try:
            if pattern.search(path.read_text()):
                needs.add(path.stem[len("task_") :])
        except OSError:
            continue

    out: dict[str, list[str]] = {}
    for split in ("osworld_v2_train_50", "osworld_v2_val_25", "osworld_v2_test"):
        try:
            ids = set(json.loads((root / "eval_sets" / f"{split}.json").read_text()))
        except (OSError, json.JSONDecodeError):
            continue
        out[split] = sorted(ids & needs)
    return out


def selected_tasks_require_proxy(
    root: Path,
    selected_task_ids: set[str] | None,
) -> bool:
    """Whether this invocation includes a task that declares ``proxy=True``.

    ``None`` means the caller did not provide an exact task subset, so the
    conservative answer is True.  This lets a formal rerun of only non-proxy
    tasks proceed without the unsafe placeholder waiver while preserving the
    existing fail-closed behavior for split/full-suite runs.
    """
    if selected_task_ids is None:
        return True
    proxy_ids = {
        task_id for ids in proxy_dependent_task_ids(root).values() for task_id in ids
    }
    return bool(selected_task_ids & proxy_ids)


def validate_workspace(
    *,
    root: Path,
    release_name: str,
    assets_dir: Path,
    website_suffix: str,
    proxy_config: Path,
    release_pin: dict | None = None,
    allow_placeholder_proxy: bool = False,
    selected_task_ids: set[str] | None = None,
    required_harnesses: tuple[str, ...] = ("prime_agent",),
) -> list[str]:
    errors: list[str] = []
    base = root / "OSWorldV2_Base"
    evol = root / "OSWorldV2_Evol"
    if release_pin is None:
        pin_path = root / PIN_RELATIVE_PATH
        try:
            release_pin = load_release_pin(root)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return [
                f"parent-owned release pin is unreadable/invalid: {pin_path}: {exc}"
            ]
    if release_name != release_pin.get("release"):
        return [
            f"release {release_name!r} is not parent-pinned; expected "
            f"{release_pin.get('release')!r}"
        ]

    release_path = base / "benchmark_releases" / f"{release_name}.json"
    try:
        release = json.loads(release_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"release manifest is unreadable/invalid: {release_path}: {exc}"]

    expected_manifest_sha = release_pin.get("release_manifest_sha256")
    if expected_manifest_sha and _sha256(release_path) != expected_manifest_sha:
        errors.append(f"release manifest hash differs from parent pin: {release_path}")
    expected_tag = release_pin.get("code_tag")
    expected_commit = release_pin.get("code_commit")
    expected_repository = release_pin.get("code_repository")
    if release.get("release") != release_name or release.get("status") != "active":
        errors.append(
            "clone release manifest identity/status does not match the active parent pin"
        )
    code = release.get("osworld_code", {})
    if code.get("tag") != expected_tag or code.get("repository") != expected_repository:
        errors.append(
            "clone release manifest code repository/tag differs from parent pin"
        )

    try:
        if base.resolve() == evol.resolve():
            errors.append("Base and Evol resolve to the same checkout")
    except OSError as exc:
        errors.append(f"could not resolve Base/Evol checkout paths: {exc}")
    base_git_dir = (
        _git_output(base, "rev-parse", "--absolute-git-dir") if base.is_dir() else None
    )
    evol_git_dir = (
        _git_output(evol, "rev-parse", "--absolute-git-dir") if evol.is_dir() else None
    )
    if (
        base_git_dir
        and evol_git_dir
        and Path(base_git_dir).resolve() == Path(evol_git_dir).resolve()
    ):
        errors.append("Base and Evol share the same git directory")

    for label, repo in (("Base", base), ("Evol", evol)):
        actual = _exact_tag(repo) if repo.is_dir() else None
        if actual != expected_tag:
            errors.append(
                f"{label} checkout tag is {actual!r}, expected {expected_tag!r}"
            )
        actual_commit = (
            _git_output(repo, "rev-parse", "HEAD") if repo.is_dir() else None
        )
        if actual_commit != expected_commit:
            errors.append(
                f"{label} checkout commit is {actual_commit!r}, expected {expected_commit!r}"
            )
        origin = (
            _git_output(repo, "remote", "get-url", "origin") if repo.is_dir() else None
        )
        if _repository_slug(origin) != expected_repository:
            errors.append(
                f"{label} origin is {_repository_slug(origin)!r}, expected "
                f"{expected_repository!r}"
            )
        if actual == expected_tag and not _worktree_clean(repo):
            errors.append(
                f"{label} checkout has tracked or untracked changes at {expected_tag}; "
                "release code provenance is not exact"
            )

    # Gated task definitions are evaluator-only gold logic. They must never be
    # present in the Evol template or an existing island clone, which the
    # builder can read and which initialize_clone_v2 rsyncs wholesale.
    builder_checkouts = [evol]
    island_checkouts = [path for path in root.glob("OSWorldV2_Evol_*") if path.is_dir()]
    builder_checkouts.extend(island_checkouts)
    for checkout in builder_checkouts:
        leaked = _gated_task_paths(checkout)
        if leaked:
            errors.append(
                f"Builder-visible checkout {checkout} contains {len(leaked)} gated "
                "task definition(s); remove them to restore the evaluation boundary"
            )
    expected_island_provenance = {
        "benchmark": "osworld_v2",
        "source_template_commit": str(expected_commit),
        "source_template_tag": str(expected_tag),
        "release": str(release_pin.get("release")),
    }
    try:
        expected_island_provenance["release_pin_sha256"] = _sha256(
            root / PIN_RELATIVE_PATH
        )
    except OSError:
        # ``release_pin`` may be injected by an isolated unit fixture. The
        # production entry point below always requires the parent pin file.
        pass
    for checkout in island_checkouts:
        marker = checkout / ".shadow_evol_init"
        try:
            values = _init_marker_values(marker)
        except OSError as exc:
            errors.append(
                f"Builder island {checkout} lacks readable release provenance: {exc}"
            )
            continue
        mismatched = {
            key: (values.get(key), expected)
            for key, expected in expected_island_provenance.items()
            if values.get(key) != expected
        }
        if mismatched:
            errors.append(
                f"Builder island {checkout} has stale release provenance: {mismatched}"
            )

    expected_suffix = release_pin.get("website_suffix")
    if website_suffix != expected_suffix:
        # The pin binds release, code tag, task hashes and website deployment
        # together so a mismatched website deployment cannot quietly produce a
        # reportable number. That guard stays on by default.
        #
        # It is overridable only by naming the exact substitute suffix, because
        # the pinned deployment can genuinely go away: on 2026-09-20 two of the
        # four vhosts on site.hku.icu (wandb, gitlab) began answering TLS with
        # alert 80, which failed task 072 in setup three times at 0s each and
        # recorded an imputed zero. That host is not ours to repair.
        #
        # The override does NOT edit the pin. Editing it would assert the
        # release is still intact while silently serving different websites,
        # which is the failure this check exists to prevent. Naming the
        # substitute leaves the deviation visible in the environment, in this
        # warning, and in the run config's website_suffix (which the launcher's
        # drift guard then refuses to accept silently).
        allowed = os.environ.get("SHADOW_EVOL_WEBSITE_SUFFIX_OVERRIDE", "")
        if allowed and allowed == website_suffix:
            print(
                "WARNING: website deployment does not match the release pin.\n"
                f"         pinned:   {expected_suffix!r} (for {expected_tag})\n"
                f"         serving:  {website_suffix!r}\n"
                "         Results are comparable only to other runs on the SAME\n"
                "         substitute deployment. Variants measured against the\n"
                "         pinned host and against this one are not directly\n"
                "         comparable on the 40 website-backed tasks.",
                file=sys.stderr,
            )
        else:
            errors.append(
                f"WEBSITE_HOST_SUFFIX={website_suffix!r}, expected {expected_suffix!r} "
                f"for {expected_tag}. To evaluate against a substitute deployment "
                f"set SHADOW_EVOL_WEBSITE_SUFFIX_OVERRIDE to that exact suffix; "
                f"do not edit the release pin."
            )

    task_paths = _gated_task_paths(base)
    expected_count = int(release_pin.get("task_count", 0))
    release_task_count = int(release.get("task_hash_manifest", {}).get("task_count", 0))
    if release_task_count != expected_count:
        errors.append(
            f"clone release manifest task_count={release_task_count}, parent pin has "
            f"{expected_count}"
        )
    if len(task_paths) != expected_count:
        errors.append(f"found {len(task_paths)}/{expected_count} V2 task classes")
    expected_digest = release_pin.get("task_set_sha256")
    if expected_digest and len(task_paths) == expected_count:
        actual_digest = task_set_digest(task_paths)
        if actual_digest != expected_digest:
            errors.append(
                f"task set digest mismatch for {expected_tag}: {actual_digest}"
            )
    errors.extend(task_source_errors(task_paths))
    errors.extend(task_dependency_errors())

    task_manifest_path = base / "evaluation_examples" / "test_v2.json"
    expected_task_manifest_sha = release_pin.get("task_manifest_sha256")
    try:
        task_manifest = json.loads(task_manifest_path.read_text())
        manifest_ids = (
            task_manifest.get("tasks") if isinstance(task_manifest, dict) else None
        )
        if not isinstance(manifest_ids, list) or len(manifest_ids) != len(
            set(manifest_ids)
        ):
            raise ValueError("expected a unique {'tasks': [...]} manifest")
        if (
            expected_task_manifest_sha
            and _sha256(task_manifest_path) != expected_task_manifest_sha
        ):
            errors.append("V2 test_v2.json hash differs from parent release pin")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        manifest_ids = None
        errors.append(f"invalid V2 task manifest {task_manifest_path}: {exc}")

    split_names = (
        "osworld_v2_train_50.json",
        "osworld_v2_val_25.json",
        "osworld_v2_test.json",
    )
    split_sets = []
    expected_split_sizes = tuple(release_pin.get("split_sizes", ()))
    if len(expected_split_sizes) != len(split_names):
        errors.append("parent release pin has invalid split_sizes")
        expected_split_sizes = None
    for split_index, name in enumerate(split_names):
        path = root / "eval_sets" / name
        try:
            values = json.loads(path.read_text())
            if not isinstance(values, list) or len(values) != len(set(values)):
                raise ValueError("must be a unique JSON list")
            if (
                expected_split_sizes is not None
                and len(values) != expected_split_sizes[split_index]
            ):
                errors.append(
                    f"split {name} has {len(values)} tasks, expected "
                    f"{expected_split_sizes[split_index]}"
                )
            expected_split_digest = release_pin.get("split_sha256", {}).get(name)
            if (
                expected_split_digest is not None
                and split_set_digest(values) != expected_split_digest
            ):
                errors.append(
                    f"split {name} does not match the canonical {expected_tag} ID set"
                )
            split_sets.append(set(values))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"invalid split {path}: {exc}")
    if len(split_sets) == 3:
        if any(
            split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)
        ):
            errors.append("V2 train/val/test splits overlap")
        split_union = set().union(*split_sets)
        task_ids = {path.stem.removeprefix("task_") for path in task_paths}
        if split_union != task_ids:
            missing = sorted(task_ids - split_union)
            unknown = sorted(split_union - task_ids)
            errors.append(
                "V2 train/val/test IDs do not exactly match task classes: "
                f"missing={missing}, unknown={unknown}"
            )
        if manifest_ids is not None and set(manifest_ids) != task_ids:
            errors.append("V2 test_v2.json IDs do not exactly match task classes")

    asset_count, asset_bytes = _dir_stats(assets_dir)
    min_asset_files = int(release_pin.get("asset_min_files", 1))
    min_asset_bytes = int(release_pin.get("asset_min_bytes", 4 * 1024**3))
    if asset_count < min_asset_files or asset_bytes < min_asset_bytes:
        errors.append(
            f"asset cache is incomplete: {assets_dir} has {asset_count} files / "
            f"{asset_bytes} bytes (expected at least {min_asset_files} files / "
            f"{min_asset_bytes} bytes for the parent-pinned snapshot)"
        )
    expected_asset_count = release_pin.get("asset_file_count")
    expected_asset_bytes = release_pin.get("asset_total_bytes")
    if expected_asset_count is not None and asset_count != int(expected_asset_count):
        errors.append(
            f"asset cache has {asset_count} files, expected exactly {expected_asset_count}"
        )
    if expected_asset_bytes is not None and asset_bytes != int(expected_asset_bytes):
        errors.append(
            f"asset cache has {asset_bytes} bytes, expected exactly {expected_asset_bytes}"
        )
    expected_asset_revision = release_pin.get("asset_commit")
    if expected_asset_revision:
        errors.extend(asset_snapshot_errors(assets_dir, str(expected_asset_revision)))

    if "prime_agent" in required_harnesses:
        from agent_s4.harness.prime.source import (
            validate_prime_agent_runtime_cache,
        )

        try:
            validate_prime_agent_runtime_cache(
                root / "vm_assets/prime_agent_runtime.tar.gz"
            )
        except RuntimeError as exc:
            errors.append(str(exc))

    if "opencode" in required_harnesses:
        # The pinned evaluator installs /opt/shadow-opencode/<version>/opencode
        # from this archive; the VMs never download it.
        from agent_s4.harness.opencode.source import validate_runtime_cache

        try:
            validate_runtime_cache(root / "vm_assets/opencode_runtime.tar.gz")
        except RuntimeError as exc:
            errors.append(str(exc))

    proxy_required = selected_tasks_require_proxy(root, selected_task_ids)
    proxy_error = _check_proxy(proxy_config) if proxy_required else None
    if proxy_error:
        # allow_placeholder_proxy downgrades this to a warning. The tasks that
        # declare proxy = True do not become skipped -- they still run and
        # still fail, so their zeros enter the fitness signal as though the
        # interface could not do them. Never use it for a reported number.
        if allow_placeholder_proxy:
            affected = proxy_dependent_task_ids(root)
            print(
                "WARNING: proxy check waived by --allow-placeholder-proxy. "
                f"{proxy_error}"
            )
            for split, ids in affected.items():
                if ids:
                    print(
                        f"WARNING:   {split}: {len(ids)} task(s) will fail "
                        f"for lack of an egress proxy: {', '.join(ids)}"
                    )
            print(
                "WARNING: those failures are indistinguishable from genuine "
                "interface failures, so they bias every selection step. This "
                "run is a diagnostic, not a reportable result."
            )
        else:
            errors.append(proxy_error)
    for name in ("GITLAB_URL", "GITLAB_PRIVATE_TOKEN"):
        if not os.environ.get(name):
            errors.append(f"{name} is required by the selected native train/val tasks")
    if not _has_user_sim_credentials():
        errors.append(
            "LLM HITL credentials are missing: set OSWORLD_USER_SIM_API_KEY, "
            "or OSWORLD_USER_SIM_API_KEY_ENV naming a populated variable"
        )

    grader_provider = os.environ.get("OSWORLD_EVAL_MODEL_PROVIDER")
    grader_model = os.environ.get("OSWORLD_EVAL_MODEL_NAME")
    grader_key_env = os.environ.get("OSWORLD_EVAL_MODEL_API_KEY_ENV")
    if not grader_provider or not grader_model:
        errors.append(
            "OSWORLD_EVAL_MODEL_PROVIDER and OSWORLD_EVAL_MODEL_NAME must "
            "explicitly configure V2 model-based graders"
        )
    elif grader_provider != "bedrock" and not (
        os.environ.get("OSWORLD_EVAL_MODEL_API_KEY")
        or (grader_key_env and os.environ.get(grader_key_env))
    ):
        errors.append(
            "V2 grader credentials are missing: set OSWORLD_EVAL_MODEL_API_KEY "
            "or OSWORLD_EVAL_MODEL_API_KEY_ENV naming a populated variable"
        )

    expected_ami = (
        release.get("provider_images", {})
        .get("aws", {})
        .get("ubuntu", {})
        .get("us-east-1", {})
        .get("1920x1080", {})
        .get("ami_id")
    )
    pinned_ami = release_pin.get("aws_ami", {}).get("us-east-1", {}).get("1920x1080")
    if expected_ami != pinned_ami:
        errors.append(
            f"release AMI {expected_ami!r} differs from parent pin {pinned_ami!r}"
        )
    manager_path = base / "desktop_env" / "providers" / "aws" / "manager.py"
    try:
        manager_text = manager_path.read_text()
    except OSError as exc:
        errors.append(f"V2 AWS manager is unreadable: {manager_path}: {exc}")
    else:
        if pinned_ami and pinned_ami not in manager_text:
            errors.append(f"release AMI {pinned_ami} is not present in V2 AWS manager")
    return errors


def builder_sandbox_errors() -> list[str]:
    """Can a Builder actually be confined, or only asked nicely to behave?

    Without Landlock the V2 Builder runs unsandboxed next to the gated
    task_class evaluators, the parent DB and the eval cache. Both the CLI
    preflight and the orchestrator gate check this, so --preflight-only cannot
    report a clean workspace that the run itself would then refuse.
    """
    try:
        from agent_s4.builder_sandbox import landlock_abi
    except (ImportError, OSError) as exc:
        return [f"could not verify Builder Landlock support: {exc}"]
    try:
        if landlock_abi() == 0:
            return [
                "Landlock is unavailable; formal V2 Builders cannot be isolated "
                "from gated tasks/DB/eval cache"
            ]
    except OSError as exc:
        return [f"could not verify Builder Landlock support: {exc}"]
    return []


def validate_default_workspace(
    root: Path = PROJECT_ROOT,
    *,
    required_harnesses: tuple[str, ...] = ("prime_agent",),
    allow_placeholder_proxy: bool | None = None,
    selected_task_ids: set[str] | None = None,
) -> list[str]:
    """Run the complete local-only gate used by every formal V2 entry point.

    selected_task_ids narrows the proxy check to the tasks this run will
    execute; None keeps the conservative every-task answer."""
    try:
        pin = load_release_pin(root)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [
            f"parent-owned release pin is unreadable/invalid: "
            f"{root / PIN_RELATIVE_PATH}: {exc}"
        ]
    errors = parent_pin_git_errors(root)
    errors.extend(builder_sandbox_errors())
    assets = Path(
        os.environ.get(
            "OSWORLD_FILE_BASE_URL",
            str(root / "OSWorldV2_Base/cache/osworld_v2_assets"),
        )
    )
    proxy = Path(
        os.environ.get(
            "PROXY_CONFIG_FILE",
            str(
                root
                / "OSWorldV2_Base/evaluation_examples/settings/proxy/dataimpulse.json"
            ),
        )
    )
    errors.extend(
        validate_workspace(
            root=root,
            release_name=str(pin.get("release")),
            assets_dir=assets.resolve(),
            website_suffix=os.environ.get("WEBSITE_HOST_SUFFIX", ""),
            proxy_config=proxy.resolve(),
            release_pin=pin,
            allow_placeholder_proxy=(
                os.environ.get("SHADOW_EVOL_ALLOW_PLACEHOLDER_PROXY") == "1"
                if allow_placeholder_proxy is None
                else allow_placeholder_proxy
            ),
            selected_task_ids=selected_task_ids,
            required_harnesses=required_harnesses,
        )
    )
    return errors


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--assets-dir", type=Path, default=None)
    parser.add_argument(
        "--harness",
        action="append",
        choices=("prime_agent", "opencode"),
        help="Harness used by this run (default: prime_agent).",
    )
    parser.add_argument(
        "--website-suffix", default=os.environ.get("WEBSITE_HOST_SUFFIX", "")
    )
    parser.add_argument("--proxy-config", type=Path, default=None)
    parser.add_argument(
        "--task-id",
        action="append",
        default=None,
        help="Exact selected task ID. Repeat for task-aware proxy validation. "
        "When omitted, preflight conservatively requires a real proxy.",
    )
    parser.add_argument(
        "--allow-placeholder-proxy",
        action="store_true",
        default=os.environ.get("SHADOW_EVOL_ALLOW_PLACEHOLDER_PROXY") == "1",
        help="Unsafe diagnostic escape hatch: proceed with an unconfigured "
        "egress proxy. The proxy-dependent tasks still run and still "
        "fail, so their zeros contaminate the fitness signal. Formal "
        "runs must leave this off.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    errors = parent_pin_git_errors(root)
    errors.extend(builder_sandbox_errors())
    errors.extend(
        validate_workspace(
            root=root,
            release_name=args.release,
            assets_dir=(
                args.assets_dir or root / "OSWorldV2_Base/cache/osworld_v2_assets"
            ).resolve(),
            website_suffix=args.website_suffix,
            proxy_config=(
                args.proxy_config
                or root
                / "OSWorldV2_Base/evaluation_examples/settings/proxy/dataimpulse.json"
            ).resolve(),
            allow_placeholder_proxy=args.allow_placeholder_proxy,
            selected_task_ids=set(args.task_id) if args.task_id is not None else None,
            required_harnesses=tuple(args.harness or ("prime_agent",)),
        )
    )
    if errors:
        print("OSWorld V2 preflight FAILED (no API/VM was contacted):")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("OSWorld V2 preflight passed (local verification only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
