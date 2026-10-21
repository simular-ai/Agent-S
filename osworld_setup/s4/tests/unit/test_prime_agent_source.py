from __future__ import annotations

import io
import shutil
import tarfile
from pathlib import Path

import pytest

from agent_s4.harness.prime.source import (
    PINNED_PRIME_AGENT_COMMIT,
    VENDORED_RUNTIME_PATH,
    expected_prime_agent_commit,
    validate_prime_agent_source,
    validate_prime_agent_runtime_source,
    validate_runtime_archive_layout,
)


def _runtime_copy(tmp_path: Path) -> Path:
    target = tmp_path / "prime-agent-runtime"
    shutil.copytree(VENDORED_RUNTIME_PATH, target)
    return target


def test_vendored_prime_agent_runtime_matches_pin() -> None:
    assert expected_prime_agent_commit() == PINNED_PRIME_AGENT_COMMIT
    assert validate_prime_agent_source() == PINNED_PRIME_AGENT_COMMIT
    assert validate_prime_agent_runtime_source() == PINNED_PRIME_AGENT_COMMIT


def test_rejects_modified_vendored_source(tmp_path: Path) -> None:
    runtime = _runtime_copy(tmp_path)
    harness = runtime / "src" / "rlm" / "harness.py"
    harness.write_text(harness.read_text(encoding="utf-8") + "\n# modified\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed SHA-256 verification: src/rlm/harness.py"):
        validate_prime_agent_runtime_source(runtime)


def test_rejects_changed_commit_marker(tmp_path: Path) -> None:
    runtime = _runtime_copy(tmp_path)
    (runtime / "UPSTREAM_COMMIT").write_text("0" * 40 + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match the integration pin"):
        validate_prime_agent_runtime_source(runtime)


@pytest.mark.parametrize("name", ["override.py", "override.so", "sitecustomize.pth"])
def test_rejects_unexpected_importable_artifact(tmp_path: Path, name: str) -> None:
    runtime = _runtime_copy(tmp_path)
    (runtime / "src" / name).write_bytes(b"unexpected")

    with pytest.raises(RuntimeError, match="unexpected importable artifacts"):
        validate_prime_agent_runtime_source(runtime)


def _runtime_archive(tmp_path: Path, *, link_target: str, uid: int = 1000) -> Path:
    archive_path = tmp_path / "runtime.tar.gz"
    required = (
        "opt/prime-agent/bin/prime-agent",
        "opt/prime-agent/python/bin/python3",
        "opt/prime-agent/upstream_commit",
        "opt/prime-agent/runtime_fingerprint",
    )
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in required:
            info = tarfile.TarInfo(name)
            info.size = 1
            info.uid = uid
            archive.addfile(info, io.BytesIO(b"x"))
        link = tarfile.TarInfo("opt/prime-agent/node_modules/.bin/helper")
        link.type = tarfile.SYMTYPE
        link.linkname = link_target
        link.uid = uid
        archive.addfile(link)
    return archive_path


def test_runtime_archive_allows_internal_relative_link(tmp_path: Path) -> None:
    archive = _runtime_archive(
        tmp_path, link_target="../prime-agent/dist/bundle/cli.js",
    )
    validate_runtime_archive_layout(archive)


def test_runtime_archive_rejects_files_the_vm_user_cannot_patch(tmp_path: Path) -> None:
    """Built by another uid, `sudo tar` on the VM keeps that owner and the
    task user's install-time patches fail with PermissionError."""
    archive = _runtime_archive(
        tmp_path, link_target="../prime-agent/dist/bundle/cli.js", uid=1004,
    )
    with pytest.raises(RuntimeError, match="owned by uid 1004"):
        validate_runtime_archive_layout(archive)


def test_runtime_archive_rejects_escaping_link(tmp_path: Path) -> None:
    archive = _runtime_archive(tmp_path, link_target="../../../../../../etc/passwd")
    with pytest.raises(RuntimeError, match="unsafe Prime Agent runtime link"):
        validate_runtime_archive_layout(archive)
