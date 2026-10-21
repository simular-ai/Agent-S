"""Offline provenance contract for the Prime Agent evaluator runtime."""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


from agent_s4.paths import SHARED_ROOT

VENDORED_RUNTIME = SHARED_ROOT / "agent_harness" / "prime-agent-runtime"
VENDORED_RUNTIME_PATH = VENDORED_RUNTIME
KERNEL_ENVIRONMENT = SHARED_ROOT / "agent_harness" / "prime-agent-kernel"
UPSTREAM_COMMIT_MARKER = VENDORED_RUNTIME / "UPSTREAM_COMMIT"
SOURCE_SENTINEL = VENDORED_RUNTIME / "src" / "rlm" / "harness.py"
SOURCE_MANIFEST = "SOURCE_MANIFEST.json"
PINNED_PRIME_AGENT_COMMIT = "a9b5d88b50db50860088772a408e867f8071b507"
RUNTIME_COMMIT_MARKER = "prime_agent_runtime.commit"
RUNTIME_SHA256_MARKER = "prime_agent_runtime.sha256"
RUNTIME_SCHEMA_VERSION = "3"
NODE_VERSION = "22.23.2"
NODE_ARTIFACT_URL = (
    f"https://nodejs.org/dist/v{NODE_VERSION}/" f"node-v{NODE_VERSION}-linux-x64.tar.xz"
)
NODE_ARTIFACT_SHA256 = (
    "d60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307"
)
PYTHON_VERSION = "3.11.16"
RUNTIME_TARGET = f"x86_64-linux-node{NODE_VERSION}-python{PYTHON_VERSION}"
PRIME_AGENT_VERSION = "0.8.0-beta.546.1.a9b5d88"
PRIME_AGENT_ARTIFACT_URL = (
    "https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev/releases/"
    "v0.8.0-beta.546.1.a9b5d88/prime-agent-0.8.0-beta.546.1.a9b5d88.tgz"
)
PRIME_AGENT_ARTIFACT_SHA256 = (
    "830d2b10656a7e04bece4b0d895b66fa0aa86e04bfc2088d03b00c27ef015831"
)
KERNEL_REQUIRED_IMPORTS = (
    "requests",
    "httpx",
    "yaml",
    "tomli",
    "dotenv",
    "pandas",
    "numpy",
    "scipy",
    "bs4",
    "lxml",
    "pydantic",
    "tyro",
    "dill",
    "PIL",
)
KERNEL_SKILL_IMPORTS = (
    "agent_message",
    "agent_observe",
    "attach_image",
    "compact",
    "edit",
    "goal",
    "linear",
    "notion",
    "refine",
    "rlm_heartbeat",
    "websearch",
)
REQUIRED_RUNTIME_FILES = frozenset(
    {
        "LICENSE",
        "pyproject.toml",
        "uv.lock",
        "src/rlm/__init__.py",
        "src/rlm/harness.py",
        "src/rlm/mcp.py",
        "src/rlm/mcp_base.py",
        "src/rlm/skill.py",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IMPORTABLE_SUFFIXES = frozenset(
    {".py", ".pyc", ".so", ".pyd", ".dll", ".dylib", ".pth"}
)


def expected_prime_agent_commit() -> str:
    try:
        value = UPSTREAM_COMMIT_MARKER.read_text().strip()
    except OSError as exc:
        raise RuntimeError(
            f"Prime Agent upstream pin is missing at {UPSTREAM_COMMIT_MARKER}"
        ) from exc
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value.lower()):
        raise RuntimeError(f"Prime Agent upstream pin is not a full git SHA: {value!r}")
    value = value.lower()
    if value != PINNED_PRIME_AGENT_COMMIT:
        raise RuntimeError(
            "Vendored Prime Agent commit does not match the integration pin: "
            f"expected {PINNED_PRIME_AGENT_COMMIT}, found {value}."
        )
    return value


def validate_prime_agent_source() -> str:
    """Validate the codebase-owned harness copy and return its upstream pin."""
    commit = validate_prime_agent_runtime_source()
    required = (
        KERNEL_ENVIRONMENT / "pyproject.toml",
        KERNEL_ENVIRONMENT / "uv.lock",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Prime Agent kernel dependency lock is incomplete: " + ", ".join(missing)
        )
    return commit


def _load_source_manifest(runtime_path: Path) -> dict[str, Any]:
    manifest_path = runtime_path / SOURCE_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(
            f"Prime Agent source manifest is missing at {manifest_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Prime Agent source manifest is invalid at {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise RuntimeError("Prime Agent source manifest must use schema 1")
    return manifest


def validate_prime_agent_runtime_source(
    runtime_path: str | Path = VENDORED_RUNTIME,
) -> str:
    """Require every vendored runtime file to match upstream."""
    runtime_path = Path(runtime_path).resolve()
    sentinel = runtime_path / "src" / "rlm" / "harness.py"
    if not sentinel.is_file():
        raise RuntimeError(f"vendored Prime Agent harness is missing at {sentinel}")
    marker_path = runtime_path / "UPSTREAM_COMMIT"
    try:
        marker = marker_path.read_text(encoding="utf-8").strip().lower()
    except OSError as exc:
        raise RuntimeError(
            f"Prime Agent upstream pin is missing at {marker_path}"
        ) from exc
    if marker != PINNED_PRIME_AGENT_COMMIT:
        raise RuntimeError(
            "Vendored Prime Agent commit does not match the integration pin: "
            f"expected {PINNED_PRIME_AGENT_COMMIT}, found {marker or '<empty>'}."
        )

    manifest = _load_source_manifest(runtime_path)
    if manifest.get("upstream_commit") != PINNED_PRIME_AGENT_COMMIT:
        raise RuntimeError("Prime Agent source manifest has the wrong upstream commit")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != REQUIRED_RUNTIME_FILES:
        names = set(files) if isinstance(files, dict) else set()
        raise RuntimeError(
            "Prime Agent source manifest has the wrong required file set: "
            f"missing={sorted(REQUIRED_RUNTIME_FILES - names)}, "
            f"extra={sorted(names - REQUIRED_RUNTIME_FILES)}"
        )
    for raw_name, expected_digest in files.items():
        name = PurePosixPath(raw_name)
        if name.is_absolute() or ".." in name.parts:
            raise RuntimeError(f"unsafe Prime Agent source manifest path: {raw_name!r}")
        if not isinstance(expected_digest, str) or not _SHA256_RE.fullmatch(
            expected_digest
        ):
            raise RuntimeError(
                f"invalid SHA-256 for Prime Agent source file {raw_name!r}"
            )
        path = runtime_path.joinpath(*name.parts)
        try:
            actual_digest = file_sha256(path)
        except OSError as exc:
            raise RuntimeError(
                f"vendored Prime Agent source file is missing: {path}"
            ) from exc
        if actual_digest != expected_digest:
            raise RuntimeError(
                "Vendored Prime Agent source file failed SHA-256 verification: "
                f"{raw_name}; expected {expected_digest}, found {actual_digest}."
            )

    unexpected_importable = []
    for path in (runtime_path / "src").rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _IMPORTABLE_SUFFIXES:
            continue
        relative = path.relative_to(runtime_path).as_posix()
        if path.suffix.lower() == ".pyc" and "__pycache__" in path.parts:
            continue
        if relative not in REQUIRED_RUNTIME_FILES:
            unexpected_importable.append(relative)
    if unexpected_importable:
        raise RuntimeError(
            "Vendored Prime Agent runtime contains unexpected importable artifacts: "
            + ", ".join(sorted(unexpected_importable)[:5])
        )
    return marker


def runtime_cache_fingerprint(commit: str | None = None) -> str:
    commit = commit or validate_prime_agent_source()
    digest = hashlib.sha256()
    for path in sorted(VENDORED_RUNTIME.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(path.relative_to(VENDORED_RUNTIME).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    for path in sorted(KERNEL_ENVIRONMENT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(b"kernel\0")
            digest.update(path.relative_to(KERNEL_ENVIRONMENT).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return (
        f"v{RUNTIME_SCHEMA_VERSION}:{RUNTIME_TARGET}:{commit}:"
        f"{PRIME_AGENT_ARTIFACT_SHA256}:{NODE_ARTIFACT_SHA256}:"
        f"{digest.hexdigest()}"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# The VM extracts the runtime with `sudo tar`, which keeps the recorded owner,
# then patches files in it as the unprivileged task user. Anything not owned by
# that user (uid 1000) is read-only to it, and every task fails at agent install.
RUNTIME_ARCHIVE_UID = 1000
RUNTIME_ARCHIVE_USER = "user"


def validate_runtime_archive_layout(path: Path) -> None:
    root = PurePosixPath("opt/prime-agent")
    required = {
        PurePosixPath("opt/prime-agent/bin/prime-agent"),
        PurePosixPath("opt/prime-agent/python/bin/python3"),
        PurePosixPath("opt/prime-agent/upstream_commit"),
        PurePosixPath("opt/prime-agent/runtime_fingerprint"),
    }
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError(f"Prime Agent runtime archive is invalid: {exc}") from exc
    paths = {PurePosixPath(member.name.lstrip("./")) for member in members}
    missing = sorted(map(str, required - paths))
    if missing:
        raise RuntimeError(
            f"Prime Agent runtime archive lacks required files: {missing}"
        )
    for member in members:
        member_path = PurePosixPath(member.name.lstrip("./"))
        if (
            member_path.is_absolute()
            or ".." in member_path.parts
            or member_path != root
            and root not in member_path.parents
        ):
            raise RuntimeError(
                f"unsafe Prime Agent runtime archive member: {member.name!r}"
            )
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            raise RuntimeError(
                f"unsupported Prime Agent runtime archive member: {member.name!r}"
            )
        if member.uid != RUNTIME_ARCHIVE_UID:
            raise RuntimeError(
                f"Prime Agent runtime archive member {member.name!r} is owned by "
                f"uid {member.uid}, not the VM task user's uid "
                f"{RUNTIME_ARCHIVE_UID}; rebuild it with "
                "scripts/assets/cache_prime_agent_assets.py"
            )
        if member.issym() or member.islnk():
            target = PurePosixPath(member.linkname)
            if target.is_absolute():
                raise RuntimeError(
                    f"unsafe Prime Agent runtime link target: {member.linkname!r}"
                )
            # Symlinks are relative to their containing directory; tar hard
            # links are archive-root relative. Normalize either form and
            # require the result to remain under /opt/prime-agent. This keeps
            # npm's ordinary ../package links while rejecting traversal.
            combined = member_path.parent / target if member.issym() else target
            normalized_parts: list[str] = []
            for part in combined.parts:
                if part in {"", "."}:
                    continue
                if part == "..":
                    if not normalized_parts:
                        raise RuntimeError(
                            "unsafe Prime Agent runtime link target: "
                            f"{member.linkname!r}"
                        )
                    normalized_parts.pop()
                else:
                    normalized_parts.append(part)
            normalized = PurePosixPath(*normalized_parts)
            if normalized != root and root not in normalized.parents:
                raise RuntimeError(
                    f"unsafe Prime Agent runtime link target: {member.linkname!r}"
                )


def validate_prime_agent_runtime_cache(cache: Path) -> str:
    expected_commit = validate_prime_agent_source()
    if not cache.is_file():
        raise RuntimeError(
            f"Prime Agent VM runtime is missing at {cache}. Build it with "
            "`python3 scripts/assets/cache_prime_agent_assets.py`."
        )
    fingerprint = runtime_cache_fingerprint(expected_commit)
    marker = cache.parent / RUNTIME_COMMIT_MARKER
    sha_marker = cache.parent / RUNTIME_SHA256_MARKER
    try:
        recorded_fingerprint = marker.read_text().strip()
        recorded_sha = sha_marker.read_text().strip()
    except OSError as exc:
        raise RuntimeError(
            "Prime Agent runtime provenance markers are missing; rebuild the cache"
        ) from exc
    if recorded_fingerprint != fingerprint:
        raise RuntimeError(
            "Prime Agent VM runtime is stale: expected fingerprint "
            f"{fingerprint}, found {recorded_fingerprint or '<empty>'}"
        )
    actual_sha = file_sha256(cache)
    if recorded_sha != actual_sha:
        raise RuntimeError(
            "Prime Agent VM runtime archive failed SHA256 verification: "
            f"expected {recorded_sha or '<empty>'}, found {actual_sha}"
        )
    validate_runtime_archive_layout(cache)
    return expected_commit
