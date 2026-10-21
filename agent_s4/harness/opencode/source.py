"""Pin and verify OpenCode separately from the running Prime installation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from agent_s4.paths import SHARED_ROOT

OPENCODE_VERSION = "1.18.34"
RELEASE_ARCHIVE_SHA256 = (
    "0f22479647226d1d2dd99595d20082ee7bda3870b62dc6a90b41efc1a71d7e9a"
)
RELEASE_URL = f"https://github.com/anomalyco/opencode/releases/download/v{OPENCODE_VERSION}/opencode-linux-x64.tar.gz"
VM_BINARY = f"/opt/shadow-opencode/{OPENCODE_VERSION}/opencode"


def assets_dir() -> Path:
    return Path(os.environ.get("SHADOW_VM_ASSETS_DIR") or SHARED_ROOT / "vm_assets")


def resolve_binary() -> Path:
    explicit = os.environ.get("SHADOW_OPENCODE_BIN")
    local = assets_dir() / "opencode" / OPENCODE_VERSION / "opencode"
    name = explicit or (str(local) if local.is_file() else shutil.which("opencode"))
    if not name:
        raise RuntimeError(
            "Run scripts/assets/prepare_opencode_runtime.py or set SHADOW_OPENCODE_BIN"
        )
    binary = Path(name).expanduser().resolve()
    result = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, timeout=30
    )
    if result.returncode or result.stdout.strip() != OPENCODE_VERSION:
        raise RuntimeError(
            f"OpenCode must be pinned to {OPENCODE_VERSION}; got {result.stdout.strip()!r}"
        )
    return binary


def validate_runtime_cache(path: Path) -> dict:
    try:
        metadata = json.loads(
            path.with_suffix(path.suffix + ".metadata.json").read_text()
        )
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "OpenCode runtime cache missing or invalid; run scripts/assets/prepare_opencode_runtime.py"
        ) from exc
    if (
        metadata.get("opencode_version") != OPENCODE_VERSION
        or metadata.get("sha256") != digest
        or metadata.get("binary") != VM_BINARY
    ):
        raise RuntimeError("OpenCode cache version, layout or SHA256 mismatch")
    return metadata
