"""Prepare the pinned Linux x64 OpenCode locally, without provisioning a VM.

The official release digest is pinned in source. Produces both a host binary
and the cache/layout expected by the new evaluator; no system installation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import urllib.request

# agent_s4 imports from the Agent-S repository root, installed or not.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from agent_s4.harness.opencode.source import (
    OPENCODE_VERSION, RELEASE_ARCHIVE_SHA256, RELEASE_URL, VM_BINARY, assets_dir,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=assets_dir())
    parser.add_argument("--release-archive", type=Path, help="Use an already downloaded official archive")
    args = parser.parse_args()
    data = (args.release_archive.read_bytes() if args.release_archive
            else urllib.request.urlopen(RELEASE_URL, timeout=120).read())
    if hashlib.sha256(data).hexdigest() != RELEASE_ARCHIVE_SHA256:
        raise SystemExit("Official OpenCode archive SHA256 mismatch")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.removeprefix("./") == "opencode"]
        if len(members) != 1:
            raise SystemExit("Unexpected OpenCode release archive layout")
        binary_data = archive.extractfile(members[0]).read()
    root = args.assets_dir.resolve()
    binary = root / "opencode" / OPENCODE_VERSION / "opencode"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(binary_data)
    binary.chmod(0o755)
    cache = root / "opencode_runtime.tar.gz"
    temporary = cache.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        info = tarfile.TarInfo(VM_BINARY.lstrip("/"))
        info.size, info.mode = len(binary_data), 0o755
        archive.addfile(info, io.BytesIO(binary_data))
    with temporary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    temporary.replace(cache)
    cache.with_suffix(cache.suffix + ".metadata.json").write_text(json.dumps({
        "opencode_version": OPENCODE_VERSION, "sha256": digest, "binary": VM_BINARY,
        "release_archive_sha256": RELEASE_ARCHIVE_SHA256,
    }, indent=2) + "\n")
    print(f"Host binary: {binary}\nVM cache: {cache}")


if __name__ == "__main__":
    main()
