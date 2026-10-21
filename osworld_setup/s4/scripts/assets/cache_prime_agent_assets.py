"""Build the pinned, offline Prime Agent runtime used by OSWorld VMs.

The build host may access npm, but evaluation VMs never do.  The resulting
archive contains Node, the exact Prime Agent npm package, and a Python venv
with this repository's vendored ``prime-agent-runtime`` harness.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
# The agent_s4 package lives at the Agent-S root, two levels up.
for _path in (REPO_ROOT, REPO_ROOT.parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from agent_s4.harness.prime.source import (
    KERNEL_ENVIRONMENT,
    KERNEL_REQUIRED_IMPORTS,
    KERNEL_SKILL_IMPORTS,
    NODE_ARTIFACT_SHA256,
    NODE_ARTIFACT_URL,
    NODE_VERSION,
    PRIME_AGENT_ARTIFACT_SHA256,
    PRIME_AGENT_ARTIFACT_URL,
    PRIME_AGENT_VERSION,
    PYTHON_VERSION,
    RUNTIME_ARCHIVE_UID,
    RUNTIME_ARCHIVE_USER,
    RUNTIME_COMMIT_MARKER,
    RUNTIME_SHA256_MARKER,
    VENDORED_RUNTIME,
    file_sha256,
    runtime_cache_fingerprint,
    validate_prime_agent_source,
    validate_runtime_archive_layout,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=REPO_ROOT / "vm_assets")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--artifact", type=Path, default=None,
        help="Use an already-downloaded official release artifact.",
    )
    parser.add_argument(
        "--node-artifact", type=Path, default=None,
        help="Use an already-downloaded official Node.js linux-x64 archive.",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Require npm and pip dependencies to be available in local caches.",
    )
    return parser.parse_args(argv)


def _run(argv, **kwargs):
    subprocess.run(argv, check=True, **kwargs)


def _vm_user_owned(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Record the VM task user as owner, whoever built the archive."""
    info.uid = info.gid = RUNTIME_ARCHIVE_UID
    info.uname = info.gname = RUNTIME_ARCHIVE_USER
    return info


def _download_or_validate(
    *, supplied: Path | None, url: str, expected_sha: str,
    destination: Path, offline: bool, label: str,
) -> Path:
    if supplied is None:
        if offline:
            raise RuntimeError(f"--offline requires a supplied {label} artifact")
        request = urllib.request.Request(
            url, headers={"User-Agent": "shadow-evol-prime-agent-cache/1"},
        )
        with urllib.request.urlopen(request) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
        artifact = destination
    else:
        artifact = supplied.resolve()
    actual_sha = file_sha256(artifact)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"official {label} artifact failed SHA256 verification: "
            f"expected {expected_sha}, found {actual_sha}"
        )
    return artifact


def _extract_node(artifact: Path, destination: Path) -> None:
    expected_root = f"node-v{NODE_VERSION}-linux-x64"
    with tarfile.open(artifact, "r:xz") as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            if (path.is_absolute() or ".." in path.parts
                    or not path.parts or path.parts[0] != expected_root):
                raise RuntimeError(
                    f"unsafe Node.js artifact member: {member.name!r}"
                )
        archive.extractall(destination.parent, filter="data")
    extracted = destination.parent / expected_root
    extracted.replace(destination)


def main(argv=None) -> int:
    args = parse_args(argv)
    commit = validate_prime_agent_source()
    fingerprint = runtime_cache_fingerprint(commit)
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    output = args.assets_dir / "prime_agent_runtime.tar.gz"
    marker = args.assets_dir / RUNTIME_COMMIT_MARKER
    sha_marker = args.assets_dir / RUNTIME_SHA256_MARKER
    if output.is_file() and not args.force:
        if (marker.is_file() and marker.read_text().strip() == fingerprint
                and sha_marker.is_file()
                and sha_marker.read_text().strip() == file_sha256(output)):
            validate_runtime_archive_layout(output)
            print(f"{output} already matches {fingerprint}")
            return 0

    if sys.platform != "linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise RuntimeError("Prime Agent VM runtime must be built on Linux x86_64")
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required to build the Prime Agent runtime")

    with tempfile.TemporaryDirectory(prefix="prime-agent-runtime-") as temporary:
        stage = Path(temporary)
        runtime = stage / "opt" / "prime-agent"
        (runtime / "bin").mkdir(parents=True)
        node_artifact = _download_or_validate(
            supplied=args.node_artifact,
            url=NODE_ARTIFACT_URL,
            expected_sha=NODE_ARTIFACT_SHA256,
            destination=stage / "node.tar.xz",
            offline=args.offline,
            label="Node.js",
        )
        _extract_node(node_artifact, runtime / "node")
        node = runtime / "node" / "bin" / "node"
        npm = runtime / "node" / "bin" / "npm"
        node_version = subprocess.check_output(
            [node, "-p", "process.versions.node"], text=True,
        ).strip()
        if node_version != NODE_VERSION:
            raise RuntimeError(
                f"Node artifact version mismatch: expected {NODE_VERSION}, "
                f"found {node_version}"
            )

        artifact = _download_or_validate(
            supplied=args.artifact,
            url=PRIME_AGENT_ARTIFACT_URL,
            expected_sha=PRIME_AGENT_ARTIFACT_SHA256,
            destination=stage / "prime-agent.tgz",
            offline=args.offline,
            label="Prime Agent",
        )
        install_env = os.environ.copy()
        install_env["PATH"] = (
            str(runtime / "node" / "bin") + os.pathsep
            + install_env.get("PATH", "")
        )
        npm_args = [
            str(npm), "install", "--prefix", str(runtime), "--omit=dev",
            "--ignore-scripts", str(artifact),
        ]
        if args.offline:
            npm_args.insert(-1, "--offline")
        _run(npm_args, env=install_env)

        # A venv still points at uv's build-host interpreter even with
        # --relocatable. Copy the complete pinned managed distribution and
        # install into it so the archive contains its interpreter and stdlib.
        install_python_args = [
            uv, "python", "install", PYTHON_VERSION,
            "--managed-python", "--no-config",
        ]
        if args.offline:
            install_python_args.append("--offline")
        _run(install_python_args)
        python = Path(subprocess.check_output(
            [uv, "python", "find", "--managed-python", "--no-project",
             "--no-config", PYTHON_VERSION],
            text=True,
        ).strip()).resolve()
        if not python.is_file():
            raise RuntimeError(f"uv managed Python is missing at {python}")
        shutil.copytree(python.parent.parent, runtime / "python", symlinks=True)
        runtime_python = runtime / "python" / "bin" / f"python{PYTHON_VERSION[:4]}"
        # The copied distribution is now an application-private runtime, not
        # uv's shared managed installation. Remove its PEP 668 marker so the
        # locked kernel dependencies can be installed into this private copy.
        externally_managed = (
            runtime / "python" / "lib" / f"python{PYTHON_VERSION[:4]}"
            / "EXTERNALLY-MANAGED"
        )
        externally_managed.unlink(missing_ok=True)
        actual_python = subprocess.check_output(
            [runtime_python, "-c", "import platform; print(platform.python_version())"],
            text=True,
        ).strip()
        if actual_python != PYTHON_VERSION:
            raise RuntimeError(
                f"Python artifact version mismatch: expected {PYTHON_VERSION}, "
                f"found {actual_python}"
            )
        requirements = stage / "prime-agent-kernel.requirements.txt"
        export_args = [
            uv, "export", "--project", str(KERNEL_ENVIRONMENT), "--locked",
            "--no-dev", "--format", "requirements-txt", "--no-emit-project",
            "--output-file", str(requirements),
        ]
        if args.offline:
            export_args.append("--offline")
        _run(export_args, stdout=subprocess.DEVNULL)
        pip_args = [
            uv, "pip", "install", "--python",
            str(runtime_python), "--system",
            "--requirement", str(requirements),
        ]
        if args.offline:
            pip_args.append("--offline")
        _run(pip_args)
        project_args = [
            uv, "pip", "install", "--python",
            str(runtime_python), "--system", "--no-deps", str(VENDORED_RUNTIME),
        ]
        if args.offline:
            project_args.insert(-2, "--offline")
        _run(project_args)

        # PRIME_AGENT_KERNEL_PYTHON is an override, so the CLI will not create
        # or mutate its own venv. Install every bundled Python skill into the
        # private interpreter just as upstream's kernel bootstrap would.
        skills_root = runtime / "node_modules" / "prime-agent" / "dist" / "skills"
        skill_projects = sorted(
            path.parent for path in skills_root.glob("*/pyproject.toml")
        )
        if not skill_projects:
            raise RuntimeError(f"Prime Agent artifact has no Python skills at {skills_root}")
        skill_args = [
            uv, "pip", "install", "--python", str(runtime_python),
            "--system", "--no-deps", *(str(path) for path in skill_projects),
        ]
        if args.offline:
            skill_args.insert(5, "--offline")
        _run(skill_args)

        required_imports = ("rlm", *KERNEL_REQUIRED_IMPORTS, *KERNEL_SKILL_IMPORTS)
        import_probe = "; ".join(f"import {name}" for name in required_imports)
        _run([runtime_python, "-c", import_probe])

        cli = runtime / "bin" / "prime-agent"
        cli.write_text(
            "#!/bin/sh\n"
            "export PATH=/opt/prime-agent/node/bin:$PATH\n"
            "exec /opt/prime-agent/node_modules/.bin/prime-agent \"$@\"\n"
        )
        cli.chmod(0o755)
        (runtime / "upstream_commit").write_text(commit + "\n")
        (runtime / "runtime_fingerprint").write_text(fingerprint + "\n")
        (runtime / "runtime_metadata.json").write_text(json.dumps({
            "schema_version": 1,
            "upstream_repository": "PrimeIntellect-ai/prime-agent",
            "upstream_commit": commit,
            "artifact_url": PRIME_AGENT_ARTIFACT_URL,
            "artifact_sha256": PRIME_AGENT_ARTIFACT_SHA256,
            "prime_agent_version": PRIME_AGENT_VERSION,
            "node_version": node_version,
            "node_artifact_url": NODE_ARTIFACT_URL,
            "node_artifact_sha256": NODE_ARTIFACT_SHA256,
            "python_version": actual_python,
            "python_abi": "cp311",
            "fingerprint": fingerprint,
        }, indent=2, sort_keys=True) + "\n")

        temporary_output = output.with_suffix(output.suffix + ".tmp")
        with tarfile.open(temporary_output, "w:gz") as archive:
            archive.add(runtime, arcname="opt/prime-agent", filter=_vm_user_owned)
        temporary_output.replace(output)

    validate_runtime_archive_layout(output)
    marker.write_text(fingerprint + "\n")
    sha_marker.write_text(file_sha256(output) + "\n")
    print(f"wrote {output} ({output.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
