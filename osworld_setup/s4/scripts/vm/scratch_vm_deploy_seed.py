"""Deploy seeds/<island>/ to the live scratch VM as /home/user/shadow_interface/.

Reuses the chunked-base64 tar pattern from agent_s4.vm.utils.upload_shadow_interface
but talks to the VM directly via the state file written by scratch_vm_up.py
(no DesktopEnv controller required).

Usage:
    python scripts/vm/scratch_vm_deploy_seed.py                      # combined island (default)
    python scripts/vm/scratch_vm_deploy_seed.py --island gui
    python scripts/vm/scratch_vm_deploy_seed.py --src seeds/combined # explicit dir
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent.parent
STATE_PATH = Path("/tmp/scratch_vm_state.json")
VM_SHADOW_DIR = "/home/user/shadow_interface"
CHUNK = 50000


def _tar_directory(dir_path: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")]
            for f in files:
                if f.endswith(".pyc"):
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, dir_path)
                tar.add(full, arcname=os.path.join("shadow_interface", rel))
    return buf.getvalue()


def _exec(url: str, command: str, timeout: int = 60) -> dict:
    r = requests.post(url, json={"command": command, "shell": True}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--island", default="combined", help="island name under seeds/")
    p.add_argument("--src", default=None, help="explicit src directory (overrides --island)")
    args = p.parse_args()

    src = Path(args.src) if args.src else REPO / "seeds" / args.island
    if not src.is_dir():
        print(f"src not a directory: {src}", file=sys.stderr)
        return 2

    state = json.loads(STATE_PATH.read_text())
    url = f"http://{state['vm_ip']}:{state['server_port']}/execute"
    print(f"Deploying {src} → {VM_SHADOW_DIR} on {state['vm_ip']}")

    tar_bytes = _tar_directory(src)
    b64 = base64.b64encode(tar_bytes).decode("ascii")
    print(f"  tar size: {len(tar_bytes)} bytes, b64 len: {len(b64)}")

    chunks = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)]

    # First chunk + clean target
    r = _exec(url,
              f"rm -rf {VM_SHADOW_DIR} && "
              f"echo -n '{chunks[0]}' > /tmp/shadow_tar_b64",
              timeout=30)
    if r.get("returncode", -1) != 0:
        print(f"  ! first chunk failed: {r}", file=sys.stderr); return 1

    for i, ch in enumerate(chunks[1:], 2):
        r = _exec(url, f"echo -n '{ch}' >> /tmp/shadow_tar_b64", timeout=30)
        if r.get("returncode", -1) != 0:
            print(f"  ! chunk {i}/{len(chunks)} failed: {r}", file=sys.stderr); return 1
    print(f"  uploaded in {len(chunks)} chunk(s)")

    r = _exec(url,
              f"base64 -d /tmp/shadow_tar_b64 | tar xz -C /home/user/ && "
              f"rm /tmp/shadow_tar_b64 && ls {VM_SHADOW_DIR}/",
              timeout=60)
    if r.get("returncode", -1) != 0:
        print(f"  ! extract failed: {r}", file=sys.stderr); return 1
    print(f"  extracted. {VM_SHADOW_DIR}/ contains:")
    for line in r.get("output", "").strip().splitlines():
        print(f"    {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
