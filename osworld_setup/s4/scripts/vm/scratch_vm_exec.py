"""Run a shell command on the live scratch VM (provisioned by scratch_vm_up.py).

Reads /tmp/scratch_vm_state.json, POSTs to the VM's /execute endpoint, prints
{returncode, stdout, stderr}.

Usage:
    python scripts/vm/scratch_vm_exec.py 'echo hello'
    python scripts/vm/scratch_vm_exec.py --timeout 30 'apt-get install ...'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

STATE_PATH = Path("/tmp/scratch_vm_state.json")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command", help="bash command to run on the VM")
    p.add_argument("--timeout", type=int, default=30, help="HTTP timeout (s)")
    p.add_argument("--quiet", action="store_true", help="print only stdout")
    args = p.parse_args()

    if not STATE_PATH.exists():
        print(f"No VM state at {STATE_PATH}. Run scripts/vm/scratch_vm_up.py first.", file=sys.stderr)
        return 2

    state = json.loads(STATE_PATH.read_text())
    url = f"http://{state['vm_ip']}:{state['server_port']}/execute"

    r = requests.post(
        url, json={"command": args.command, "shell": True}, timeout=args.timeout,
    )
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text}", file=sys.stderr)
        return 1
    j = r.json()

    if args.quiet:
        sys.stdout.write(j.get("output", ""))
        return j.get("returncode", 0)

    print(f"$ {args.command}")
    print(f"  returncode: {j.get('returncode')}")
    if j.get("output"):
        print("  stdout:")
        for line in j["output"].splitlines():
            print(f"    {line}")
    if j.get("error"):
        print("  stderr:")
        for line in j["error"].splitlines():
            print(f"    {line}")
    return j.get("returncode", 0)


if __name__ == "__main__":
    sys.exit(main())
