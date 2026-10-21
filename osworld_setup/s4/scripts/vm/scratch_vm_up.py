"""Provision a scratch OSWorld VM and leave it running for ad-hoc probing.

Writes the VM handle to /tmp/scratch_vm_state.json so other scripts
(scratch_vm_exec.py, scratch_vm_down.py) can talk to it. The VM survives
this script exiting because we never call DesktopEnv.close().
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from desktop_env.desktop_env import DesktopEnv

STATE_PATH = Path("/tmp/scratch_vm_state.json")


def main() -> int:
    if STATE_PATH.exists():
        existing = json.loads(STATE_PATH.read_text())
        print(f"State file already exists: {STATE_PATH}")
        print(f"  instance: {existing['instance_id']}  ip: {existing['vm_ip']}:{existing['server_port']}")
        print("  Run scratch_vm_down.py first if you want a fresh one.")
        return 1

    env = DesktopEnv(
        provider_name="aws",
        region="us-east-1",
        os_type="Ubuntu",
        headless=True,
        require_a11y_tree=True,
    )

    state = {
        "vm_ip": env.vm_ip,
        "server_port": env.server_port,
        "chromium_port": env.chromium_port,
        "vnc_port": env.vnc_port,
        "vlc_port": env.vlc_port,
        "instance_id": env.path_to_vm,
        "client_password": env.client_password,
    }
    STATE_PATH.write_text(json.dumps(state, indent=2))

    # Tag for orphan discovery
    import boto3
    ec2 = boto3.client("ec2", region_name="us-east-1")
    ec2.create_tags(
        Resources=[env.path_to_vm],
        Tags=[
            {"Key": "scratch-vm", "Value": "true"},
            {"Key": "Name", "Value": "atree-probe"},
            {"Key": "purpose", "Value": "atree-investigation"},
        ],
    )

    print()
    print(f"VM up. State written to {STATE_PATH}")
    print(f"  instance:  {state['instance_id']}")
    print(f"  ip:port:   {state['vm_ip']}:{state['server_port']}")
    print(f"  password:  {state['client_password']}")
    print()
    print("Use scripts/vm/scratch_vm_exec.py '<bash command>' to probe.")
    print("Use scripts/vm/scratch_vm_down.py to terminate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
