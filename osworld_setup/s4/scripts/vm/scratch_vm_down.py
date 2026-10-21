"""Terminate the scratch VM provisioned by scratch_vm_up.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import boto3

STATE_PATH = Path("/tmp/scratch_vm_state.json")


def main() -> int:
    if not STATE_PATH.exists():
        print(f"No state at {STATE_PATH} — nothing to terminate.")
        return 0
    state = json.loads(STATE_PATH.read_text())
    iid = state["instance_id"]
    ec2 = boto3.client("ec2", region_name="us-east-1")
    ec2.terminate_instances(InstanceIds=[iid])
    print(f"Terminated {iid}")
    STATE_PATH.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
