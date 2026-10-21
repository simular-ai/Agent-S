#!/usr/bin/env python3
"""Operator tool: list EC2 instances tagged `scratch-vm=true` that are still
running. Use to surface leaks after an orchestrator crash.

Usage:
    python scripts/ops/list_orphan_scratch_vms.py [--region us-east-1]

Output is a table: instance_id, launch_time, age_minutes.
"""
from __future__ import annotations

import argparse
import datetime
import os
from typing import Optional


def list_orphans(region: str, *, now: Optional[datetime.datetime] = None) -> list[dict]:
    """Return a list of {instance_id, launch_time, age_minutes} dicts for
    currently-running EC2 instances tagged `scratch-vm=true` in `region`."""
    import boto3
    ec2 = boto3.client("ec2", region_name=region)
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:scratch-vm", "Values": ["true"]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]},
        ],
    )
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)

    orphans = []
    for reservation in resp.get("Reservations", []):
        for inst in reservation.get("Instances", []):
            launch = inst["LaunchTime"]
            age = (now - launch).total_seconds() / 60.0
            orphans.append({
                "instance_id": inst["InstanceId"],
                "launch_time": launch,
                "age_minutes": int(age),
            })
    return orphans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region",
                        default=os.environ.get("AWS_REGION", "us-east-1"))
    args = parser.parse_args()

    orphans = list_orphans(args.region)
    if not orphans:
        print("No orphan scratch VMs found.")
        return

    print(f"{'INSTANCE_ID':<22}  {'LAUNCH_TIME (UTC)':<26}  {'AGE (min)':>9}")
    for o in sorted(orphans, key=lambda x: x["age_minutes"], reverse=True):
        print(f"{o['instance_id']:<22}  {o['launch_time']!s:<26}  {o['age_minutes']:>9d}")
    print(f"\n{len(orphans)} orphan(s). To terminate manually:")
    print(f"  aws ec2 terminate-instances --region {args.region} --instance-ids "
          + " ".join(o['instance_id'] for o in orphans))


if __name__ == "__main__":
    main()
