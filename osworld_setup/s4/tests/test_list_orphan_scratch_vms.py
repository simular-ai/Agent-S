"""Tests for the orphan-VM lister."""
import datetime
from unittest.mock import MagicMock, patch


def _fake_describe_response(instances):
    """Build a mock ec2_client.describe_instances response. `instances` is a
    list of (instance_id, launch_time, state) triples."""
    return {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": iid,
                        "LaunchTime": launch,
                        "State": {"Name": state},
                        "Tags": [{"Key": "scratch-vm", "Value": "true"}],
                    }
                    for (iid, launch, state) in instances
                ]
            }
        ]
    }


def test_lists_running_scratch_tagged_instances_with_ages():
    """Returns a list of (instance_id, age_minutes) for tagged running VMs."""
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts" / "ops"))
    from list_orphan_scratch_vms import list_orphans

    now = datetime.datetime(2026, 4, 21, 20, 0, 0, tzinfo=datetime.timezone.utc)
    launch = datetime.datetime(2026, 4, 21, 19, 30, 0, tzinfo=datetime.timezone.utc)

    fake_client = MagicMock()
    fake_client.describe_instances.return_value = _fake_describe_response([
        ("i-running", launch, "running"),
    ])
    with patch("boto3.client", return_value=fake_client):
        orphans = list_orphans(region="us-east-1", now=now)

    assert len(orphans) == 1
    assert orphans[0]["instance_id"] == "i-running"
    assert orphans[0]["age_minutes"] == 30


def test_applies_correct_ec2_filters():
    """describe_instances is called with both tag:scratch-vm and state filters."""
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts" / "ops"))
    from list_orphan_scratch_vms import list_orphans

    fake_client = MagicMock()
    fake_client.describe_instances.return_value = _fake_describe_response([])
    with patch("boto3.client", return_value=fake_client):
        list_orphans(region="us-east-1")

    args, kwargs = fake_client.describe_instances.call_args
    filters = kwargs["Filters"]
    names = {f["Name"] for f in filters}
    assert "tag:scratch-vm" in names
    assert "instance-state-name" in names
