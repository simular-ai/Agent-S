"""The Owner tag is the only thing that makes a task VM terminable.

DenyTerminateOthersInstances allows terminate only when an instance's Owner tag
matches the caller's principal tag, and DenyRetaggingOwner blocks adding it
afterwards -- so if RunInstances goes out without it, that instance is
unreclaimable for good. These tests pin the injection rather than the wiring.
"""
from __future__ import annotations

import agent_s4.vm.owner_tag as owner_tag
from agent_s4.vm.owner_tag import (
    OWNER_TAG_ENV,
    _inject_owner_tag,
    install_owner_tag_hook,
)


def test_adds_the_spec_when_the_request_has_no_tags():
    params = {}
    _inject_owner_tag(params, "example-owner")
    assert params == {
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": [{"Key": "Owner", "Value": "example-owner"}]}
        ]
    }


def test_appends_to_the_providers_existing_name_tag():
    # What the vendored provider actually sends (provider.py sets Name only).
    params = {
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": "shadow-evol-abc"}]}
        ]
    }
    _inject_owner_tag(params, "example-owner")
    tags = params["TagSpecifications"][0]["Tags"]
    assert {"Key": "Name", "Value": "shadow-evol-abc"} in tags
    assert {"Key": "Owner", "Value": "example-owner"} in tags


def test_ignores_a_non_instance_spec_and_adds_its_own():
    params = {"TagSpecifications": [{"ResourceType": "volume", "Tags": []}]}
    _inject_owner_tag(params, "example-owner")
    kinds = [s["ResourceType"] for s in params["TagSpecifications"]]
    assert kinds == ["volume", "instance"]


def test_never_rewrites_an_owner_the_caller_set():
    params = {
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": [{"Key": "Owner", "Value": "someone"}]}
        ]
    }
    _inject_owner_tag(params, "example-owner")
    assert params["TagSpecifications"][0]["Tags"] == [
        {"Key": "Owner", "Value": "someone"}
    ]


def test_hook_is_off_without_the_env_var(monkeypatch):
    monkeypatch.delenv(OWNER_TAG_ENV, raising=False)
    monkeypatch.setattr(owner_tag, "_installed", False)
    assert install_owner_tag_hook() is None


def test_blank_env_var_counts_as_off(monkeypatch):
    monkeypatch.setenv(OWNER_TAG_ENV, "   ")
    monkeypatch.setattr(owner_tag, "_installed", False)
    assert install_owner_tag_hook() is None
