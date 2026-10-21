"""Stamp an Owner tag onto every EC2 instance this repo launches.

Why this is a botocore hook rather than an edit to the provider
---------------------------------------------------------------
The account's SSO role carries two deliberate deny statements:

``DenyTerminateOthersInstances``
    denies ``ec2:TerminateInstances`` whenever an instance's ``Owner`` tag
    does not equal the caller's ``aws:PrincipalTag/Owner``. An instance with
    no ``Owner`` tag at all can never satisfy that, so it can never be
    terminated by us.

``DenyRetaggingOwner``
    denies ``ec2:CreateTags``/``ec2:DeleteTags`` on the ``Owner`` key, so the
    tag cannot be added after the fact.

Together those mean the tag has to be present in the ``RunInstances`` call
itself. The only code that calls ``RunInstances`` lives in OSWorld V2's
vendored ``desktop_env/providers/aws/{provider,manager}.py``, and both the
``OSWorldV2_Base`` and ``OSWorldV2_Evol`` checkouts must stay byte-identical
to the pinned release — ``osworld_v2_preflight._worktree_clean`` fails the run
otherwise. So we inject the tag from outside, on the boto3 default session
that every one of those ``boto3.client('ec2', ...)`` calls resolves through.

Off by default: with ``SHADOW_EVOL_OWNER_TAG`` unset this module changes
nothing, so callers that have no ownership convention are unaffected.
"""

from __future__ import annotations

import os

OWNER_TAG_ENV = "SHADOW_EVOL_OWNER_TAG"

_installed = False


def _inject_owner_tag(params, owner: str) -> None:
    """Add Owner=<owner> to the instance TagSpecification of a RunInstances call.

    An explicit Owner already in the request wins — a caller that set it meant
    it, and silently rewriting ownership is exactly what DenyRetaggingOwner
    exists to prevent.
    """
    specs = params.setdefault("TagSpecifications", [])
    for spec in specs:
        if spec.get("ResourceType") != "instance":
            continue
        tags = spec.setdefault("Tags", [])
        if any(t.get("Key") == "Owner" for t in tags):
            return
        tags.append({"Key": "Owner", "Value": owner})
        return
    specs.append(
        {
            "ResourceType": "instance",
            "Tags": [{"Key": "Owner", "Value": owner}],
        }
    )


def install_owner_tag_hook() -> str | None:
    """Register the hook on boto3's default session. Idempotent.

    Returns the owner value in force, or None when the feature is off.
    Registering on the *default* session is what makes this work without
    touching the provider: ``boto3.client(...)`` with no explicit session
    resolves to it, which is how every vendored EC2 call is made.
    """
    global _installed

    owner = os.environ.get(OWNER_TAG_ENV, "").strip()
    if not owner:
        return None
    if _installed:
        return owner

    import boto3

    if boto3.DEFAULT_SESSION is None:
        boto3.setup_default_session()

    def _handler(params, **_kwargs):
        _inject_owner_tag(params, owner)

    boto3.DEFAULT_SESSION.events.register(
        "provide-client-params.ec2.RunInstances",
        _handler,
        unique_id="shadow-evol-owner-tag",
    )
    _installed = True
    return owner
