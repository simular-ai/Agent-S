"""VM addressing has to be opt-in, and has to reach the evaluator.

The AWS provider returns each instance's private address unless
``use_public_ip`` is set. An orchestrator inside the benchmark VPC wants that;
an orchestrator outside it cannot route those addresses at all, and the
failure is silent -- connections sit in SYN-SENT, the setup controller never
reports ready, and every task burns its retries and records an imputed zero
that counts as fitness. So the flag has to travel from the launcher all the
way into DesktopEnv, and the default has to stay private for in-VPC runs.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _args(**overrides):
    base = dict(
        path_to_vm=None,
        provider_name="aws",
        screen_width=1920,
        screen_height=1080,
        headless=True,
        client_password="",
        region="us-east-1",
        snapshot_name="ami-test",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _spec():
    from agent_s4.eval.osworld import EnvSpec

    return EnvSpec(
        os_type="Ubuntu",
        snapshot_name="ami-test",
        instance_type=None,
        volume_size=60,
        enable_proxy=True,
        force_disable_vnc=False,
        force_disable_recording=False,
        cache_dir="/tmp/cache",
        is_v2=True,
    )


def test_v2_env_kwargs_default_to_private_addressing():
    """In-VPC runs must keep private addressing without asking for it."""
    from agent_s4.eval.osworld_v2 import build_env_kwargs

    kwargs = build_env_kwargs(_spec(), _args())

    assert kwargs["use_public_ip"] is False


def test_v2_env_kwargs_forward_public_addressing_when_requested():
    from agent_s4.eval.osworld_v2 import build_env_kwargs

    kwargs = build_env_kwargs(_spec(), _args(use_public_ip=True))

    assert kwargs["use_public_ip"] is True


def test_v1_env_kwargs_never_carry_the_v2_only_flag():
    """v1's DesktopEnv has no such parameter and would reject it."""
    from agent_s4.eval.osworld import build_env_kwargs

    kwargs = build_env_kwargs(_spec(), _args(use_public_ip=True))

    assert "use_public_ip" not in kwargs


@pytest.mark.parametrize(
    "value,expected",
    [(None, False), ("false", False), ("true", True), ("1", True), ("no", False)],
)
def test_eval_command_requests_public_addressing_only_when_enabled(
    value, expected, monkeypatch,
):
    """The flag reaches the eval subprocess through the environment."""
    from agent_s4.harness.prime.openrouter import _parse_bool_env

    if value is None:
        monkeypatch.delenv("SHADOW_EVOL_USE_PUBLIC_IP", raising=False)
    else:
        monkeypatch.setenv("SHADOW_EVOL_USE_PUBLIC_IP", value)

    assert _parse_bool_env("SHADOW_EVOL_USE_PUBLIC_IP", "false") is expected


def test_eval_command_rejects_a_malformed_addressing_flag(monkeypatch):
    """A typo must fail at config time, not after 40 tasks have hung."""
    from agent_s4.harness.prime.openrouter import _parse_bool_env

    monkeypatch.setenv("SHADOW_EVOL_USE_PUBLIC_IP", "public")
    with pytest.raises(ValueError, match="must be true/false"):
        _parse_bool_env("SHADOW_EVOL_USE_PUBLIC_IP", "false")


def test_osworld_cli_accepts_the_addressing_flag():
    """The eval module must accept the flag ubuntu_vm appends."""
    from agent_s4.eval.osworld import add_common_args
    import argparse

    parser = argparse.ArgumentParser()
    add_common_args(parser)

    assert parser.parse_args(["--use-public-ip"]).use_public_ip is True
    assert parser.parse_args([]).use_public_ip is False
