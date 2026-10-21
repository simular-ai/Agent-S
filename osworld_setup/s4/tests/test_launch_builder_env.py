"""Tests for the builder subprocess env construction.

Target moved from island_common._build_subprocess_env to
agent_s4.harness.builder_common.build_subprocess_env during the
framework refactor. The new signature takes explicit strip_keys so the
Claude builder strips ANTHROPIC_API_KEY and the Qwen builder can strip
OpenAI keys if needed.
"""

from agent_s4.harness.builder_common import build_subprocess_env


_CLAUDE_STRIP = ("ANTHROPIC_API_KEY", "CLAUDECODE")


def test_build_subprocess_env_excludes_claude_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-xxx")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("HARMLESS", "keep-me")

    env = build_subprocess_env(None, strip_keys=_CLAUDE_STRIP)
    assert "ANTHROPIC_API_KEY" not in env
    assert "CLAUDECODE" not in env
    assert env["HARMLESS"] == "keep-me"


def test_build_subprocess_env_merges_env_extra(monkeypatch):
    monkeypatch.setenv("EXISTING", "old")
    env = build_subprocess_env(
        {"EXISTING": "new", "SCRATCH_VM_IP": "10.0.0.5"},
        strip_keys=_CLAUDE_STRIP,
    )
    assert env["EXISTING"] == "new"
    assert env["SCRATCH_VM_IP"] == "10.0.0.5"


def test_build_subprocess_env_env_extra_none_equivalent_to_empty_dict(monkeypatch):
    monkeypatch.setenv("FOO", "bar")
    assert (
        build_subprocess_env(None, strip_keys=_CLAUDE_STRIP)
        == build_subprocess_env({}, strip_keys=_CLAUDE_STRIP)
    )


def test_build_subprocess_env_no_strip_keys_keeps_everything(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk")
    env = build_subprocess_env(None)           # no strip_keys
    assert env["ANTHROPIC_API_KEY"] == "sk"


def test_build_subprocess_env_never_exposes_evaluator_or_aws_secrets(monkeypatch):
    monkeypatch.setenv("GITLAB_PRIVATE_TOKEN", "gold-secret")
    monkeypatch.setenv("OSWORLD_EVAL_MODEL_API_KEY", "grader-secret")
    monkeypatch.setenv("OSWORLD_USER_SIM_API_KEY", "sim-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "cloud-secret")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "builder-key")
    env = build_subprocess_env(None)
    assert env["ANTHROPIC_AUTH_TOKEN"] == "builder-key"
    assert "GITLAB_PRIVATE_TOKEN" not in env
    assert "OSWORLD_EVAL_MODEL_API_KEY" not in env
    assert "OSWORLD_USER_SIM_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
