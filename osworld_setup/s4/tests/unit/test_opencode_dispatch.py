"""Unit tests for the --harness flag in agent_s4.eval.osworld."""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def skip_runtime_archive_for_parser_tests(monkeypatch):
    monkeypatch.setenv("SHADOW_EVOL_DRYRUN", "1")


def _run_config(argv):
    """Invoke osworld.config() with a synthesized sys.argv."""
    from agent_s4.eval import osworld
    with patch.object(sys, "argv", ["osworld"] + argv):
        return osworld.config()


def test_config_harness_opencode_accepts_freeform_model(monkeypatch):
    """--harness opencode with a free-form OpenRouter slug parses cleanly
    (no argparse `choices` validation fires)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    args = _run_config([
        "--harness", "opencode",
        "--model", "openrouter/qwen/qwen3.5-397b-a17b",
        "--task_ids", "t1",
    ])
    assert args.harness == "opencode"
    assert args.model == "openrouter/qwen/qwen3.5-397b-a17b"


def test_config_harness_opencode_rejects_unknown_provider(capsys):
    """A model slug without a known provider prefix fails at config() time
    rather than reaching the upstream API as a bogus model id."""
    with pytest.raises(SystemExit):
        _run_config(["--harness", "opencode", "--model", "sonnet", "--task_ids", "t1"])
    err = capsys.readouterr().err
    assert "opencode" in err.lower()
    assert "provider prefix" in err


def test_config_default_harness_is_prime_agent(monkeypatch):
    """Without --harness the task agent stays Prime Agent."""
    from agent_s4.harness.prime.evaluator import (
        MODEL_CONFIG as PRIME_CFG,
    )

    args = _run_config(["--model", "openai/test", "--task_ids", "t1"])
    assert args.harness == "prime_agent"
    assert args.model_cfg is PRIME_CFG["prime_agent"]


def test_config_harness_opencode_requires_openrouter_key(capsys, monkeypatch):
    """--harness opencode with no OPENROUTER_API_KEY in env and no
    --openrouter_api_key flag hard-fails at config() time. OpenCode's
    downstream 'User not found' error with an empty key is misleading
    (cf. sst/opencode#2245); fail early with a clear message instead."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        _run_config([
            "--harness", "opencode",
            "--model", "openrouter/qwen/qwen3.5-397b-a17b",
            "--task_ids", "t1",
        ])
    err = capsys.readouterr().err
    assert "OPENROUTER_API_KEY" in err


def test_config_dispatch_produces_correct_model_cfg_for_each_harness(monkeypatch):
    """End-to-end check at the unit level: for each supported (harness,
    model) combination, config() produces an args.model_cfg whose
    'binary' field matches the expected agent CLI."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")

    cases = [
        # (argv, expected binary)
        (["--model", "openai/test", "--task_ids", "t1"], "prime-agent"),
        (["--harness", "prime_agent", "--model", "openai/test",
          "--task_ids", "t1"], "prime-agent"),
        (["--harness", "opencode", "--model", "openrouter/qwen/qwen3.5-397b-a17b",
          "--task_ids", "t1"], "opencode"),
    ]
    for argv, expected_binary in cases:
        args = _run_config(argv)
        assert args.model_cfg["binary"] == expected_binary, \
            f"argv={argv} produced model_cfg={args.model_cfg}"


def test_opencode_accepts_selfhosted_model_and_budgets(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:12345/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "local-only")
    args = _run_config(["--harness", "opencode", "--model", "openai/test",
                        "--task_ids", "t1", "--max-steps", "50", "--no-time-limit"])
    assert args.max_steps == 50 and args.no_time_limit
    assert args.model_cfg["binary"] == "opencode"


def test_opencode_invalid_sampling_rejected_before_vm_launch(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "local-only")
    monkeypatch.setenv("SHADOW_OPENCODE_TOP_P", "nan")
    with pytest.raises(SystemExit):
        _run_config(["--harness", "opencode", "--model", "openrouter/test", "--task_ids", "t1"])
