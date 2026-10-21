from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from agent_s4.harness.prime.compaction import compaction_settings, load_compaction_config


def _written_json(script, target):
    words = shlex.split(script)
    return json.loads(words[words.index(target) - 2])


@pytest.mark.parametrize("enabled", [True, False])
def test_osworld_policy_reaches_actual_prime_settings(tmp_path, monkeypatch, caplog, enabled):
    from agent_s4.eval.osworld import _build_agent_command
    from agent_s4.harness.prime.evaluator import MODEL_CONFIG

    monkeypatch.setenv("OPENAI_BASE_URL", "https://qwen.example/v1")
    monkeypatch.delenv("SHADOW_PRIME_CONTEXT_WINDOW_SOURCE", raising=False)
    monkeypatch.setenv("SHADOW_PRIME_MODEL_CONTEXT_WINDOW", "262144")
    policy = {
        "enabled": enabled, "reserve_tokens": 16384, "keep_recent_tokens": 12000,
        "context_window_tokens": 131072, "summary_max_tokens": 4000,
        "summary_instructions": "custom prompt", "min_compact_tokens": 4000,
    }
    (tmp_path / "context_compaction.json").write_text(json.dumps(policy))
    args = SimpleNamespace(
        model="openai/qwen3.8-27b-deployment",
        timeout=1200, shadow_dir=str(tmp_path),
    )
    built = _build_agent_command(
        args, MODEL_CONFIG["prime_agent"], "task", "policy",
        api_key=None, openrouter_key=None, parasail_key=None,
    )
    settings = _written_json(built.script, "/tmp/claude_task/prime-config/settings.json")
    assert settings["compaction"] == {
        "enabled": enabled, "reserveTokens": 36096, "keepRecentTokens": 12000,
    }
    models = _written_json(built.script, "/tmp/claude_task/prime-config/models.json")
    model = models["providers"]["shadow-openai-compatible"]["models"][0]
    assert model["maxTokens"] == 32000
    assert model["contextWindow"] == (131072 if enabled else 262144)
    for name in ("summary_max_tokens", "summary_instructions", "min_compact_tokens"):
        assert name in caplog.text


def test_reserve_preserves_larger_user_value_and_tracks_requested_output():
    assert compaction_settings({"reserve_tokens": 45000}, max_output_tokens=32000)["reserveTokens"] == 45000
    assert compaction_settings(max_output_tokens=65536)["reserveTokens"] == 69632


@pytest.mark.parametrize("policy", [
    {"enabled": "false"}, {"reserve_tokens": True}, {"reserve_tokens": 0},
    {"reserve_tokens": "32000"}, {"keep_recent_tokens": -1},
    {"keep_recent_tokens": 0}, {"keep_recent_tokens": 1.5},
])
def test_invalid_policy_fails_before_launch(policy):
    with pytest.raises(ValueError):
        compaction_settings(policy, context_window=262144, max_output_tokens=32000)


@pytest.mark.parametrize("policy,window", [
    ({"keep_recent_tokens": 226048}, 262144),
    ({"reserve_tokens": 262144}, 262144),
    ({}, 32768),
])
def test_invalid_combined_budget_fails_before_launch(policy, window):
    with pytest.raises(ValueError, match="must be smaller"):
        compaction_settings(policy, context_window=window, max_output_tokens=32000)


@pytest.mark.parametrize("raw", ["[]", "null", "invalid"])
def test_invalid_policy_file_is_not_silently_ignored(tmp_path, raw):
    (tmp_path / "context_compaction.json").write_text(raw)
    with pytest.raises(ValueError):
        load_compaction_config(tmp_path)


def test_pinned_runtime_compacts_and_dispatches_overflow_recovery_offline():
    root = Path(__file__).resolve().parents[2]
    runtime = root / "vm_assets/prime_agent_host/opt/prime-agent"
    node = runtime / "node/bin/node"
    session = runtime / "node_modules/prime-agent/dist/core/agent-session.js"
    if not node.is_file() or not session.is_file():
        pytest.skip("optional pinned Prime runtime cache is not installed")
    result = subprocess.run(
        [str(node), str(root / "tests/fixtures/prime_compaction_runtime.mjs"), str(runtime),
         json.dumps(compaction_settings(context_window=262144, max_output_tokens=32000))],
        capture_output=True, text=True, timeout=30, check=True,
    )
    evidence = json.loads(result.stdout)
    assert evidence["thresholdTokens"] == 226050
    assert evidence["summaryBudget"] == 28876
    assert evidence["compactedMessageCount"] < evidence["originalMessageCount"]
    assert evidence["overflowDispatch"] == {"reason": "overflow", "retry": True}
