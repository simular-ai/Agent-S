from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_s4.harness.base import BuilderResult
from agent_s4.harness.prime.builder import (
    PrimeAgentBuilder,
    context_window_tokens,
    bundled_kernel_python,
    prime_agent_runtime_root,
    validate_prime_agent_binary,
)


def _launch(
    tmp_path: Path, *, exit_code: int = 0, result_event: bool = True,
    timeout: int | None = 600, model: str = "openai/gpt-5",
    context_compaction: dict | None = None,
):
    clone = tmp_path / "clone"
    logs = tmp_path / "logs"
    clone.mkdir()
    logs.mkdir()
    if context_compaction is not None:
        (clone / "shadow_interface").mkdir()
        (clone / "shadow_interface" / "context_compaction.json").write_text(
            json.dumps(context_compaction), encoding="utf-8",
        )
    captured = {}

    def fake_run(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        log = logs / "mutation.jsonl"
        records = [{"type": "agent_end", "messages": []}]
        if result_event:
            records.append({"type": "result", "total_cost_usd": 0.2})
        log.write_text("".join(json.dumps(r) + "\n" for r in records))
        return BuilderResult(exit_code, 0.2, log, False)

    with patch(
        "agent_s4.harness.prime.builder.resolve_prime_agent_binary",
        return_value=Path("/opt/prime-agent/prime-agent"),
    ), patch(
        "agent_s4.harness.prime.builder.run_builder_subprocess",
        side_effect=fake_run,
    ):
        result = PrimeAgentBuilder().launch(
            clone_dir=clone,
            prompt="mutate",
            model=model,
            timeout=timeout,
            env_extra={"EXTRA": "yes"},
            variant_id="v1",
            logs_dir=logs,
            log_name="mutation",
        )
    return result, captured


def test_builder_stages_adapter_and_isolates_prime_state(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    result, captured = _launch(tmp_path)
    argv = captured["argv"]
    env = captured["env"]
    assert result.crashed is False
    assert argv[1].endswith("/.builder_tmp/prime_agent_runner.py")
    assert argv[argv.index("--prime-bin") + 1] == "/opt/prime-agent/prime-agent"
    assert argv[argv.index("--session-dir") + 1].endswith(
        "/.builder_tmp/prime-sessions"
    )
    assert env["HOME"].endswith("/.builder_tmp/prime-home")
    assert env["PRIME_AGENT_CODING_AGENT_DIR"] == env["HOME"]
    assert env["PI_CODING_AGENT_DIR"] == env["HOME"]
    assert env["EXTRA"] == "yes"
    expected_paths = [Path("/opt/prime-agent")]
    if uv_binary := __import__("shutil").which("uv"):
        expected_paths.append(Path(uv_binary).resolve())
    assert captured["sandbox_read_paths"] == expected_paths
    settings = json.loads((Path(env["HOME"]) / "settings.json").read_text())
    assert settings == {
        "compaction": {
            "enabled": True,
            "reserveTokens": 16384,
            "keepRecentTokens": 20000,
        },
        "autoRefine": {"enabled": False},
        "telemetry": {"enabled": False},
    }


def test_builder_writes_strict_openrouter_routing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "novita")
    monkeypatch.setenv("OPENROUTER_ALLOW_FALLBACKS", "false")
    monkeypatch.setenv("OPENROUTER_ZDR", "true")
    monkeypatch.setenv("OPENROUTER_DATA_COLLECTION", "deny")

    _, captured = _launch(
        tmp_path, model="openrouter/qwen/qwen3.8-27b"
    )
    agent_dir = Path(captured["env"]["HOME"])
    models = json.loads((agent_dir / "models.json").read_text())
    assert models["providers"]["openrouter"]["compat"][
        "openRouterRouting"
    ] == {
        "allow_fallbacks": False,
        "order": ["novita"],
        "only": ["novita"],
        "zdr": True,
        "data_collection": "deny",
    }
    settings = json.loads((agent_dir / "settings.json").read_text())
    assert settings["compaction"]["reserveTokens"] == 36096


@pytest.mark.parametrize("enabled", [True, False])
def test_builder_applies_supported_interface_compaction_policy(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://qwen.example/v1")
    monkeypatch.setenv("SHADOW_PRIME_MODEL_CONTEXT_WINDOW", "262144")
    _, captured = _launch(
        tmp_path, model="openai/qwen3.8-27b-deployment",
        context_compaction={
            "enabled": enabled, "reserve_tokens": 40000,
            "keep_recent_tokens": 12000, "summary_max_tokens": 4000,
        },
    )
    agent_dir = Path(captured["env"]["HOME"])
    settings = json.loads((agent_dir / "settings.json").read_text())
    assert settings["compaction"] == {
        "enabled": enabled, "reserveTokens": 40000, "keepRecentTokens": 12000,
    }
    models = json.loads((agent_dir / "models.json").read_text())
    model = models["providers"]["shadow-openai-compatible"]["models"][0]
    assert model["maxTokens"] == 32000
    assert model["contextWindow"] == 262144


@pytest.mark.parametrize("exit_code,result_event", [(2, True), (0, False)])
def test_builder_marks_failed_or_incomplete_adapter_as_crash(
    tmp_path, exit_code, result_event,
):
    result, _ = _launch(
        tmp_path, exit_code=exit_code, result_event=result_event,
    )
    assert result.crashed is True


def test_builder_rejects_too_short_timeout(tmp_path):
    with pytest.raises(ValueError, match="at least 90 seconds"):
        PrimeAgentBuilder().launch(
            clone_dir=tmp_path,
            prompt="x",
            model="m",
            timeout=89,
            env_extra=None,
            variant_id="v",
            logs_dir=tmp_path,
            log_name="x",
        )


def test_builder_accepts_no_wall_clock_limit(tmp_path):
    result, captured = _launch(tmp_path, timeout=None)
    assert result.crashed is False
    assert captured["timeout"] is None


def test_runtime_root_includes_bundled_skills_and_python_runtime(tmp_path):
    package = tmp_path / "prime-agent-package"
    cli = package / "dist" / "bundle" / "cli.js"
    cli.parent.mkdir(parents=True)
    cli.touch()
    assert prime_agent_runtime_root(cli) == package.resolve()


def test_runtime_root_for_standalone_binary_is_its_parent(tmp_path):
    binary = tmp_path / "bin" / "prime-agent"
    binary.parent.mkdir()
    binary.touch()
    assert prime_agent_runtime_root(binary) == binary.parent.resolve()


def test_runtime_root_for_offline_shell_launcher_is_package(tmp_path):
    package = tmp_path / "prime-agent"
    binary = package / "bin" / "prime-agent"
    binary.parent.mkdir(parents=True)
    binary.touch()
    (package / "node").mkdir()
    (package / "node_modules").mkdir()
    assert prime_agent_runtime_root(binary) == package.resolve()


def test_bundled_kernel_python_is_used_when_present(tmp_path):
    runtime = tmp_path / "prime-agent"
    python = runtime / "python" / "bin" / "python3.11"
    python.parent.mkdir(parents=True)
    python.touch()
    assert bundled_kernel_python(runtime) == python.resolve()
    assert bundled_kernel_python(tmp_path / "missing") is None


def test_binary_version_must_match_pin(tmp_path):
    binary = tmp_path / "prime-agent"
    binary.write_text("#!/bin/sh\necho 0.0.0\n")
    binary.chmod(0o755)
    with pytest.raises(RuntimeError, match="does not match"):
        validate_prime_agent_binary(binary)


def test_context_window_tokens_absent_field_means_no_override(tmp_path):
    """An interface without the field leaves Prime on its catalog window."""
    iface = tmp_path / "shadow_interface"
    iface.mkdir()
    (iface / "context_compaction.json").write_text(
        json.dumps({"enabled": True, "reserve_tokens": 16384})
    )
    assert context_window_tokens(tmp_path) is None


def test_context_window_tokens_disabled_policy_means_no_override(tmp_path):
    iface = tmp_path / "shadow_interface"
    iface.mkdir()
    (iface / "context_compaction.json").write_text(
        json.dumps({"enabled": False, "context_window_tokens": 131072})
    )
    assert context_window_tokens(tmp_path) is None


def test_context_window_tokens_missing_file_means_no_override(tmp_path):
    assert context_window_tokens(tmp_path) is None


def test_context_window_tokens_reads_an_explicit_override(tmp_path):
    iface = tmp_path / "shadow_interface"
    iface.mkdir()
    (iface / "context_compaction.json").write_text(
        json.dumps({"enabled": True, "context_window_tokens": 131072})
    )
    assert context_window_tokens(tmp_path) == 131072


@pytest.mark.parametrize("bad", [0, -1, True, "131072", 1.5])
def test_context_window_tokens_rejects_a_present_but_invalid_value(tmp_path, bad):
    """A mutation that writes nonsense should fail loudly, not silently."""
    iface = tmp_path / "shadow_interface"
    iface.mkdir()
    (iface / "context_compaction.json").write_text(
        json.dumps({"enabled": True, "context_window_tokens": bad})
    )
    with pytest.raises(RuntimeError, match="positive integer"):
        context_window_tokens(tmp_path)


def test_catalog_source_ignores_the_interface_window(tmp_path, monkeypatch):
    """A run resumed at Prime's catalog window must not pick the field up.

    prime25v15r4 scored steps 0-5 at the catalog window because the field was
    not plumbed yet, while its interfaces carry 131072. Honouring the field on
    resume would measure later steps in a different window from their parents.
    """
    monkeypatch.setenv("SHADOW_PRIME_CONTEXT_WINDOW_SOURCE", "catalog")
    iface = tmp_path / "shadow_interface"
    iface.mkdir()
    (iface / "context_compaction.json").write_text(
        json.dumps({"enabled": True, "context_window_tokens": 131072})
    )
    assert context_window_tokens(tmp_path) is None


def test_interface_source_is_the_default(tmp_path, monkeypatch):
    monkeypatch.delenv("SHADOW_PRIME_CONTEXT_WINDOW_SOURCE", raising=False)
    iface = tmp_path / "shadow_interface"
    iface.mkdir()
    (iface / "context_compaction.json").write_text(
        json.dumps({"enabled": True, "context_window_tokens": 131072})
    )
    assert context_window_tokens(tmp_path) == 131072


def test_context_window_source_rejects_an_unknown_value(monkeypatch):
    from agent_s4.harness.prime.context_window import context_window_source

    monkeypatch.setenv("SHADOW_PRIME_CONTEXT_WINDOW_SOURCE", "1m")
    with pytest.raises(ValueError, match="must be one of"):
        context_window_source()


def test_builder_declares_self_hosted_openai_deployment(tmp_path, monkeypatch):
    """A self-hosted OpenAI-compatible endpoint is unknown to Prime, so the
    Builder must declare it -- and declare it the way the evaluator does, or
    the interface is written against different model capabilities than it is
    later measured under."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.parasail.io/v1")
    _, captured = _launch(tmp_path, model="openai/qwen3.8-27b-deployment")
    argv = captured["argv"]
    assert argv[argv.index("--model") + 1] == (
        "shadow-openai-compatible/qwen3.8-27b-deployment"
    )
    config = json.loads(
        (Path(captured["env"]["PRIME_AGENT_CODING_AGENT_DIR"]) / "models.json")
        .read_text()
    )
    provider = config["providers"]["shadow-openai-compatible"]
    assert provider["baseUrl"] == "https://api.parasail.io/v1"
    assert provider["apiKey"] == "OPENAI_API_KEY"
    assert provider["models"][0]["id"] == "qwen3.8-27b-deployment"
    assert provider["models"][0]["input"] == ["text", "image"]


def test_builder_leaves_real_openai_models_to_prime(tmp_path, monkeypatch):
    """Without a base URL, openai/... is a genuine OpenAI model; declaring a
    shadow provider for it would point Prime at the wrong endpoint."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    _, captured = _launch(tmp_path, model="openai/gpt-5")
    argv = captured["argv"]
    assert argv[argv.index("--model") + 1] == "openai/gpt-5"
    assert not (
        Path(captured["env"]["PRIME_AGENT_CODING_AGENT_DIR"]) / "models.json"
    ).exists()
