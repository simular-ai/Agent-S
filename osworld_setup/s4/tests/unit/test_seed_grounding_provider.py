from __future__ import annotations


def test_explicit_openrouter_grounding_wins_over_ambient_openai(monkeypatch):
    from seeds.combined.helpers.gui import _grounding_provider

    monkeypatch.setenv("SHADOW_GROUNDING_MODEL", "openrouter/qwen/qwen3.8-27b")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://old-openai.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "old-key")
    monkeypatch.setenv("OPENAI_MODEL", "/deployments/old/Qwen/Qwen3.8")

    assert _grounding_provider() == (
        "openai",
        "https://openrouter.ai/api/v1/chat/completions",
        "or-key",
        "qwen/qwen3.8-27b",
    )
