"""Shared model-provider configuration for Prime Agent harnesses."""

from __future__ import annotations

import json
import os


def _parse_bool_env(name: str, default: str) -> bool:
    raw = os.environ.get(name, default).strip().lower()
    if raw not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise ValueError(f"{name} must be true/false, yes/no, on/off, or 1/0")
    return raw in {"1", "true", "yes", "on"}


def openrouter_routing(or_provider=None) -> dict:
    """Resolve one routing policy for host Builder and VM evaluator calls."""
    if or_provider:
        order = [or_provider] if isinstance(or_provider, str) else list(or_provider)
    else:
        raw_order = os.environ.get("OPENROUTER_PROVIDER_ORDER", "")
        order = [item.strip() for item in raw_order.split(",") if item.strip()]

    allow_fallbacks = _parse_bool_env("OPENROUTER_ALLOW_FALLBACKS", "true")
    routing: dict[str, object] = {"allow_fallbacks": allow_fallbacks}
    if order:
        routing["order"] = order
        if not allow_fallbacks:
            # ``order`` expresses preference; ``only`` makes the no-fallback
            # policy explicit and prevents OpenRouter from selecting an
            # unlisted endpoint.
            routing["only"] = order

    if _parse_bool_env("OPENROUTER_ZDR", "false"):
        routing["zdr"] = True
    data_collection = (
        os.environ.get("OPENROUTER_DATA_COLLECTION", "allow").strip().lower()
    )
    if data_collection not in {"allow", "deny"}:
        raise ValueError("OPENROUTER_DATA_COLLECTION must be allow or deny")
    routing["data_collection"] = data_collection
    return routing


# Prime has no built-in provider for a self-hosted OpenAI-compatible
# deployment, so both harnesses declare one under this name.
OPENAI_COMPATIBLE_PROVIDER = "shadow-openai-compatible"


# Measured: under a proxy that returns reasoning inline in ``content`` rather
# than as separate thinking blocks, an 8000-token output cap terminated 33 of
# 45 episodes -- every single message that reached it was the last of its
# episode. The same model through OpenRouter emitted 8000+ token replies in 38
# messages and continued normally, peaking near 17k. 32k leaves real headroom;
# the deployment accepts up to 65536.
DEFAULT_MAX_OUTPUT_TOKENS = 32000

# Used when neither the interface nor the caller supplies a context window.
# Set SHADOW_PRIME_MODEL_CONTEXT_WINDOW to the deployment's verified limit
# (input + output), e.g. 262144 for a 256K Qwen3.8 deployment.
DEFAULT_CONTEXT_WINDOW = 131072


def default_context_window() -> int:
    raw = os.environ.get("SHADOW_PRIME_MODEL_CONTEXT_WINDOW", "").strip()
    if not raw:
        return DEFAULT_CONTEXT_WINDOW
    value = int(raw)
    if value <= 0:
        raise ValueError(
            f"SHADOW_PRIME_MODEL_CONTEXT_WINDOW must be positive, got {raw!r}"
        )
    return value


def openai_compatible_models_config(
    model_id: str,
    base_url: str,
    context_window: int | None,
    max_output_tokens: int | None = None,
) -> str:
    """Declare a self-hosted OpenAI-compatible deployment to Prime.

    Builder and evaluator must describe the same deployment the same way: an
    interface evolved against one set of model capabilities and then run under
    another is not the interface that was measured.
    """
    return json.dumps(
        {
            "providers": {
                OPENAI_COMPATIBLE_PROVIDER: {
                    "baseUrl": base_url,
                    "api": "openai-completions",
                    "apiKey": "OPENAI_API_KEY",
                    "compat": {
                        "supportsDeveloperRole": False,
                        "supportsReasoningEffort": False,
                        "maxTokensField": "max_tokens",
                        "thinkingFormat": "qwen",
                        "requiresReasoningContentOnAssistantMessages": True,
                    },
                    "models": [
                        {
                            "id": model_id,
                            "name": "Qwen3.8",
                            "reasoning": True,
                            "input": ["text", "image"],
                            "contextWindow": context_window or default_context_window(),
                            "maxTokens": max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                        }
                    ],
                },
            },
        },
        separators=(",", ":"),
    )
