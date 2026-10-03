"""Helpers for any OpenAI-compatible backend (LM Studio, Ollama, vLLM, etc.).

LM Studio exposes an OpenAI-compatible server (by default
``http://localhost:1234/v1``) with ``GET /models`` and
``POST /chat/completions``. Everything in ``gui_agents`` talks to LLMs
through the ``openai`` python package, so any such server already works
via ``engine_type="openai"`` + ``base_url`` -- this module just adds
sensible defaults, URL normalization, and model discovery for the UI.
"""

import json
import urllib.request
import ipaddress
from urllib.parse import urlsplit, urlunsplit
from typing import Dict, List, Optional

LMSTUDIO_DEFAULT_URL = "http://localhost:1234/v1"
LMSTUDIO_DEFAULT_API_KEY = "lm-studio"

OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"

# Prefill values used by the web UI provider dropdown.
PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "openai": {"base_url": "", "api_key": "", "hint": "Uses OPENAI_API_KEY env var."},
    "lmstudio": {
        "base_url": LMSTUDIO_DEFAULT_URL,
        "api_key": LMSTUDIO_DEFAULT_API_KEY,
        "hint": "LM Studio local server. Serve a VISION-capable model (e.g. qwen2-vl, llava,Pixtral) with 'Serve on local network' off is fine.",
    },
    "openai_compatible": {
        "base_url": LMSTUDIO_DEFAULT_URL,
        "api_key": LMSTUDIO_DEFAULT_API_KEY,
        "hint": "Any OpenAI-compatible server (text-generation-webui, llama.cpp server, vLLM, Ollama...). Must support chat completions + vision (image_url).",
    },
    "ollama": {
        "base_url": OLLAMA_DEFAULT_URL,
        "api_key": "ollama",
        "hint": "Ollama OpenAI endpoint. Serve a vision model (e.g. llava, qwen2-vl, llama3.2-vision).",
    },
    "vllm": {
        "base_url": "",
        "api_key": "",
        "hint": "vLLM OpenAI server URL, e.g. http://localhost:8000/v1. Uses vLLM_API_KEY env var if api key left blank.",
    },
    "open_router": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "",
        "hint": "Uses OPENROUTER_API_KEY env var if api key left blank.",
    },
    "anthropic": {"base_url": "", "api_key": "", "hint": "Uses ANTHROPIC_API_KEY."},
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": "",
        "hint": "Uses GEMINI_API_KEY.",
    },
    "huggingface": {
        "base_url": "",
        "api_key": "",
        "hint": "Hugging Face Inference Endpoint URL. Uses HF_TOKEN.",
    },
    "codex": {
        "base_url": "",
        "api_key": "",
        "hint": "ChatGPT/Codex subscription through the Codex SDK. Use Login with Codex; no API key is stored.",
    },
}

# engine_type aliases that should behave like plain OpenAI (custom base_url).
OPENAI_COMPATIBLE_ALIASES = {
    "lmstudio",
    "lm_studio",
    "lm-studio",
    "openai_compatible",
    "openai-compatible",
    "openai_compat",
    "local",
    "local-openai",
}


def normalize_base_url(base_url: Optional[str]) -> str:
    """Normalize an OpenAI-compatible base URL.

    - strips whitespace/trailing slashes
    - appends ``/v1`` to a bare server root
    - preserves explicit API/custom paths, including Gemini and Azure v1 paths
    """
    if not base_url:
        return ""
    parts = urlsplit(base_url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("Endpoint must be an absolute HTTP(S) URL")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError(
            "Endpoint URLs cannot contain credentials, queries, or fragments"
        )
    _ = parts.port  # Validate the port before constructing a request.
    path = parts.path.rstrip("/") or "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def is_local_url(base_url: Optional[str]) -> bool:
    if not base_url:
        return False
    try:
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or parts.username or parts.password:
            return False
        host = parts.hostname or ""
        if host.lower() == "localhost":
            return True
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_lmstudio_params(engine_params: Dict) -> Dict:
    """Fill defaults for LM Studio / generic OpenAI-compatible engines."""
    params = dict(engine_params)
    if not params.get("base_url"):
        params["base_url"] = LMSTUDIO_DEFAULT_URL
    else:
        params["base_url"] = normalize_base_url(params["base_url"])
    if not params.get("api_key") and is_local_url(params["base_url"]):
        params["api_key"] = LMSTUDIO_DEFAULT_API_KEY
    # Rewrite the alias to plain openai so downstream isinstance checks hold.
    params["engine_type"] = "openai"
    return params


def list_openai_models(
    base_url: str, api_key: Optional[str] = None, timeout: int = 10
) -> List[Dict]:
    """List models from any OpenAI-compatible ``GET {base_url}/models``."""
    url = normalize_base_url(base_url).rstrip("/") + "/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    return []


def test_openai_connection(
    base_url: str, api_key: Optional[str] = None, timeout: int = 10
) -> Dict:
    """Return {'ok': bool, 'models': [...], 'error': str} for UI health checks."""
    try:
        models = list_openai_models(base_url, api_key, timeout=timeout)
        return {
            "ok": True,
            "models": [m.get("id", m) for m in models if isinstance(m, dict)],
            "raw": models,
        }
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        return {"ok": False, "models": [], "error": str(exc)}
