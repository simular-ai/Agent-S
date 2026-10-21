"""Explicit, shared Builder/Evaluator configuration for pinned OpenCode.

Only SHADOW_OPENCODE_* variables are read: resuming a Prime experiment must
never pick up a different harness's configuration implicitly.
"""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
import warnings
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_CONTEXT = 131072
DEFAULT_OUTPUT = 32000
COMPACTION_MARGIN = 4096
HERMES_PLUGIN = Path(__file__).resolve().parent / "hermes_sanitizer.js"
TOOL_LIMITS_PLUGIN = Path(__file__).resolve().parent / "tool_limits.js"
# Appended to every system prompt when the endpoint emulates tool calling in
# the prompt (Hermes <tool_call> text). Qwen otherwise drifts into its native
# `<function=NAME>` syntax when choosing among several tools: on the s_009
# opening this took unparseable first replies from 9/20 to 0/30. It states only
# the correct form: an earlier wording that named the forbidden syntax was read
# by the model as an instruction to use it (if-001 s_017).
HERMES_FORMAT_NOTE = (
    "\n\n## Tool-call format\n\nEvery tool call is one JSON object with exactly two keys, "
    '"name" and "arguments", inside <tool_call></tool_call> tags:\n<tool_call>\n'
    '{"name": "bash", "arguments": {"command": "ls /home/user"}}\n</tool_call>\n'
    "Write it as plain JSON: unescaped quotes around keys and string values, and the "
    "arguments object written directly (not as a string)."
)


def positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_context_config(shadow_dir: str | Path | None) -> dict:
    if shadow_dir is None:
        return {}
    path = Path(shadow_dir) / "context_compaction.json"
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _integer(env: dict, name: str, fallback: int) -> int:
    raw = env.get("SHADOW_OPENCODE_" + name, "").strip()
    return positive_int(name, int(raw) if raw else fallback)


def _sampling(env: dict, name: str, upper: float) -> float | None:
    raw = env.get("SHADOW_OPENCODE_" + name, "").strip()
    if not raw:
        return None
    value = float(raw)
    if not math.isfinite(value) or not 0 <= value <= upper:
        raise ValueError(f"SHADOW_OPENCODE_{name} must be finite and in [0, {upper}]")
    return value


def hermes_tool_format(env: dict | None = None) -> bool:
    env = os.environ if env is None else env
    raw = env.get("SHADOW_OPENCODE_HERMES_TOOL_FORMAT", "").strip()
    if raw not in {"", "0", "1"}:
        raise ValueError("SHADOW_OPENCODE_HERMES_TOOL_FORMAT must be 0 or 1")
    return raw == "1"


# Native tools that may be withheld from the model. Under prompt-emulated tool
# calling, `write` carries whole files in one JSON string and broke its syntax
# on ~9% of calls (bash ~0.7%); without it the model writes files via bash.
DISABLEABLE_TOOLS = {"write", "edit", "todowrite", "webfetch", "skill", "glob", "grep"}


def disabled_tools(env: dict | None = None) -> list[str]:
    env = os.environ if env is None else env
    names = [
        n.strip()
        for n in env.get("SHADOW_OPENCODE_DISABLED_TOOLS", "").split(",")
        if n.strip()
    ]
    unknown = sorted(set(names) - DISABLEABLE_TOOLS)
    if unknown:
        raise ValueError(
            f"SHADOW_OPENCODE_DISABLED_TOOLS: cannot disable {unknown}; "
            f"allowed: {sorted(DISABLEABLE_TOOLS)}"
        )
    # v1.18.34 maps write/edit/patch onto one `edit` permission, last key
    # winning, so disabling either withholds both. Say so in the config.
    if {"write", "edit"} & set(names):
        names += ["write", "edit"]
    return sorted(set(names))


def max_bash_timeout_ms(env: dict | None = None) -> int | None:
    env = os.environ if env is None else env
    raw = env.get("SHADOW_OPENCODE_MAX_BASH_TIMEOUT_MS", "").strip()
    if not raw:
        return None
    value = positive_int("SHADOW_OPENCODE_MAX_BASH_TIMEOUT_MS", int(raw))
    if value > 1_800_000:
        raise ValueError(
            "SHADOW_OPENCODE_MAX_BASH_TIMEOUT_MS must stay below the 1800s idle budget"
        )
    return value


def build_config(
    model: str,
    system_prompt: str,
    context: dict | None = None,
    *,
    max_steps: int | None = None,
    env: dict | None = None,
    hermes_plugin_path: str | Path | None = None,
    tool_limits_plugin_path: str | Path | None = None,
) -> dict:
    """Produce OpenCode v1.18.34's native config, without embedding secrets.

    Unlike Prime's historical policy, enabled=false disables compaction but
    does not discard the declared model context limit.
    """
    env = os.environ if env is None else env
    context = {} if context is None else context
    if not isinstance(context, dict):
        raise ValueError("context compaction config must be an object")
    provider, sep, model_id = model.partition("/")
    if not sep or not model_id or provider not in {"openai", "openrouter", "parasail"}:
        raise ValueError(
            "OpenCode model must be openai/<id>, openrouter/<id>, or parasail/<id>"
        )
    enabled = context.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    window = _integer(
        env, "CONTEXT_WINDOW", context.get("context_window_tokens", DEFAULT_CONTEXT)
    )
    output = _integer(env, "MAX_OUTPUT_TOKENS", DEFAULT_OUTPUT)
    # An explicit override wins over the interface, as for the other limits.
    # Prime never applied the interface field (its summary budget is
    # 0.8 * reserve); the override is how a run matches that.
    raw_summary = env.get("SHADOW_OPENCODE_SUMMARY_MAX_TOKENS", "").strip()
    summary_output = (
        int(raw_summary) if raw_summary else context.get("summary_max_tokens")
    )
    if summary_output is not None:
        summary_output = positive_int("summary_max_tokens", summary_output)
        if summary_output > output:
            raise ValueError("summary_max_tokens must not exceed MAX_OUTPUT_TOKENS")
    reserve = max(
        _integer(env, "RESERVE_TOKENS", context.get("reserve_tokens", 16384)),
        output + COMPACTION_MARGIN,
    )
    keep = _integer(env, "KEEP_RECENT_TOKENS", context.get("keep_recent_tokens", 20000))
    if output >= window or reserve >= window or (enabled and reserve + keep >= window):
        raise ValueError(
            "context window must exceed output/reserve and, with compaction, reserve + keep"
        )
    unknown = sorted(
        set(context)
        - {
            "enabled",
            "context_window_tokens",
            "reserve_tokens",
            "keep_recent_tokens",
            "summary_instructions",
            "summary_max_tokens",
        }
    )
    if unknown:
        warnings.warn(
            "OpenCode has no equivalent for context fields: " + ", ".join(unknown),
            UserWarning,
            stacklevel=2,
        )
    temperature = _sampling(env, "TEMPERATURE", 2)
    top_p = _sampling(env, "TOP_P", 1)
    sampling = {}
    if temperature is not None:
        sampling["temperature"] = temperature
    if top_p is not None:
        sampling["top_p"] = top_p
    request_timeout = _integer(env, "REQUEST_TIMEOUT", 1800)
    urls = {
        "openrouter": "https://openrouter.ai/api/v1",
        "parasail": "https://api.parasail.io/v1",
    }
    url = (
        env.get("SHADOW_OPENCODE_BASE_URL", "").strip()
        or urls.get(provider)
        or env.get("OPENAI_BASE_URL", "").strip()
    )
    if not url:
        raise ValueError(
            "openai/<id> requires OPENAI_BASE_URL or SHADOW_OPENCODE_BASE_URL"
        )
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "OpenCode base URL must be HTTP(S), without credentials, query, or fragment"
        )
    # OpenCode's built-in "openai" loader calls the Responses API even when
    # npm is overridden. A custom provider ID is required for Chat Completions.
    runtime_provider = "shadow-openai" if provider == "openai" else provider
    runtime_model = f"{runtime_provider}/{model_id}"
    hermes = hermes_tool_format(env)
    if hermes:
        system_prompt = system_prompt.rstrip() + HERMES_FORMAT_NOTE
    agent = {
        "mode": "primary",
        "model": runtime_model,
        "prompt": system_prompt,
        **sampling,
    }
    if max_steps is not None:
        agent["steps"] = positive_int("max_steps", max_steps)
    withheld = disabled_tools(env)
    if withheld:
        agent["tools"] = {name: False for name in withheld}
    summary = context.get("summary_instructions")
    if summary is not None and not isinstance(summary, str):
        raise ValueError("summary_instructions must be a string")
    compaction_agent = {"model": runtime_model, **sampling}
    if summary:
        compaction_agent["prompt"] = (
            "Summarize the conversation so another agent can continue the task. "
            "Do not execute the task or call tools.\n" + summary
        )
    key_name = {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "parasail": "PARASAIL_API_KEY",
    }[provider]
    result = {
        "$schema": "https://opencode.ai/config.json",
        "model": runtime_model,
        "small_model": runtime_model,
        "default_agent": "shadow",
        "autoupdate": False,
        "share": "disabled",
        "snapshot": False,
        "lsp": False,
        "formatter": False,
        "enabled_providers": [runtime_provider],
        "provider": {
            runtime_provider: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Shadow " + provider,
                "options": {
                    "baseURL": url,
                    "apiKey": "{env:" + key_name + "}",
                    "timeout": request_timeout * 1000,
                    "headerTimeout": request_timeout * 1000,
                    "chunkTimeout": request_timeout * 1000,
                },
                "models": {
                    model_id: {
                        "id": model_id,
                        "name": model_id,
                        "temperature": True,
                        "tool_call": True,
                        "attachment": True,
                        "modalities": {"input": ["text", "image"], "output": ["text"]},
                        # v1.18.34 ignores compaction.reserved when limit.input is
                        # absent. Supplying input=context makes usable=context-reserve.
                        "limit": {"context": window, "input": window, "output": output},
                    }
                },
            }
        },
        "compaction": {
            "auto": enabled,
            "prune": False,
            "reserved": reserve,
            "preserve_recent_tokens": keep,
        },
        "agent": {
            "shadow": agent,
            "compaction": compaction_agent,
            "title": {"disable": True},
            "summary": {"disable": True},
        },
        # No interactive permission prompts or unbudgeted subagents in batch runs.
        "permission": {
            "*": "allow",
            "question": "deny",
            "task": "deny",
            "plan_enter": "deny",
            "plan_exit": "deny",
        },
    }
    plugins = []
    if hermes and hermes_plugin_path is not None:
        plugins.append("file://" + str(hermes_plugin_path))
    bash_cap = max_bash_timeout_ms(env)
    if bash_cap is not None and tool_limits_plugin_path is not None:
        plugins.append(
            ["file://" + str(tool_limits_plugin_path), {"maxBashTimeoutMs": bash_cap}]
        )
    if plugins:
        result["plugin"] = plugins
    if summary_output is not None:
        # Native per-agent model selection gives the summarizer its own output
        # cap without changing the upstream model ID or normal-turn budget.
        summary_id = model_id + "--shadow-compaction"
        models = result["provider"][runtime_provider]["models"]
        models[summary_id] = deepcopy(models[model_id])
        models[summary_id]["limit"]["output"] = summary_output
        result["agent"]["compaction"]["model"] = f"{runtime_provider}/{summary_id}"
    return result


def runner_settings(env: dict | None = None) -> dict:
    env = os.environ if env is None else env
    raw = env.get("SHADOW_OPENCODE_LENGTH_CONTINUATIONS", "0").strip() or "0"
    continuations = int(raw)
    if continuations < 0:
        raise ValueError("SHADOW_OPENCODE_LENGTH_CONTINUATIONS must be nonnegative")
    raw = env.get("SHADOW_OPENCODE_TOOLCALL_RECOVERIES", "0").strip() or "0"
    recoveries = int(raw)
    if recoveries < 0:
        raise ValueError("SHADOW_OPENCODE_TOOLCALL_RECOVERIES must be nonnegative")
    request_timeout = _integer(env, "REQUEST_TIMEOUT", 1800)
    idle_timeout = _integer(env, "IDLE_TIMEOUT", request_timeout + 120)
    if idle_timeout <= request_timeout:
        raise ValueError(
            "SHADOW_OPENCODE_IDLE_TIMEOUT must exceed REQUEST_TIMEOUT for buffered model responses"
        )
    settings = {"idle_timeout": idle_timeout, "length_continuations": continuations}
    if recoveries:
        # Only present when enabled, so existing runs keep identical settings.
        settings["toolcall_recoveries"] = recoveries
    return settings
