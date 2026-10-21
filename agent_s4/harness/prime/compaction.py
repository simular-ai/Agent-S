"""Translate interface policy into settings the pinned Prime runtime consumes."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agent_s4.harness.prime.context_window import interface_window_is_honoured

logger = logging.getLogger(__name__)
# Prime checks last successful total usage plus estimated trailing messages.
# Leave headroom for that estimate and provider serialization. This is not a
# bound on arbitrary tool output; recognizable overflow errors remain necessary.
COMPACTION_MARGIN_TOKENS = 4096


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_compaction_config(shadow_dir: str | Path | None) -> dict:
    if shadow_dir is None:
        return {}
    path = Path(shadow_dir) / "context_compaction.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load Prime context config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{path}: Prime context config must be an object")
    return config


def context_window_from_config(config: dict) -> int | None:
    # Preserve the historical window-source contract, independently of how
    # the now-connected enabled/reserve/keep settings are applied.
    if not interface_window_is_honoured() or config.get("enabled", True) is False:
        return None
    value = config.get("context_window_tokens")
    return None if value is None else _positive_integer("context_window_tokens", value)


def compaction_settings(
    config: dict | None = None,
    *,
    context_window: int | None = None,
    max_output_tokens: int | None = None,
) -> dict:
    """Map supported fields, with an output reserve for a known request cap.

    ``max_output_tokens`` is the provider's actual normal-turn request budget,
    not an estimate of observed output. The pinned OpenAI-completions path
    requests at most 32000 by default, including on OpenRouter. Other provider
    paths can adjust the budget for thinking and retain their current defaults.

    Prime has no settings for min_compact_tokens, summary_max_tokens or
    summary_instructions. In particular its summary request budget is
    floor(0.8 * reserveTokens), and a split-turn summary additionally requests
    floor(0.5 * reserveTokens). Increasing reserve also increases these budgets.
    """
    config = {} if config is None else config
    if not isinstance(config, dict):
        raise ValueError("Prime context config must be an object")
    enabled = config.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    reserve = _positive_integer("reserve_tokens", config.get("reserve_tokens", 16384))
    keep = _positive_integer(
        "keep_recent_tokens", config.get("keep_recent_tokens", 20000)
    )
    unsupported = sorted(
        set(config)
        - {
            "enabled",
            "context_window_tokens",
            "reserve_tokens",
            "keep_recent_tokens",
        }
    )
    if unsupported:
        logger.warning(
            "Prime does not support these context_compaction fields; ignored: %s. "
            "Summary budgets and instructions remain owned by the pinned runtime.",
            ", ".join(unsupported),
        )
    if max_output_tokens is not None:
        max_output_tokens = _positive_integer("max_output_tokens", max_output_tokens)
        minimum_reserve = max_output_tokens + COMPACTION_MARGIN_TOKENS
        if reserve < minimum_reserve:
            logger.info(
                "Prime reserve_tokens raised from %s to %s for max output %s plus margin %s",
                reserve,
                minimum_reserve,
                max_output_tokens,
                COMPACTION_MARGIN_TOKENS,
            )
            reserve = minimum_reserve
    if context_window is not None:
        context_window = _positive_integer("context_window_tokens", context_window)
        if max_output_tokens is not None and max_output_tokens >= context_window:
            raise ValueError(
                "max_output_tokens must be smaller than context_window_tokens"
            )
        if enabled and reserve + keep >= context_window:
            raise ValueError(
                "effective reserve_tokens + keep_recent_tokens must be smaller "
                "than context_window_tokens"
            )
    return {"enabled": enabled, "reserveTokens": reserve, "keepRecentTokens": keep}
