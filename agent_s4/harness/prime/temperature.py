"""Opt-in per-request sampling temperature for Prime Agent.

Prime has no temperature setting: it calls the provider without one, and the
OpenAI-completions client only sends ``temperature`` when it is given. Without
this, every Builder and evaluator request runs at whatever default the upstream
server applies. A ``before_provider_request`` extension fills it in instead;
the shared proxy stays untouched, so other runs keep the server default.
"""

from __future__ import annotations

import json
import os

TEMPERATURE_ENV = "SHADOW_PRIME_AGENT_TEMPERATURE"
EXTENSION_FILENAME = "sampling_temperature.ts"


def sampling_temperature() -> float | None:
    """Return the opted-in temperature; unset keeps the server default."""
    raw = os.environ.get(TEMPERATURE_ENV, "").strip()
    if not raw:
        return None
    value = float(raw)
    if not 0 <= value <= 2:
        raise ValueError(f"{TEMPERATURE_ENV} must be in [0, 2], got {raw!r}")
    return value


def temperature_extension_source(temperature: float) -> str:
    # The hook may replace the serialized payload. A payload that already
    # carries a temperature is left alone.
    return (
        "export default function (pi: any) {\n"
        '  pi.on("before_provider_request", (event: any) => {\n'
        "    const payload = event.payload;\n"
        '    if (payload && typeof payload === "object" && payload.temperature === undefined) {\n'
        f"      return {{ ...payload, temperature: {json.dumps(temperature)} }};\n"
        "    }\n"
        "  });\n"
        "}\n"
    )
