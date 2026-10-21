#!/usr/bin/env bash
# Start the LiteLLM proxy that gives a vLLM deployment tool calling.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${LITELLM_PORT:-4000}"
# Upstream credentials come from .env under their own names, so the proxy's own
# OPENAI_* (which point at the proxy itself) can never shadow them.
export HERMES_UPSTREAM_BASE_URL="${HERMES_UPSTREAM_BASE_URL:-$("$ROOT/.venv/bin/dotenv" -f "$ROOT/.env" get PARASAIL_BASE_URL 2>/dev/null || echo https://api.parasail.io/v1)}"
export HERMES_UPSTREAM_API_KEY="${HERMES_UPSTREAM_API_KEY:-$("$ROOT/.venv/bin/dotenv" -f "$ROOT/.env" get PARASAIL_API_KEY)}"
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:?set LITELLM_MASTER_KEY}"
HERMES_UPSTREAM_MODEL="${HERMES_UPSTREAM_MODEL:-$("$ROOT/.venv/bin/dotenv" -f "$ROOT/.env" get HERMES_UPSTREAM_MODEL 2>/dev/null || true)}"
if [[ ! "$HERMES_UPSTREAM_MODEL" =~ ^[A-Za-z0-9._/-]+$ ]]; then
    printf 'ERROR: set HERMES_UPSTREAM_MODEL to the upstream model name\n' >&2
    exit 2
fi
CONFIG="$(mktemp "${TMPDIR:-/tmp}/litellm_config.XXXXXX.yaml")"
sed "s|__HERMES_UPSTREAM_MODEL__|$HERMES_UPSTREAM_MODEL|g" \
    "$ROOT/scripts/proxy/litellm_config.yaml" > "$CONFIG"
# custom_handler is resolved as a module path, so its directory must be importable.
export PYTHONPATH="$ROOT/scripts/proxy${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.venv-litellm/bin/litellm" \
    --config "$CONFIG" \
    --port "$PORT" --num_workers "${LITELLM_WORKERS:-1}"
