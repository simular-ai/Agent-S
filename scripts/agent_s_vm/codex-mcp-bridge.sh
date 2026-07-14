#!/usr/bin/env bash
set -euo pipefail
# The bridge loads credentials below.  Keep xtrace disabled even when a caller
# invokes this script through `bash -x`.
set +x
source "$(dirname "$0")/common.sh"

if ! test -f "$SSH_KEY"; then
    echo "Observer SSH key is missing: $SSH_KEY" >&2
    exit 1
fi
if test -z "${HF_TOKEN:-}" && test -s "$HOME/.cache/huggingface/token"; then
    HF_TOKEN="$(<"$HOME/.cache/huggingface/token")"
    export HF_TOKEN
fi
if test -s "$ENDPOINT_ENV"; then
    main_url_line="$(grep -m1 -E '^AGENT_S_MAIN_BASE_URL=(https://|http://10[.]0[.]2[.]2:18082/v1/?$)' "$ENDPOINT_ENV" || true)"
    if test -n "$main_url_line"; then
        AGENT_S_MAIN_BASE_URL="${main_url_line#AGENT_S_MAIN_BASE_URL=}"
        export AGENT_S_MAIN_BASE_URL
    fi
    ground_url_line="$(grep -m1 -E '^AGENT_S_GROUND_BASE_URL=(https://|http://10[.]0[.]2[.]2:18082/v1/?$)' "$ENDPOINT_ENV" || true)"
    if test -n "$ground_url_line"; then
        AGENT_S_GROUND_BASE_URL="${ground_url_line#AGENT_S_GROUND_BASE_URL=}"
        export AGENT_S_GROUND_BASE_URL
    fi
    main_model_line="$(grep -m1 -E '^AGENT_S_MAIN_MODEL=[A-Za-z0-9._/-]+$' "$ENDPOINT_ENV" || true)"
    if test -n "$main_model_line"; then
        AGENT_S_MAIN_MODEL="${main_model_line#AGENT_S_MAIN_MODEL=}"
        export AGENT_S_MAIN_MODEL
    fi
    ground_model_line="$(grep -m1 -E '^AGENT_S_GROUND_MODEL=[A-Za-z0-9._/-]+$' "$ENDPOINT_ENV" || true)"
    if test -n "$ground_model_line"; then
        AGENT_S_GROUND_MODEL="${ground_model_line#AGENT_S_GROUND_MODEL=}"
        export AGENT_S_GROUND_MODEL
    fi
    endpoint_line="$(grep -m1 '^HF_ENDPOINT_URL=https://' "$ENDPOINT_ENV" || true)"
    if test -n "$endpoint_line"; then
        HF_ENDPOINT_URL="${endpoint_line#HF_ENDPOINT_URL=}"
        export HF_ENDPOINT_URL
    fi
fi
if [[ "${AGENT_S_MAIN_BASE_URL:-}" == "http://10.0.2.2:18082/v1" || "${AGENT_S_MAIN_BASE_URL:-}" == "http://10.0.2.2:18082/v1/" ]]; then
    export AGENT_S_MAIN_API_KEY="${AGENT_S_MAIN_API_KEY:-local-observer}"
fi
if [[ "${AGENT_S_GROUND_BASE_URL:-}" == "http://10.0.2.2:18082/v1" || "${AGENT_S_GROUND_BASE_URL:-}" == "http://10.0.2.2:18082/v1/" ]]; then
    export AGENT_S_GROUND_API_KEY="${AGENT_S_GROUND_API_KEY:-local-observer}"
    unset HF_TOKEN HF_ENDPOINT_URL
fi
export AGENT_S_MAIN_MODEL="${AGENT_S_MAIN_MODEL:-gpt-5-2025-08-07}"
export AGENT_S_GROUND_MODEL="${AGENT_S_GROUND_MODEL:-tgi}"

exec ssh -q -T \
    -i "$SSH_KEY" \
    -p "$SSH_PORT" \
    -o BatchMode=yes \
    -o ConnectionAttempts=1 \
    -o ConnectTimeout=10 \
    -o LogLevel=ERROR \
    -o ServerAliveCountMax=2 \
    -o ServerAliveInterval=5 \
    -o SendEnv=OPENAI_API_KEY \
    -o SendEnv=HF_TOKEN \
    -o SendEnv=HF_ENDPOINT_URL \
    -o SendEnv=AGENT_S_MAIN_API_KEY \
    -o SendEnv=AGENT_S_MAIN_BASE_URL \
    -o SendEnv=AGENT_S_GROUND_API_KEY \
    -o SendEnv=AGENT_S_GROUND_BASE_URL \
    -o SendEnv=AGENT_S_MAIN_MODEL \
    -o SendEnv=AGENT_S_GROUND_MODEL \
    -o StrictHostKeyChecking=yes \
    -o "UserKnownHostsFile=$KNOWN_HOSTS" \
    agent-s@127.0.0.1 /opt/agent-s/venv/bin/agent_s_observer_mcp
