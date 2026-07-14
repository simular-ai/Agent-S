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
    endpoint_line="$(grep -m1 '^HF_ENDPOINT_URL=https://' "$ENDPOINT_ENV" || true)"
    if test -n "$endpoint_line"; then
        HF_ENDPOINT_URL="${endpoint_line#HF_ENDPOINT_URL=}"
        export HF_ENDPOINT_URL
    fi
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
    -o SendEnv=AGENT_S_MAIN_MODEL \
    -o SendEnv=AGENT_S_GROUND_MODEL \
    -o StrictHostKeyChecking=yes \
    -o "UserKnownHostsFile=$KNOWN_HOSTS" \
    agent-s@127.0.0.1 /opt/agent-s/venv/bin/agent_s_observer_mcp
