#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

deep=false
case "${1:-}" in
    "") ;;
    --deep) deep=true ;;
    *) echo "Usage: $0 [--deep]" >&2; exit 2 ;;
esac

if test -f "$BASE_IMAGE"; then
    echo "base_image=ready"
else
    echo "base_image=missing"
fi
if pid_is_running; then
    echo "vm=running"
    echo "pid=$(<"$PID_FILE")"
    echo "ssh=127.0.0.1:$SSH_PORT"
    echo "vnc=127.0.0.1:$VNC_PORT"
else
    echo "vm=stopped"
fi

ssh_reachable=false
if pid_is_running; then
    if timeout 3 bash -c "</dev/tcp/127.0.0.1/$SSH_PORT" 2>/dev/null; then
        ssh_reachable=true
        echo "ssh_transport=reachable"
    else
        echo "ssh_transport=unreachable_or_blocked"
    fi
else
    echo "ssh_transport=not_checked"
fi

endpoint_configured=false
if test -s "$ENDPOINT_ENV" && grep -q '^HF_ENDPOINT_URL=https://' "$ENDPOINT_ENV"; then
    endpoint_configured=true
    echo "hf_endpoint=configured"
else
    echo "hf_endpoint=missing"
fi

if ! pid_is_running || ! $ssh_reachable; then
    echo "integration=guest_unreachable"
elif ! $endpoint_configured; then
    echo "integration=model_unconfigured"
else
    echo "integration=transport_ready"
fi

if $deep; then
    echo "mcp_canary=running"
    if python3 "$SCRIPT_DIR/smoke_test.py" --health-only; then
        echo "mcp_canary=pass"
    else
        echo "mcp_canary=fail"
        exit 1
    fi
fi
