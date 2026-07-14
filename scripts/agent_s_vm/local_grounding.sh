#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

DOC2DB_ROOT="${DOC2DB_ROOT:-$HOME/code/Doc2DB}"
DOC2DB_ENV="$DOC2DB_ROOT/config/local_ai.env"
REMOTE_SESSION="local-ai-agent-s-ui-tars"
TUNNEL_SESSION="agent-s-ui-tars-tunnel"
REMOTE_PORT="${AGENT_S_REMOTE_GROUND_PORT:-8082}"
LOCAL_PORT="${AGENT_S_LOCAL_GROUND_PORT:-18082}"

load_route() {
    if ! test -f "$DOC2DB_ENV"; then
        echo "Private route config is missing: $DOC2DB_ENV" >&2
        exit 2
    fi
    set -a
    # shellcheck source=/dev/null
    source "$DOC2DB_ENV"
    set +a
    SSH_USER="${DOC2DB_REMOTE_LLM_SSH_USER:-${DOC2DB_MAC_LLM_SSH_USER:-}}"
    SSH_HOST="${DOC2DB_REMOTE_LLM_SSH_HOST:-${DOC2DB_MAC_LLM_SSH_HOST:-}}"
    SSH_REMOTE_PORT="${DOC2DB_REMOTE_LLM_SSH_PORT:-${DOC2DB_MAC_LLM_SSH_PORT:-22}}"
    if test -z "$SSH_USER" || test -z "$SSH_HOST"; then
        echo "Private model-host SSH route is not configured." >&2
        exit 2
    fi
    SSH_TARGET="$SSH_USER@$SSH_HOST"
    SSH=(ssh -q -p "$SSH_REMOTE_PORT" -o BatchMode=yes -o ConnectTimeout=10)
    SCP=(scp -q -P "$SSH_REMOTE_PORT" -o BatchMode=yes -o ConnectTimeout=10)
}

start_remote() {
    "${SCP[@]}" "$SCRIPT_DIR/mac_ui_tars_server.py" \
        "$SSH_TARGET:code/local-ai-lab/scripts/agent_s_ui_tars_server.py"
    "${SSH[@]}" "$SSH_TARGET" bash -s -- "$REMOTE_SESSION" "$REMOTE_PORT" <<'REMOTE'
set -euo pipefail
session="$1"
port="$2"
model_id="ByteDance-Seed/UI-TARS-1.5-7B"
python="$HOME/code/local-ai-lab/.venv-ui-tars-transformers-py312/bin/python"
server="$HOME/code/local-ai-lab/scripts/agent_s_ui_tars_server.py"
log_dir="$HOME/Runs/local-ai-lab/agent-s-ui-tars"
log="$log_dir/server.log"
test -x "$python"
test -f "$server"
mkdir -p "$log_dir"
if tmux has-session -t "$session" 2>/dev/null; then
    echo remote_server=already_running
    exit 0
fi
tmux new-session -d -s "$session" \
  "/usr/bin/caffeinate -dimsu '$python' '$server' --host 127.0.0.1 --port '$port' --model '$model_id' >>'$log' 2>&1"
echo remote_server=started
REMOTE
}

start_tunnel() {
    if tmux has-session -t "$TUNNEL_SESSION" 2>/dev/null; then
        echo tunnel=already_running
        return
    fi
    tmux new-session -d -s "$TUNNEL_SESSION" \
        "ssh -q -N -p '$SSH_REMOTE_PORT' -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -L '127.0.0.1:$LOCAL_PORT:127.0.0.1:$REMOTE_PORT' '$SSH_TARGET'"
    echo tunnel=started
}

status() {
    if tmux has-session -t "$TUNNEL_SESSION" 2>/dev/null; then
        echo tunnel=running
    else
        echo tunnel=stopped
    fi
    if curl --fail --silent --max-time 3 "http://127.0.0.1:$LOCAL_PORT/health" >/dev/null 2>&1; then
        echo local_endpoint=ready
    else
        echo local_endpoint=not_ready
    fi
    "${SSH[@]}" "$SSH_TARGET" bash -s -- "$REMOTE_SESSION" "$REMOTE_PORT" <<'REMOTE'
session="$1"
port="$2"
if tmux has-session -t "$session" 2>/dev/null; then echo remote_server=running; else echo remote_server=stopped; fi
if curl --fail --silent --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then echo remote_endpoint=ready; else echo remote_endpoint=not_ready; fi
REMOTE
}

stop_all() {
    tmux kill-session -t "$TUNNEL_SESSION" 2>/dev/null || true
    "${SSH[@]}" "$SSH_TARGET" "tmux kill-session -t '$REMOTE_SESSION' 2>/dev/null || true"
    echo local_grounding=stopped
}

case "${1:-status}" in
    start)
        load_route
        start_remote
        start_tunnel
        ;;
    status)
        load_route
        status
        ;;
    stop)
        load_route
        stop_all
        ;;
    restart)
        load_route
        stop_all
        start_remote
        start_tunnel
        ;;
    *)
        echo "usage: $0 {start|status|stop|restart}" >&2
        exit 2
        ;;
esac
