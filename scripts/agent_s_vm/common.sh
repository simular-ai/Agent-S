#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LAB_HOME="${AGENT_S_LAB_HOME:-$HOME/.local/share/agent-s-lab}"
CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/agent-s-lab"
DOWNLOAD_DIR="$LAB_HOME/downloads"
BASE_IMAGE="$LAB_HOME/agent-s-base.qcow2"
RUNTIME_DIR="$LAB_HOME/runtime"
RUNTIME_PATH_FILE="$RUNTIME_DIR/current-overlay"
PID_FILE="$RUNTIME_DIR/qemu.pid"
QEMU_LOG="$RUNTIME_DIR/qemu.log"
SSH_KEY="$CONFIG_HOME/observer_mcp_ed25519"
KNOWN_HOSTS="$CONFIG_HOME/known_hosts"
ENDPOINT_ENV="$CONFIG_HOME/endpoint.env"
SSH_PORT="${AGENT_S_SSH_PORT:-2202}"
VNC_DISPLAY="${AGENT_S_VNC_DISPLAY:-5}"
VNC_PORT="$((5900 + VNC_DISPLAY))"

mkdir -p "$LAB_HOME" "$CONFIG_HOME" "$DOWNLOAD_DIR" "$RUNTIME_DIR"
chmod 700 "$CONFIG_HOME" "$RUNTIME_DIR"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command is missing: $1" >&2
        exit 1
    fi
}

pid_is_running() {
    test -s "$PID_FILE" && kill -0 "$(<"$PID_FILE")" 2>/dev/null
}
