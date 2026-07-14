#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

if pid_is_running; then
    pid="$(<"$PID_FILE")"
    kill "$pid"
    deadline=$((SECONDS + 30))
    while kill -0 "$pid" 2>/dev/null; do
        if (( SECONDS >= deadline )); then
            echo "VM did not stop after 30 seconds." >&2
            exit 1
        fi
        sleep 1
    done
fi

if test -s "$RUNTIME_PATH_FILE"; then
    overlay="$(<"$RUNTIME_PATH_FILE")"
    case "$overlay" in
        "$RUNTIME_DIR"/overlay-*.qcow2) rm -f "$overlay" ;;
        *) echo "Refusing to remove unexpected overlay path: $overlay" >&2; exit 1 ;;
    esac
fi
rm -f "$PID_FILE" "$RUNTIME_PATH_FILE"
echo "Agent S VM stopped and its disposable overlay was removed."
