#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

observer_processes="$(observer_pids)"
if test -n "$observer_processes"; then
    for pid in $observer_processes; do
        kill "$pid"
    done
    deadline=$((SECONDS + 30))
    for pid in $observer_processes; do
        while kill -0 "$pid" 2>/dev/null; do
            if (( SECONDS >= deadline )); then
                echo "VM PID $pid did not stop after 30 seconds." >&2
                exit 1
            fi
            sleep 1
        done
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
