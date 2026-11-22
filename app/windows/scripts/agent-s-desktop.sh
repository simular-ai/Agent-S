#!/bin/bash
set -euo pipefail
umask 077

state="$HOME/.local/state/agent-s"
mkdir -p -- "$state"

if [[ -r /etc/agent-s/env ]]; then . /etc/agent-s/env; fi
unset WAYLAND_DISPLAY

if [[ ${1:-} != session ]]; then
    exec dbus-run-session -- xvfb-run --server-num="${AGENT_S_DISPLAY:-0}" \
        --server-args="-screen 0 1024x768x24 -nolisten tcp ${AGENT_S_XVFB_EXTRA:-}" \
        --auth-file="$state/Xauthority" /usr/local/bin/agent-s-desktop session
fi

printf '%s\n' "$DBUS_SESSION_BUS_ADDRESS" >"$state/dbus-address"
xsetroot -solid '#253245'
openbox-session &
window_manager=$!
/usr/local/bin/agent-s-guest open ZGVtbw==
wait "$window_manager"
