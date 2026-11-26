#!/bin/bash
# Starts the dock (tint2) on the Private computer's display if it isn't running. Its launcher opens
# Chromium, the browser this guest has (tint2's default config lists browsers it doesn't).
set -uo pipefail

if [[ -r /etc/agent-s/env ]]; then . /etc/agent-s/env; fi
export DISPLAY=":${AGENT_S_DISPLAY:-0}"
export XAUTHORITY="$HOME/.local/state/agent-s/Xauthority"
unset WAYLAND_DISPLAY

marker='# Agent S dock'
rc="$HOME/.config/tint2/tint2rc"
changed=false
if [[ -r /etc/xdg/tint2/tint2rc ]] && ! grep -qxF "$marker" "$rc" 2>/dev/null; then
    mkdir -p -- "${rc%/*}"
    # tint2 writes its default config on first start; keep any earlier one beside ours.
    [[ -e $rc ]] && cp -f -- "$rc" "$rc.bak"
    { printf '%s\n' "$marker"; sed -e '/^launcher_item_app/d' /etc/xdg/tint2/tint2rc
      printf '%s\n' 'launcher_item_app = /usr/share/applications/chromium.desktop'; } >"$rc.new"
    mv -f -- "$rc.new" "$rc"
    changed=true
fi

if pgrep -u "$(id -u)" -x tint2 >/dev/null; then
    $changed && pkill -USR1 -u "$(id -u)" -x tint2
else
    setsid --fork tint2 </dev/null >/dev/null 2>&1
fi
