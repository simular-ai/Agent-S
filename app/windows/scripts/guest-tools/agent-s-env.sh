# Sourced by web, ui and harness turns: reach the Private computer's desktop from outside its session.
# WSLg owns :0, so Agent S's desktop uses the display in /etc/agent-s/env (normally :99).
if [ -r /etc/agent-s/env ]; then . /etc/agent-s/env; fi
export DISPLAY=":${AGENT_S_DISPLAY:-0}"
unset WAYLAND_DISPLAY
export XAUTHORITY="$HOME/.local/state/agent-s/Xauthority"
if [ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ] && [ -r "$HOME/.local/state/agent-s/dbus-address" ]; then
  IFS= read -r DBUS_SESSION_BUS_ADDRESS < "$HOME/.local/state/agent-s/dbus-address"
  export DBUS_SESSION_BUS_ADDRESS
fi
