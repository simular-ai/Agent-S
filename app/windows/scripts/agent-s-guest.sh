#!/bin/bash
set -euo pipefail
umask 077

die() { printf '%s\n' "$1" >&2; exit 2; }

# Only this executable and the five bounded desktop actions are available over SSH.
if [[ ${SSH_ORIGINAL_COMMAND+x} ]]; then
    [[ $SSH_ORIGINAL_COMMAND != *$'\n'* && $SSH_ORIGINAL_COMMAND != *$'\r'* ]] ||
        die "Invalid desktop command."
    read -r -a request <<<"$SSH_ORIGINAL_COMMAND"
    [[ ${request[0]:-} == /usr/local/bin/agent-s-guest ]] || die "Invalid desktop command."
    set -- "${request[@]:1}"
fi

# WSL guests use a non-default display (WSLg owns :0); /etc/agent-s/env is root-owned and optional.
if [[ -r /etc/agent-s/env ]]; then . /etc/agent-s/env; fi
export DISPLAY=":${AGENT_S_DISPLAY:-0}"
unset WAYLAND_DISPLAY
export XAUTHORITY="$HOME/.local/state/agent-s/Xauthority"
state="$HOME/.local/state/agent-s"
demo="file:///usr/local/share/agent-s/demo.html"

if [[ -r "$state/dbus-address" ]]; then
    IFS= read -r DBUS_SESSION_BUS_ADDRESS < "$state/dbus-address"
    export DBUS_SESSION_BUS_ADDRESS
fi

decode() {
    local LC_ALL=C
    [[ $1 =~ ^[A-Za-z0-9+/]+=*$ && ${#1} -le 5464 ]] ||
        die "Invalid desktop action payload."
    # The sentinel preserves trailing newlines. Round-trip validation rejects embedded NUL.
    decoded=$(printf '%s' "$1" | base64 -d | iconv -f UTF-8 -t UTF-8 && printf '.') ||
        die "Invalid desktop action payload."
    decoded=${decoded%.}
    [[ ${#decoded} -le 4096 && $(printf '%s' "$decoded" | base64 -w0) == "$1" ]] ||
        die "Invalid desktop action payload."
}

action=${1:-}
case "$action" in
    screenshot)
        [[ $# == 1 ]] || die "Screenshot takes no arguments."
        temp=$(mktemp -d)
        trap 'rm -rf -- "$temp"' EXIT
        timeout 8s scrot -z "$temp/desktop.png"
        size=$(stat -c %s "$temp/desktop.png")
        (( size > 0 && size <= 3145728 )) || die "Guest screenshot exceeds the size limit."
        base64 -w0 "$temp/desktop.png"
        printf '\n'
        ;;
    open)
        [[ $# == 2 ]] || die "Open takes one URL."
        decode "$2"
        if [[ $decoded == demo ]]; then decoded=$demo; fi
        [[ $decoded == "$demo" || $decoded == about:blank ||
           $decoded =~ ^https?://[^[:space:]/@?#]+([/?#][^[:space:]]*)?$ ]] ||
            die "Use the offline demo or an HTTP(S) URL."
        setsid --fork /usr/bin/chromium --user-data-dir="$state/chromium" \
            --disable-background-networking --disable-component-update --disable-sync \
            --no-first-run --no-default-browser-check -- "$decoded" </dev/null >/dev/null 2>&1
        # Chromium handles navigation in its existing process after this command forks.
        sleep 0.35
        ;;
    click)
        [[ $# == 3 && $2 =~ ^-?[0-9]{1,6}$ && $3 =~ ^-?[0-9]{1,6}$ ]] ||
            die "Click coordinates must be integers."
        x=$(( 10#${2#-} )); y=$(( 10#${3#-} ))
        [[ $2 != -* ]] || x=0
        [[ $3 != -* ]] || y=0
        if (( x > 1023 )); then x=1023; fi
        if (( y > 767 )); then y=767; fi
        xdotool mousemove "$x" "$y" click 1
        ;;
    type)
        [[ $# == 2 ]] || die "Type takes one text value."
        decode "$2"
        # Paste through the guest's private X clipboard to support Unicode and multiline input.
        printf '%s' "$decoded" | xclip -selection clipboard -in >/dev/null 2>&1
        xdotool key --clearmodifiers Control_L+v
        ;;
    key)
        [[ $# == 2 ]] || die "Key takes one chord."
        decode "$2"
        case "$decoded" in
            Return|Tab|Shift_L+Tab|Escape|BackSpace|Delete|Left|Right|Up|Down|Home|End|Page_Up|Page_Down|Control_L+[aclfvz]|Alt_L+Left|Alt_L+Right) ;;
            *) die "Unsupported desktop key." ;;
        esac
        xdotool key --clearmodifiers "$decoded"
        ;;
    scroll)
        [[ $# == 2 ]] || die "Scroll takes one direction."
        decode "$2"
        case "$decoded" in
            up|-1) button=4 ;;
            down|1|+1) button=5 ;;
            *) die "Scroll direction must be up or down." ;;
        esac
        xdotool click --repeat 3 --delay 70 "$button"
        ;;
    *)
        die "Unsupported desktop action."
        ;;
esac
