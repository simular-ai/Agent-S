#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

for command in qemu-img qemu-system-x86_64; do
    require_command "$command"
done
if ! test -f "$BASE_IMAGE"; then
    echo "Base image is missing. Run $SCRIPT_DIR/build_base.sh first." >&2
    exit 1
fi
if pid_is_running; then
    echo "Agent S VM is already running with PID $(<"$PID_FILE")."
    exit 0
fi
if test -e "$PID_FILE"; then
    rm -f "$PID_FILE"
fi

overlay="$RUNTIME_DIR/overlay-$(date -u +%Y%m%dT%H%M%SZ).qcow2"
qemu-img create -q -f qcow2 -F qcow2 -b "$BASE_IMAGE" "$overlay"
printf '%s\n' "$overlay" >"$RUNTIME_PATH_FILE"

qemu-system-x86_64 \
    -name agent-s-observer \
    -machine accel=kvm \
    -cpu host \
    -smp 8 \
    -m 16384 \
    -device virtio-vga,xres=1920,yres=1080 \
    -display "vnc=127.0.0.1:$VNC_DISPLAY" \
    -drive "if=virtio,format=qcow2,file=$overlay" \
    -netdev "user,id=net0,hostfwd=tcp:127.0.0.1:$SSH_PORT-:22" \
    -device virtio-net-pci,netdev=net0 \
    -daemonize \
    -pidfile "$PID_FILE" \
    -D "$QEMU_LOG"

deadline=$((SECONDS + 180))
until timeout 1 bash -c "</dev/tcp/127.0.0.1/$SSH_PORT" 2>/dev/null; do
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for VM SSH. See $QEMU_LOG" >&2
        exit 1
    fi
    sleep 3
done

echo "Agent S observation VM is running."
echo "  PID: $(<"$PID_FILE")"
echo "  SSH bridge: 127.0.0.1:$SSH_PORT"
echo "  VNC viewer: 127.0.0.1:$VNC_PORT"
