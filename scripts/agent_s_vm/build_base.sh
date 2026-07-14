#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

for command in curl genisoimage gpgv qemu-img qemu-system-x86_64 scp ssh ssh-keygen tar; do
    require_command "$command"
done

if test -e "$BASE_IMAGE"; then
    echo "Base image already exists: $BASE_IMAGE" >&2
    echo "Move it aside explicitly before rebuilding." >&2
    exit 1
fi
if observer_is_running; then
    echo "Runtime VM is active; stop it before building." >&2
    exit 1
fi

CLOUD_BASE_URL="https://cloud-images.ubuntu.com/jammy/current"
CLOUD_IMAGE="$DOWNLOAD_DIR/jammy-server-cloudimg-amd64.img"
CHECKSUMS="$DOWNLOAD_DIR/SHA256SUMS"
CHECKSUM_SIGNATURE="$DOWNLOAD_DIR/SHA256SUMS.gpg"
KEYRING="/usr/share/keyrings/ubuntu-cloudimage-keyring.gpg"

if ! test -f "$KEYRING"; then
    echo "Ubuntu cloud image keyring is missing: $KEYRING" >&2
    exit 1
fi

echo "Downloading and verifying the Ubuntu Jammy cloud image..."
curl --fail --location --proto '=https' --tlsv1.2 \
    --output "$CHECKSUMS" "$CLOUD_BASE_URL/SHA256SUMS"
curl --fail --location --proto '=https' --tlsv1.2 \
    --output "$CHECKSUM_SIGNATURE" "$CLOUD_BASE_URL/SHA256SUMS.gpg"
gpgv --keyring "$KEYRING" "$CHECKSUM_SIGNATURE" "$CHECKSUMS"
verify_cloud_image() {
    (
        cd "$DOWNLOAD_DIR"
        grep ' [*]jammy-server-cloudimg-amd64.img$' SHA256SUMS | sha256sum --check
    )
}
if ! test -f "$CLOUD_IMAGE" || ! verify_cloud_image; then
    curl --fail --location --proto '=https' --tlsv1.2 \
        --output "$CLOUD_IMAGE" "$CLOUD_BASE_URL/jammy-server-cloudimg-amd64.img"
    verify_cloud_image
fi

if ! test -f "$SSH_KEY"; then
    ssh-keygen -q -t ed25519 -N '' -C 'agent-s-observer-mcp' -f "$SSH_KEY"
    chmod 600 "$SSH_KEY"
fi
PUBLIC_KEY="$(<"$SSH_KEY.pub")"
if test -f "$KNOWN_HOSTS"; then
    ssh-keygen -q -f "$KNOWN_HOSTS" -R "[127.0.0.1]:$SSH_PORT" >/dev/null || true
fi

WORK_DIR="$(mktemp -d "$LAB_HOME/build.XXXXXX")"
BUILD_PID_FILE="$WORK_DIR/qemu.pid"
cleanup() {
    status=$?
    trap - EXIT
    if test -s "$BUILD_PID_FILE" && kill -0 "$(<"$BUILD_PID_FILE")" 2>/dev/null; then
        kill "$(<"$BUILD_PID_FILE")" 2>/dev/null || true
    fi
    if (( status != 0 )); then
        if test -f "$WORK_DIR/qemu.log"; then
            cp "$WORK_DIR/qemu.log" "$LAB_HOME/last-build-qemu.log"
            echo "Builder log preserved at $LAB_HOME/last-build-qemu.log" >&2
        fi
        rm -f "$BASE_IMAGE"
        echo "Removed incomplete base-image candidate: $BASE_IMAGE" >&2
    else
        rm -f "$LAB_HOME/last-build-qemu.log"
    fi
    rm -rf "$WORK_DIR"
    exit "$status"
}
trap cleanup EXIT

cat >"$WORK_DIR/meta-data" <<'EOF'
instance-id: agent-s-observer-base-v1
local-hostname: agent-s-observer
EOF

cat >"$WORK_DIR/user-data" <<EOF
#cloud-config
users:
  - name: agent-s
    gecos: Agent S Observer
    shell: /bin/bash
    lock_passwd: true
    groups: [adm, sudo]
    sudo: ["ALL=(ALL) NOPASSWD:ALL"]
    ssh_authorized_keys:
      - $PUBLIC_KEY
ssh_pwauth: false
package_update: true
packages:
  - dbus-x11
  - firefox
  - gdm3
  - gnome-session
  - gnome-shell
  - gnome-terminal
  - nftables
  - openssh-server
  - policykit-1
  - python3-pip
  - python3-venv
  - x11-xserver-utils
  - xauth
  - xorg
  - xserver-xorg-video-qxl
write_files:
  - path: /etc/gdm3/custom.conf
    permissions: '0644'
    content: |
      [daemon]
      AutomaticLoginEnable=true
      AutomaticLogin=agent-s
      WaylandEnable=false
  - path: /home/agent-s/.xprofile
    permissions: '0755'
    content: |
      #!/bin/sh
      xrandr --fb 1920x1080 || true
      xset s off || true
      xset -dpms || true
  - path: /home/agent-s/.config/autostart/agent-s-test-page.desktop
    permissions: '0644'
    content: |
      [Desktop Entry]
      Type=Application
      Name=Agent S observer safety page
      Exec=firefox --kiosk file:///home/agent-s/agent-s-observer-test.html
      X-GNOME-Autostart-enabled=true
runcmd:
  - systemctl enable ssh
  - systemctl set-default graphical.target
  - chown -R agent-s:agent-s /home/agent-s
power_state:
  mode: reboot
  timeout: 1800
  condition: true
EOF

genisoimage -quiet -output "$WORK_DIR/seed.iso" -volid cidata -joliet -rock \
    "$WORK_DIR/user-data" "$WORK_DIR/meta-data"
qemu-img create -q -f qcow2 -F qcow2 -b "$CLOUD_IMAGE" "$BASE_IMAGE" 60G

echo "Booting the builder VM; desktop package installation can take several minutes..."
qemu-system-x86_64 \
    -name agent-s-base-builder \
    -machine accel=kvm \
    -cpu host \
    -smp 8 \
    -m 16384 \
    -device virtio-vga,xres=1920,yres=1080 \
    -display "vnc=127.0.0.1:$VNC_DISPLAY" \
    -drive "if=virtio,format=qcow2,file=$BASE_IMAGE" \
    -drive "if=virtio,format=raw,readonly=on,file=$WORK_DIR/seed.iso" \
    -netdev "user,id=net0,hostfwd=tcp:127.0.0.1:$SSH_PORT-:22" \
    -device virtio-net-pci,netdev=net0 \
    -daemonize \
    -pidfile "$BUILD_PID_FILE" \
    -D "$WORK_DIR/qemu.log"
builder_pid="$(<"$BUILD_PID_FILE")"

SSH=(ssh -q -i "$SSH_KEY" -p "$SSH_PORT" -o BatchMode=yes \
    -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
    -o "UserKnownHostsFile=$KNOWN_HOSTS" agent-s@127.0.0.1)

deadline=$((SECONDS + 10800))
while true; do
    cloud_status="$("${SSH[@]}" cloud-init status --long 2>&1 || true)"
    if grep -q '^status: done' <<<"$cloud_status"; then
        break
    fi
    if grep -q '^status: error' <<<"$cloud_status"; then
        echo "Cloud-init failed:" >&2
        echo "$cloud_status" >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for cloud-init. Builder log: $WORK_DIR/qemu.log" >&2
        exit 1
    fi
    sleep 10
done

echo "Installing the pinned observer runtime..."
tar --exclude=.git --exclude=.pytest_cache --exclude='*/__pycache__' \
    -czf "$WORK_DIR/agent-s-source.tgz" -C "$REPO_ROOT" \
    README.md \
    setup.py \
    requirements-observer.lock \
    gui_agents/__init__.py \
    gui_agents/s3/__init__.py \
    gui_agents/s3/observer \
    scripts/agent_s_vm/guest
source_commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
source_dirty=false
if test -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal 2>/dev/null)"; then
    source_dirty=true
fi
source_archive_sha256="$(sha256sum "$WORK_DIR/agent-s-source.tgz" | awk '{print $1}')"
requirements_lock_sha256="$(sha256sum "$REPO_ROOT/requirements-observer.lock" | awk '{print $1}')"
built_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' \
    '{' \
    "  \"source_commit\": \"$source_commit\"," \
    "  \"source_dirty\": $source_dirty," \
    "  \"built_at\": \"$built_at\"," \
    "  \"source_archive_sha256\": \"$source_archive_sha256\"," \
    "  \"requirements_lock_sha256\": \"$requirements_lock_sha256\"" \
    '}' >"$WORK_DIR/observer-build.json"
scp -q -i "$SSH_KEY" -P "$SSH_PORT" -o StrictHostKeyChecking=accept-new \
    -o "UserKnownHostsFile=$KNOWN_HOSTS" \
    "$WORK_DIR/agent-s-source.tgz" "$WORK_DIR/observer-build.json" \
    agent-s@127.0.0.1:/tmp/

"${SSH[@]}" sudo bash -s <<'GUEST_PROVISION'
set -euo pipefail
install -d -m 0755 /opt/agent-s
tar -xzf /tmp/agent-s-source.tgz -C /opt/agent-s
install -m 0444 /tmp/observer-build.json /opt/agent-s/observer-build.json
python3 -m venv /opt/agent-s/venv
/opt/agent-s/venv/bin/pip install --disable-pip-version-check \
    --requirement /opt/agent-s/requirements-observer.lock
/opt/agent-s/venv/bin/pip install --disable-pip-version-check --no-deps \
    --no-build-isolation /opt/agent-s
install -d -m 0755 /opt/agent-s/bin
install -m 0755 /opt/agent-s/scripts/agent_s_vm/guest/agent-s-mcp-forced-command \
    /opt/agent-s/bin/agent-s-mcp-forced-command
install -m 0644 /opt/agent-s/scripts/agent_s_vm/guest/nftables.conf /etc/nftables.conf
install -m 0644 /opt/agent-s/scripts/agent_s_vm/guest/synthetic-page.html \
    /home/agent-s/agent-s-observer-test.html
chown agent-s:agent-s /home/agent-s/agent-s-observer-test.html
cat >/etc/ssh/sshd_config.d/agent-s-observer.conf <<'EOF'
AcceptEnv OPENAI_API_KEY HF_TOKEN HF_ENDPOINT_URL AGENT_S_MAIN_MODEL AGENT_S_GROUND_MODEL
PasswordAuthentication no
KbdInteractiveAuthentication no
EOF
pubkey="$(cat /home/agent-s/.ssh/authorized_keys)"
printf 'restrict,command="/opt/agent-s/bin/agent-s-mcp-forced-command" %s\n' "$pubkey" \
    >/home/agent-s/.ssh/authorized_keys
chmod 600 /home/agent-s/.ssh/authorized_keys
chown -R agent-s:agent-s /home/agent-s/.ssh
deluser agent-s sudo >/dev/null 2>&1 || true
rm -f /etc/sudoers.d/90-cloud-init-users
passwd -l agent-s >/dev/null
systemctl enable nftables
systemctl restart ssh
sync
shutdown -h +1
GUEST_PROVISION

deadline=$((SECONDS + 180))
while kill -0 "$builder_pid" 2>/dev/null; do
    if (( SECONDS >= deadline )); then
        echo "Builder VM did not power off cleanly." >&2
        exit 1
    fi
    sleep 3
done

rm -f "$BUILD_PID_FILE"
echo "Agent S base image is ready: $BASE_IMAGE"
