#!/bin/bash
# Bootstrap a fresh OSWorld VM (eval or scratch) for the shadow interface.
# Idempotent: the apt-install path is skipped once scrot is on PATH.
PASSWORD="${CLIENT_PASSWORD:-osworld-public-evaluation}"

if ! command -v scrot >/dev/null 2>&1; then
    echo "$PASSWORD" | sudo -S apt-get update -y
    echo "$PASSWORD" | sudo -S apt-get install -y scrot xdotool xclip curl wmctrl
fi

chmod -R +r /home/user/shadow_interface/
echo "Shadow interface ready"
