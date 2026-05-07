"""Webtop (linuxserver/webtop family) provider for OSWorld desktop_env.

Runs a `ghcr.io/linuxserver/webtop` (or compatible) container with KasmVNC
on port 3001 and an HTTP control side-car on port 3000. Works with `docker`
or `podman` (set OSWORLD_WEBTOP_RUNTIME=podman).

Each `path_to_vm` is the container image reference (e.g.
`ghcr.io/dothyt/webtop:ubuntu-kde`).
"""

import logging
import os
import socket
import subprocess
import time
from typing import List, Optional

from filelock import FileLock  # type: ignore

from desktop_env.providers.base import Provider  # type: ignore

logger = logging.getLogger("desktopenv.providers.webtop")

VNC_PORT = 3001
SIDECAR_PORT = 3000
PORT_LOCK = os.path.expanduser("~/.osworld/webtop_ports.lock")


def _runtime() -> str:
    return os.environ.get("OSWORLD_WEBTOP_RUNTIME", "docker")


def _run(cmd: List[str], **kw) -> subprocess.CompletedProcess:
    logger.debug("$ %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _alloc_port(start: int) -> int:
    p = start
    while _port_in_use(p) and p < 65535:
        p += 1
    return p


class WebtopProvider(Provider):
    def __init__(self, region: Optional[str] = None):
        self.region = region
        self.runtime = _runtime()
        os.makedirs(os.path.dirname(PORT_LOCK), exist_ok=True)
        self._port_lock = FileLock(PORT_LOCK)
        self._containers: dict = {}  # image -> {name, vnc_port, sidecar_port}

    def _container_name(self, image: str, vnc_port: int) -> str:
        safe = image.replace("/", "-").replace(":", "_")
        return f"osworld-webtop-{safe}-{vnc_port}"

    def start_emulator(self, path_to_vm: str, headless: bool, *args, **kwargs) -> None:
        image = path_to_vm
        if image in self._containers:
            return
        with self._port_lock:
            vnc = _alloc_port(38000)
            side = _alloc_port(vnc + 1)
        name = self._container_name(image, vnc)
        cmd = [
            self.runtime, "run", "-d", "--rm",
            "--name", name,
            "--security-opt", "seccomp=unconfined",
            "--shm-size", "2g",
            "-e", "PUID=1000", "-e", "PGID=1000", "-e", "TZ=Etc/UTC",
            "-e", f"CUSTOM_USER={os.environ.get('WEBTOP_USER', 'admin')}",
            "-e", f"PASSWORD={os.environ.get('WEBTOP_PASSWORD', 'changeme')}",
            "-p", f"{vnc}:{VNC_PORT}",
            "-p", f"{side}:{SIDECAR_PORT}",
            image,
        ]
        proc = _run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"failed to start webtop {image}: {proc.stderr}")
        self._containers[image] = {"name": name, "vnc_port": vnc, "sidecar_port": side}
        # Wait for the KasmVNC HTTP endpoint to come up.
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if _port_in_use(vnc):
                return
            time.sleep(1)
        raise TimeoutError(f"webtop {name} did not expose port {vnc}")

    def get_ip_address(self, path_to_vm: str) -> str:
        info = self._containers.get(path_to_vm)
        if not info:
            raise RuntimeError(f"webtop {path_to_vm} not started")
        # Mirror DockerProvider's encoded form so downstream code that splits
        # the four ports keeps working: server, chromium, vnc, vlc.
        # Webtop has no separate chromium/vlc; we duplicate the sidecar port.
        s, c, v, l = info["sidecar_port"], info["sidecar_port"], info["vnc_port"], info["sidecar_port"]
        return f"localhost:{s}:{c}:{v}:{l}"

    def save_state(self, path_to_vm: str, snapshot_name: str) -> None:
        info = self._containers.get(path_to_vm)
        if not info:
            raise RuntimeError(f"webtop {path_to_vm} not started")
        proc = _run([self.runtime, "commit", info["name"], f"{snapshot_name}:latest"])
        if proc.returncode != 0:
            raise RuntimeError(f"webtop snapshot failed: {proc.stderr}")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str) -> str:
        # Stop current and re-launch from the snapshot image.
        self.stop_emulator(path_to_vm)
        self.start_emulator(f"{snapshot_name}:latest", headless=True)
        return f"{snapshot_name}:latest"

    def stop_emulator(self, path_to_vm: str, region: Optional[str] = None, *args, **kwargs) -> None:
        info = self._containers.pop(path_to_vm, None)
        if not info:
            return
        _run([self.runtime, "rm", "-f", info["name"]])
