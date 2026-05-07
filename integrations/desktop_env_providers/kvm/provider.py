"""KVM/libvirt provider for OSWorld desktop_env.

Drop this directory into `desktop_env/providers/kvm/` in your OSWorld checkout
and add the wiring shown in the package README's `__init__.py` snippet.

Assumes:
- libvirt + virsh installed on the host running the harness.
- Each `path_to_vm` is a libvirt domain name (not a disk path).
- Guest has qemu-guest-agent installed (used to read the IP).
- Snapshots use libvirt's internal snapshot mechanism.
"""

import logging
import subprocess
import time
from typing import Optional

from desktop_env.providers.base import Provider  # type: ignore

logger = logging.getLogger("desktopenv.providers.kvm")


def _virsh(*args: str, uri: Optional[str] = None, check: bool = True, timeout: int = 60) -> str:
    cmd = ["virsh"]
    if uri:
        cmd += ["-c", uri]
    cmd += list(args)
    logger.debug("virsh: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"virsh failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
    return proc.stdout


class KVMProvider(Provider):
    """libvirt-backed virtualization provider."""

    def __init__(self, region: Optional[str] = None):
        # `region` is interpreted as a libvirt connection URI (e.g. qemu:///system).
        self.uri = region or None

    def start_emulator(self, path_to_vm: str, headless: bool, *args, **kwargs) -> None:
        domain = path_to_vm
        state = _virsh("domstate", domain, uri=self.uri).strip()
        if state == "running":
            logger.info("KVM domain %s already running", domain)
            return
        _virsh("start", domain, uri=self.uri)
        # Wait for QGA to respond, which means the guest is alive enough to
        # answer interface-addresses queries.
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            try:
                _virsh("qemu-agent-command", domain, '{"execute":"guest-ping"}', uri=self.uri, timeout=5)
                return
            except Exception:
                time.sleep(2)
        raise TimeoutError(f"KVM domain {domain} did not become QGA-ready in 300s")

    def get_ip_address(self, path_to_vm: str) -> str:
        domain = path_to_vm
        # Prefer QGA so we don't depend on DHCP lease parsing.
        out = _virsh("domifaddr", domain, "--source", "agent", uri=self.uri)
        for line in out.splitlines():
            parts = line.split()
            # virsh domifaddr columns: Name MAC Protocol Address
            if len(parts) >= 4 and "/" in parts[-1] and not parts[-1].startswith("127."):
                return parts[-1].split("/")[0]
        raise RuntimeError(f"No non-loopback IP found for KVM domain {domain}")

    def save_state(self, path_to_vm: str, snapshot_name: str) -> None:
        domain = path_to_vm
        _virsh("snapshot-create-as", "--domain", domain, "--name", snapshot_name,
               "--atomic", uri=self.uri)

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str) -> str:
        domain = path_to_vm
        _virsh("snapshot-revert", "--domain", domain, "--snapshotname", snapshot_name,
               "--running", uri=self.uri)
        return path_to_vm

    def stop_emulator(self, path_to_vm: str, region: Optional[str] = None, *args, **kwargs) -> None:
        domain = path_to_vm
        try:
            _virsh("shutdown", domain, uri=self.uri)
        except Exception:
            pass
        # Force-off after a short grace period.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = _virsh("domstate", domain, uri=self.uri, check=False).strip()
            if state in ("shut off", ""):
                return
            time.sleep(1)
        _virsh("destroy", domain, uri=self.uri, check=False)
