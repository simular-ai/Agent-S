"""KVM/libvirt VM manager for OSWorld desktop_env."""

import logging
import os
import subprocess
from typing import List, Optional

from desktop_env.providers.base import VMManager  # type: ignore

logger = logging.getLogger("desktopenv.providers.kvm")

# A simple registry stored on disk: one libvirt domain name per line, optionally
# followed by a tab-separated PID of the process holding it.
DEFAULT_REGISTRY = os.path.expanduser("~/.osworld/kvm_registry.txt")


def _read_registry(path: str) -> List[List[str]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rows.append(line.split("\t"))
    return rows


def _write_registry(path: str, rows: List[List[str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for row in rows:
            f.write("\t".join(row) + "\n")
    os.replace(tmp, path)


class KVMVMManager(VMManager):
    def __init__(self, registry_path: str = DEFAULT_REGISTRY, virsh_uri: Optional[str] = None):
        self.registry_path = registry_path
        self.virsh_uri = virsh_uri
        self.checked_and_cleaned = False

    def initialize_registry(self, **kwargs) -> None:
        if not os.path.exists(self.registry_path):
            _write_registry(self.registry_path, [])

    def add_vm(self, vm_path: str, **kwargs) -> None:
        rows = _read_registry(self.registry_path)
        if not any(r[0] == vm_path for r in rows):
            rows.append([vm_path])
            _write_registry(self.registry_path, rows)

    def delete_vm(self, vm_path: str, **kwargs) -> None:
        rows = [r for r in _read_registry(self.registry_path) if r[0] != vm_path]
        _write_registry(self.registry_path, rows)

    def occupy_vm(self, vm_path: str, pid, **kwargs) -> None:
        rows = _read_registry(self.registry_path)
        for r in rows:
            if r[0] == vm_path:
                if len(r) == 1:
                    r.append(str(pid))
                else:
                    r[1] = str(pid)
        _write_registry(self.registry_path, rows)

    def list_free_vms(self, **kwargs) -> List[str]:
        return [r[0] for r in _read_registry(self.registry_path) if len(r) == 1 or not r[1]]

    def check_and_clean(self, **kwargs) -> None:
        # Drop registry entries whose holding pid is no longer alive.
        rows = _read_registry(self.registry_path)
        cleaned = []
        for r in rows:
            if len(r) >= 2 and r[1]:
                try:
                    os.kill(int(r[1]), 0)
                    cleaned.append(r)
                except (OSError, ValueError):
                    cleaned.append([r[0]])
            else:
                cleaned.append(r)
        _write_registry(self.registry_path, cleaned)
        self.checked_and_cleaned = True

    def get_vm_path(self, **kwargs) -> str:
        free = self.list_free_vms()
        if free:
            return free[0]
        raise RuntimeError(
            "No free KVM domain registered. Pre-create domains with virt-install "
            "and register them via add_vm(domain_name)."
        )
