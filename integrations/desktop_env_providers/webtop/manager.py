"""Webtop VM manager for OSWorld desktop_env.

Mostly stubs — webtop runs are ephemeral, like the upstream Docker provider.
`get_vm_path` returns the image reference the harness should boot.
"""

import os
from typing import List

from desktop_env.providers.base import VMManager  # type: ignore

DEFAULT_IMAGE = os.environ.get("OSWORLD_WEBTOP_IMAGE", "ghcr.io/dothyt/webtop:ubuntu-kde")


class WebtopVMManager(VMManager):
    def __init__(self):
        self.checked_and_cleaned = False

    def initialize_registry(self, **kwargs) -> None:
        pass

    def add_vm(self, vm_path: str, **kwargs) -> None:
        pass

    def delete_vm(self, vm_path: str, **kwargs) -> None:
        pass

    def occupy_vm(self, vm_path: str, pid, **kwargs) -> None:
        pass

    def list_free_vms(self, **kwargs) -> List[str]:
        return [DEFAULT_IMAGE]

    def check_and_clean(self, **kwargs) -> None:
        self.checked_and_cleaned = True

    def get_vm_path(self, **kwargs) -> str:
        return kwargs.get("image") or DEFAULT_IMAGE
