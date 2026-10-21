"""Network guard shared by pytest collection and child Python processes."""
from __future__ import annotations

import ipaddress
import socket

_INSTALLED = False
_REAL_CONNECT = socket.socket.connect


def install_network_guard() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    def guarded_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            host = address[0]
            if host != "localhost":
                try:
                    is_loopback = ipaddress.ip_address(host).is_loopback
                except ValueError:
                    is_loopback = False
                if not is_loopback:
                    raise RuntimeError(
                        f"external network disabled during tests: {address!r}"
                    )
        return _REAL_CONNECT(sock, address)

    socket.socket.connect = guarded_connect
    _INSTALLED = True
