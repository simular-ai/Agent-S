from __future__ import annotations

import socket
import subprocess
import sys

import pytest


def test_external_socket_is_blocked_before_connect():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="external network disabled"):
            sock.connect(("203.0.113.1", 443))
    finally:
        sock.close()


def test_child_python_inherits_network_guard():
    result = subprocess.run([
        sys.executable, "-c",
        "import socket; socket.create_connection(('203.0.113.1', 443), 1)",
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "external network disabled" in result.stderr
