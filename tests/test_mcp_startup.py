import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from mcp.server.mcpserver.exceptions import ToolError
from gui_agents.s3 import mcp_server as server


class MCPStartupTests(unittest.TestCase):
    def test_auth_error_never_spawns(self):
        with patch.object(
            server, "_http", side_effect=ToolError("unauthorized")
        ), patch.object(server.subprocess, "Popen") as spawn:
            with self.assertRaises(ToolError):
                server.ensure_server()
            spawn.assert_not_called()

    def test_remote_url_never_autostarts(self):
        with patch.object(
            server, "UI_URL", "https://remote.example/prefix"
        ), patch.object(
            server, "_http", side_effect=urllib.error.URLError("unreachable")
        ), patch.object(
            server.subprocess, "Popen"
        ) as spawn:
            with self.assertRaises(ToolError):
                server.ensure_server()
            spawn.assert_not_called()

    def test_timeout_reaps_child(self):
        with tempfile.TemporaryDirectory() as directory:
            process = Mock()
            with patch.object(server, "UI_URL", "http://127.0.0.1:8000"), patch.object(
                server, "_http", side_effect=urllib.error.URLError("unreachable")
            ), patch.object(
                server, "state_directory", return_value=Path(directory)
            ), patch.object(
                server.subprocess, "Popen", return_value=process
            ):
                with self.assertRaises(ToolError):
                    server.ensure_server(wait_seconds=0)
                process.terminate.assert_called_once()
                process.wait.assert_called_once()

    def test_saved_credentials_resolved_on_backend(self):
        with patch.object(server, "ensure_server"), patch.object(
            server, "_http", return_value={}
        ) as http:
            server.test_connection("grounding")
            http.assert_called_once_with(
                "POST", "/api/connection/test", {"aspect": "grounding", "api_key": ""}
            )
