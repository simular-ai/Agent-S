"""Tests for the `open` grounding action.

`open` must distinguish a bare application/file name (which is typed into the
OS launcher's search box) from a filesystem path (which must be handed to the
platform's default opener, since launchers search by name and cannot resolve
a full path).
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gui_agents.s3.agents.grounding import OSWorldACI


def _aci(platform: str) -> OSWorldACI:
    aci = OSWorldACI.__new__(OSWorldACI)
    aci.platform = platform
    return aci


def _run(code: str, platform: str):
    """Executes generated action code with pyautogui/subprocess/os.startfile
    stubbed out and returns the stubs for inspection."""
    pyautogui = MagicMock()
    popen = MagicMock()
    startfile = MagicMock()
    namespace = {}
    with patch.dict("sys.modules", {"pyautogui": pyautogui}), patch(
        "subprocess.Popen", popen
    ), patch("time.sleep"):
        if platform == "windows":
            with patch("os.startfile", startfile, create=True):
                exec(code, namespace)
        else:
            exec(code, namespace)
    return SimpleNamespace(pyautogui=pyautogui, popen=popen, startfile=startfile)


class TestLooksLikePath(unittest.TestCase):
    def test_app_and_file_names_are_not_paths(self):
        for name in ["firefox", "LibreOffice Calc", "report.xlsx", "notes.txt"]:
            self.assertFalse(OSWorldACI._looks_like_path(name), name)

    def test_paths_are_detected(self):
        for path in [
            "/home/user/report.xlsx",
            "~/Documents/notes.txt",
            "Documents/notes.txt",
            "C:\\Users\\me\\budget.xlsx",
            "D:/data/budget.xlsx",
        ]:
            self.assertTrue(OSWorldACI._looks_like_path(path), path)


class TestOpenGeneratesValidCode(unittest.TestCase):
    def test_generated_code_compiles_on_every_platform(self):
        for platform in ["linux", "darwin", "windows"]:
            for arg in ["firefox", "/home/user/report.xlsx", "C:\\x\\y.xlsx"]:
                code = _aci(platform).open(arg)
                compile(code, "<action>", "exec")

    def test_quotes_in_path_are_escaped(self):
        code = _aci("linux").open('/home/user/it\'s "here".txt')
        compile(code, "<action>", "exec")
        self.assertIn(repr('/home/user/it\'s "here".txt'), code)


class TestOpenByName(unittest.TestCase):
    def test_name_is_typed_into_launcher(self):
        stubs = _run(_aci("linux").open("firefox"), "linux")
        stubs.pyautogui.hotkey.assert_any_call("win")
        stubs.pyautogui.write.assert_called_once_with("firefox")
        stubs.popen.assert_not_called()

    def test_darwin_uses_spotlight(self):
        stubs = _run(_aci("darwin").open("Safari"), "darwin")
        stubs.pyautogui.hotkey.assert_any_call("command", "space", interval=0.5)
        stubs.pyautogui.typewrite.assert_called_once_with("Safari")
        stubs.popen.assert_not_called()


class TestOpenByPath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        self.tmp.close()
        self.addCleanup(os.unlink, self.tmp.name)

    def test_linux_opens_existing_path_with_xdg_open(self):
        stubs = _run(_aci("linux").open(self.tmp.name), "linux")
        stubs.popen.assert_called_once_with(["xdg-open", self.tmp.name])
        stubs.pyautogui.write.assert_not_called()

    def test_darwin_opens_existing_path_with_open(self):
        stubs = _run(_aci("darwin").open(self.tmp.name), "darwin")
        stubs.popen.assert_called_once_with(["open", self.tmp.name])
        stubs.pyautogui.typewrite.assert_not_called()

    def test_windows_opens_existing_path_with_startfile(self):
        stubs = _run(_aci("windows").open(self.tmp.name), "windows")
        stubs.startfile.assert_called_once_with(self.tmp.name)
        stubs.pyautogui.write.assert_not_called()

    def test_tilde_is_expanded(self):
        home = os.path.expanduser("~")
        if not os.path.isdir(home):
            self.skipTest("no home directory")
        tmp = tempfile.NamedTemporaryFile(dir=home, suffix=".txt", delete=False)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        home_relative = "~/" + os.path.basename(tmp.name)
        stubs = _run(_aci("linux").open(home_relative), "linux")
        stubs.popen.assert_called_once_with(["xdg-open", tmp.name])

    def test_missing_path_falls_back_to_searching_basename(self):
        missing = "/definitely/not/here/report.xlsx"
        stubs = _run(_aci("linux").open(missing), "linux")
        stubs.popen.assert_not_called()
        stubs.pyautogui.write.assert_called_once_with("report.xlsx")

    def test_missing_windows_path_falls_back_to_basename(self):
        stubs = _run(_aci("windows").open("C:\\nope\\budget.xlsx"), "windows")
        stubs.startfile.assert_not_called()
        stubs.pyautogui.write.assert_called_once_with("budget.xlsx")


class TestOpenPromptDocstring(unittest.TestCase):
    def test_docstring_tells_the_model_paths_are_accepted(self):
        self.assertIn("full file path", OSWorldACI.open.__doc__)
        self.assertTrue(getattr(OSWorldACI.open, "is_agent_action", False))

    def test_helpers_are_not_exposed_as_agent_actions(self):
        for helper in [
            "_looks_like_path",
            "_search_and_launch_code",
            "_open_path_code",
        ]:
            self.assertFalse(
                getattr(getattr(OSWorldACI, helper), "is_agent_action", False), helper
            )


if __name__ == "__main__":
    unittest.main()
