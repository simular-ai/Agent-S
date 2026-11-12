"""Key and button names must reach pyautogui as the exact strings the agent asked for.

hotkey(), hold_and_press() and highlight_text_span() return Python source that is later exec'd. Building that
source with f"'{key}'" breaks on keys like '\\' (VS Code's split editor, Google Docs' clear formatting) or "'":
the generated line is a SyntaxError and the action never runs. These tests exec the generated code against a
recording stand-in for pyautogui, so nothing is pressed.
"""

import sys
import types
import unittest
from unittest.mock import patch

from gui_agents.s3.agents.grounding import OSWorldACI

TRICKY_KEYS = ["\\", "'", '"', "a'); import os; ('"]


class TestKeyEscaping(unittest.TestCase):
    def setUp(self):
        self.calls = []
        fake = types.ModuleType("pyautogui")
        fake.hotkey = lambda *keys, **kw: self.calls.append(("hotkey", keys))
        fake.press = lambda keys, **kw: self.calls.append(("press", keys))
        fake.keyDown = lambda key: self.calls.append(("keyDown", key))
        fake.keyUp = lambda key: self.calls.append(("keyUp", key))
        fake.moveTo = lambda *a, **kw: None
        fake.dragTo = lambda *a, **kw: self.calls.append(("dragTo", kw.get("button")))
        fake.mouseUp = lambda *a, **kw: None
        self.patcher = patch.dict(sys.modules, {"pyautogui": fake})
        self.patcher.start()
        self.aci = OSWorldACI.__new__(
            OSWorldACI
        )  # the key actions use no instance state

    def tearDown(self):
        self.patcher.stop()

    def run_code(self, code):
        exec(code, {})
        return self.calls

    def test_hotkey_passes_keys_verbatim(self):
        for key in TRICKY_KEYS:
            with self.subTest(key=key):
                self.calls.clear()
                calls = self.run_code(self.aci.hotkey(["ctrl", key]))
                self.assertEqual(calls, [("hotkey", ("ctrl", key))])

    def test_hold_and_press_passes_keys_verbatim(self):
        for key in TRICKY_KEYS:
            with self.subTest(key=key):
                self.calls.clear()
                calls = self.run_code(self.aci.hold_and_press(["shift"], [key]))
                self.assertEqual(
                    calls,
                    [("keyDown", "shift"), ("press", [key]), ("keyUp", "shift")],
                )

    def test_highlight_text_span_passes_button_verbatim(self):
        self.aci.obs = None
        with patch.object(
            OSWorldACI, "generate_text_coords", side_effect=[[1, 2], [3, 4]]
        ):
            code = self.aci.highlight_text_span("a", "b", button="left'")
        self.assertIn(("dragTo", "left'"), self.run_code(code))


if __name__ == "__main__":
    unittest.main()
