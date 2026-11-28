"""Tests for the done()/fail() detection used by the CLI run loops.

The CLI used to decide the fate of an action with `"done" in code.lower()`,
which ended the run (or skipped the action) whenever typed text contained
"done", "fail", "next" or "wait". Detection must compare exactly against the
strings the done()/fail() actions return.
"""

import unittest

from gui_agents.s3.agents.grounding import OSWorldACI as OSWorldACIS3
from gui_agents.s3.utils.common_utils import terminal_action as terminal_action_s3
from gui_agents.s2_5.utils.common_utils import (
    terminal_action as terminal_action_s2_5,
)


class TerminalActionMixin:
    terminal_action = None

    def test_done_and_fail_are_detected(self):
        self.assertEqual(self.terminal_action("DONE"), "DONE")
        self.assertEqual(self.terminal_action("FAIL"), "FAIL")

    def test_whitespace_and_case_are_tolerated(self):
        self.assertEqual(self.terminal_action("  done \n"), "DONE")
        self.assertEqual(self.terminal_action("Fail"), "FAIL")

    def test_typed_text_containing_the_words_is_not_terminal(self):
        for code in [
            "pyautogui.write('All done, thanks')",
            "pyautogui.write('Login failed, please retry')",
            "pyautogui.write('See you next week')",
            "pyautogui.write('Please wait for approval')",
            "pyautogui.write('Next.js docs')",
            "import time; time.sleep(3)",
        ]:
            self.assertIsNone(self.terminal_action(code), code)

    def test_non_strings_are_not_terminal(self):
        self.assertIsNone(self.terminal_action(None))
        self.assertIsNone(self.terminal_action(["DONE"]))


class TestTerminalActionS3(TerminalActionMixin, unittest.TestCase):
    terminal_action = staticmethod(terminal_action_s3)

    def test_matches_what_the_grounding_actions_return(self):
        aci = OSWorldACIS3.__new__(OSWorldACIS3)
        aci.platform = "linux"
        self.assertEqual(terminal_action_s3(aci.done()), "DONE")
        self.assertEqual(terminal_action_s3(aci.fail()), "FAIL")
        # Regression for simular-ai/Agent-S#235: typed text must be executed.
        self.assertIsNone(terminal_action_s3(aci.type(text="All done, thanks")))
        self.assertIsNone(terminal_action_s3(aci.open("Next.js docs")))
        self.assertIsNone(terminal_action_s3(aci.wait(3)))


class TestTerminalActionS2_5(TerminalActionMixin, unittest.TestCase):
    terminal_action = staticmethod(terminal_action_s2_5)


if __name__ == "__main__":
    unittest.main()
