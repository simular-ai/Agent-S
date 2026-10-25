"""Regression tests for safe grounded-action dispatch in Agent S3.

These tests pin the behaviour of ``create_pyautogui_code`` directly against a
fake grounding agent, with no LLM client, Config, or network involved. They
guard against arbitrary Python being executed while a model-produced grounded
action is validated/converted (the ``eval`` sink).

Covers exploit path: simular-agent-s-s3-model-action-eval
  sink: gui_agents/s3/utils/common_utils.py (create_pyautogui_code -> eval)

On the unpatched tree the security cases FAIL: ``eval`` runs the smuggled side
effect (and the bare-call / non-action / non-literal payloads evaluate without
raising). After the fix every malicious block raises ``ValueError`` and the
side effect never runs, while all legitimate grounded actions still dispatch.
"""

import unittest

from gui_agents.s3.utils import common_utils
from gui_agents.s3.utils.common_utils import create_pyautogui_code


def agent_action(func):
    func.is_agent_action = True
    return func


# Process-global marker that the malicious payloads try to flip as a side
# effect. ``create_pyautogui_code`` evaluates the model string in the module
# scope of common_utils, so the marker is exposed there for the eval to reach.
SIDE_EFFECTS = []
common_utils.SIDE_EFFECTS = SIDE_EFFECTS


class FakeGroundingAgent:
    """Minimal stand-in for the grounding ACI used by create_pyautogui_code."""

    def __init__(self):
        self.assigned = False

    def assign_screenshot(self, obs):
        self.assigned = True

    @agent_action
    def wait(self, time):
        return f"WAITED:{time}"

    @agent_action
    def click(self, element_description, num_clicks=1, button_type="left"):
        return f"CLICK:{element_description}:{num_clicks}:{button_type}"

    @agent_action
    def hotkey(self, keys):
        return f"HOTKEY:{keys}"

    # An ordinary (non-action) method must NOT be reachable via the dispatcher.
    def internal_helper(self):
        SIDE_EFFECTS.append("INTERNAL")
        return "INTERNAL"


class TestS3ActionDispatch(unittest.TestCase):
    def setUp(self):
        SIDE_EFFECTS.clear()
        self.agent = FakeGroundingAgent()
        self.obs = {"screenshot": b""}

    def _assert_blocked(self, payload):
        """The payload must raise and must not run any side effect."""
        with self.assertRaises(ValueError):
            create_pyautogui_code(self.agent, payload, self.obs)
        self.assertEqual(
            SIDE_EFFECTS, [], f"Side effect executed for payload: {payload!r}"
        )

    # ------------------------------------------------------------------
    # The exploit: a syntactically valid single-action block that smuggles
    # an arbitrary Python side effect via a tuple expression. Under eval()
    # the side effect runs and the block returns a valid agent action; the
    # fix must refuse the whole block before anything executes.
    # ------------------------------------------------------------------
    def test_tuple_side_effect_payload_is_blocked(self):
        payload = "(SIDE_EFFECTS.append('AGENT_EVAL_MARKER'), agent.wait(1.0))[1]"
        self._assert_blocked(payload)

    def test_builtin_import_payload_is_blocked(self):
        payload = (
            "(__import__('builtins').getattr"
            "(SIDE_EFFECTS, 'append')('IMPORTED'), agent.wait(1.0))[1]"
        )
        self._assert_blocked(payload)

    def test_non_action_method_is_blocked(self):
        # Real method on the agent, but not flagged @agent_action.
        self._assert_blocked("agent.internal_helper()")

    def test_non_literal_argument_is_blocked(self):
        # Args must be literals; a comprehension as an argument is rejected.
        self._assert_blocked("agent.wait([SIDE_EFFECTS.append('X') for _ in [1]])")

    def test_attribute_on_non_agent_is_blocked(self):
        self._assert_blocked("SIDE_EFFECTS.append('NOT_AGENT')")

    # ------------------------------------------------------------------
    # Backward compatibility: every legitimate grounded action still works.
    # ------------------------------------------------------------------
    def test_legit_wait_action_still_dispatches(self):
        result = create_pyautogui_code(self.agent, "agent.wait(1.0)", self.obs)
        self.assertEqual(result, "WAITED:1.0")
        self.assertTrue(self.agent.assigned)

    def test_legit_click_with_kwargs_still_dispatches(self):
        result = create_pyautogui_code(
            self.agent,
            "agent.click('the OK button', num_clicks=2, button_type='left')",
            self.obs,
        )
        self.assertEqual(result, "CLICK:the OK button:2:left")

    def test_legit_hotkey_list_literal_still_dispatches(self):
        result = create_pyautogui_code(
            self.agent, "agent.hotkey(['ctrl', 'c'])", self.obs
        )
        self.assertEqual(result, "HOTKEY:['ctrl', 'c']")


if __name__ == "__main__":
    unittest.main()
