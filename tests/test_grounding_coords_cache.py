import unittest
from unittest.mock import patch

from gui_agents.s3.agents.grounding import OSWorldACI
from gui_agents.s3.utils.common_utils import create_pyautogui_code

SCREENSHOT = b"\x89PNG\r\n\x1a\n first screen"
OTHER_SCREENSHOT = b"\x89PNG\r\n\x1a\n second screen"


def build_aci():
    """An ACI whose engines are never contacted: every model call is patched out."""
    params = {
        "engine_type": "ollama",
        "model": "llama3",
        "base_url": "http://localhost:11434/v1",
    }
    return OSWorldACI(
        env=None,
        platform="linux",
        engine_params_for_generation=params,
        engine_params_for_grounding={
            **params,
            "grounding_width": 1920,
            "grounding_height": 1080,
        },
    )


class TestGroundingCoordsCache(unittest.TestCase):
    def setUp(self):
        self.aci = build_aci()
        self.calls = []

        def fake_call_llm_safe(agent, **kwargs):
            self.calls.append(agent)
            return "Coordinates: (123, 456)"

        patcher = patch(
            "gui_agents.s3.agents.grounding.call_llm_safe",
            side_effect=fake_call_llm_safe,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_repeated_grounding_of_one_screen_calls_the_model_once(self):
        obs = {"screenshot": SCREENSHOT}
        self.aci.assign_screenshot(obs)

        first = self.aci.generate_coords("Submit button", obs)
        second = self.aci.generate_coords("Submit button", obs)

        self.assertEqual([123, 456], first)
        self.assertEqual(first, second)
        self.assertEqual(1, len(self.calls))

    def test_a_new_screen_is_grounded_again(self):
        self.aci.generate_coords("Submit button", {"screenshot": SCREENSHOT})
        self.aci.generate_coords("Submit button", {"screenshot": OTHER_SCREENSHOT})

        self.assertEqual(2, len(self.calls))

    def test_different_elements_on_one_screen_are_each_grounded(self):
        obs = {"screenshot": SCREENSHOT}

        self.aci.generate_coords("Submit button", obs)
        self.aci.generate_coords("Cancel button", obs)

        self.assertEqual(2, len(self.calls))

    def test_validation_and_execution_share_one_grounding_call(self):
        """CODE_VALID_FORMATTER grounds the action, then the worker runs it (#178)."""
        obs = {"screenshot": SCREENSHOT}
        code = "agent.click('Submit button')"

        create_pyautogui_code(self.aci, code, obs)
        create_pyautogui_code(self.aci, code, obs)

        self.assertEqual(1, len(self.calls))


if __name__ == "__main__":
    unittest.main()
