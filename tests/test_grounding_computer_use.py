import base64
import unittest
from unittest.mock import MagicMock, patch

from gui_agents.s3.agents.grounding import OSWorldACI


def make_aci(model="claude-opus-5", width=1728, height=1117):
    aci = OSWorldACI.__new__(OSWorldACI)
    aci.engine_params_for_grounding = {"engine_type": "anthropic", "model": model}
    aci.width = width
    aci.height = height
    aci._computer_use_client = None
    return aci


def fake_tool_use_response(x, y, action="left_click", stop_reason="tool_use"):
    block = MagicMock()
    block.type = "tool_use"
    block.input = {"action": action, "coordinate": [x, y]}
    response = MagicMock()
    response.content = [block]
    response.stop_reason = stop_reason
    return response


class TestComputerUseGrounding(unittest.TestCase):
    def test_generate_coords_parses_left_click_coordinate(self):
        aci = make_aci()
        fake_client = MagicMock()
        fake_client.beta.messages.create.return_value = fake_tool_use_response(512, 300)
        aci._computer_use_client = fake_client

        result = aci.generate_coords("botão Salvar", {"screenshot": b"fake-png-bytes"})

        self.assertEqual(result, [512, 300])

    def test_sends_computer_use_tool_with_forced_tool_choice(self):
        aci = make_aci(model="claude-opus-5", width=1728, height=1117)
        fake_client = MagicMock()
        fake_client.beta.messages.create.return_value = fake_tool_use_response(1, 2)
        aci._computer_use_client = fake_client

        aci.generate_coords("x", {"screenshot": b"fake-png-bytes"})

        kwargs = fake_client.beta.messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-opus-5")
        self.assertEqual(kwargs["thinking"], {"type": "disabled"})
        self.assertEqual(kwargs["tool_choice"], {"type": "tool", "name": "computer"})
        self.assertIn("computer-use-2025-11-24", kwargs["betas"])
        tool = kwargs["tools"][0]
        self.assertEqual(tool["type"], "computer_20251124")
        self.assertEqual(tool["display_width_px"], 1728)
        self.assertEqual(tool["display_height_px"], 1117)

    def test_instruction_text_comes_before_image_in_content(self):
        aci = make_aci()
        fake_client = MagicMock()
        fake_client.beta.messages.create.return_value = fake_tool_use_response(1, 2)
        aci._computer_use_client = fake_client

        aci.generate_coords("o botão X", {"screenshot": b"fake-png-bytes"})

        content = fake_client.beta.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("o botão X", content[0]["text"])
        self.assertEqual(content[1]["type"], "image")
        self.assertEqual(
            content[1]["source"]["data"], base64.b64encode(b"fake-png-bytes").decode("utf-8")
        )

    def test_raises_clear_error_when_no_left_click_action_returned(self):
        aci = make_aci()
        fake_client = MagicMock()
        block = MagicMock()
        block.type = "text"
        response = MagicMock()
        response.content = [block]
        response.stop_reason = "end_turn"
        fake_client.beta.messages.create.return_value = response
        aci._computer_use_client = fake_client

        with self.assertRaises(RuntimeError) as cm:
            aci.generate_coords("x", {"screenshot": b"fake-png-bytes"})
        self.assertIn("nenhuma ação left_click", str(cm.exception))

    def test_client_created_lazily_only_once(self):
        aci = make_aci()
        self.assertIsNone(aci._computer_use_client)
        fake_client = MagicMock()
        fake_client.beta.messages.create.return_value = fake_tool_use_response(1, 2)

        with patch(
            "gui_agents.s3.agents.grounding.Anthropic", return_value=fake_client
        ) as mock_ctor:
            aci.generate_coords("x", {"screenshot": b"fake-png-bytes"})
            aci.generate_coords("y", {"screenshot": b"fake-png-bytes"})
            mock_ctor.assert_called_once()

    def test_non_anthropic_engine_uses_original_text_regex_path(self):
        aci = OSWorldACI.__new__(OSWorldACI)
        aci.engine_params_for_grounding = {"engine_type": "huggingface", "model": "ui-tars"}
        aci.width = 1024
        aci.height = 768
        aci._computer_use_client = None
        aci.grounding_model = MagicMock()

        with patch(
            "gui_agents.s3.agents.grounding.call_llm_safe", return_value="click at 100 200"
        ):
            result = aci.generate_coords("x", {"screenshot": b"fake-png-bytes"})

        self.assertEqual(result, [100, 200])
        aci.grounding_model.reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()
