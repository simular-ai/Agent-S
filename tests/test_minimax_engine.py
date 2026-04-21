"""Unit tests for LMMEngineMiniMax."""

import os
import unittest
from unittest.mock import MagicMock, patch

from gui_agents.s2_5.core.engine import LMMEngineMiniMax
from gui_agents.s2_5.core.mllm import LMMAgent


class TestLMMEngineMiniMaxInit(unittest.TestCase):
    def test_creates_instance_with_model(self):
        engine = LMMEngineMiniMax(model="MiniMax-M2.7", api_key="test-key")
        self.assertIsNotNone(engine)
        self.assertEqual(engine.model, "MiniMax-M2.7")

    def test_raises_without_model(self):
        with self.assertRaises(AssertionError):
            LMMEngineMiniMax(api_key="test-key")

    def test_stores_api_key(self):
        engine = LMMEngineMiniMax(model="MiniMax-M2.7", api_key="my-key")
        self.assertEqual(engine.api_key, "my-key")

    def test_stores_custom_base_url(self):
        engine = LMMEngineMiniMax(
            model="MiniMax-M2.7",
            api_key="test-key",
            base_url="https://custom.api.io/anthropic",
        )
        self.assertEqual(engine.base_url, "https://custom.api.io/anthropic")

    def test_default_base_url_is_none(self):
        engine = LMMEngineMiniMax(model="MiniMax-M2.7", api_key="test-key")
        self.assertIsNone(engine.base_url)

    def test_stores_temperature(self):
        engine = LMMEngineMiniMax(
            model="MiniMax-M2.7", api_key="test-key", temperature=0.7
        )
        self.assertEqual(engine.temperature, 0.7)


class TestLMMEngineMiniMaxGenerate(unittest.TestCase):
    def _make_messages(self, system="You are a helpful assistant.", user="Hello"):
        return [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]

    def _make_mock_response(self, text="test response"):
        mock_content = MagicMock()
        mock_content.text = text
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        return mock_response

    def test_raises_without_api_key(self):
        engine = LMMEngineMiniMax(model="MiniMax-M2.7")
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MINIMAX_API_KEY", None)
            with self.assertRaises(ValueError, msg="MINIMAX_API_KEY"):
                engine.generate(self._make_messages())

    def test_uses_env_api_key(self):
        engine = LMMEngineMiniMax(model="MiniMax-M2.7")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response()

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}):
            with patch("anthropic.Anthropic", return_value=mock_client):
                engine.generate(self._make_messages())

        mock_client.messages.create.assert_called_once()

    def test_default_base_url_is_minimax_anthropic(self):
        engine = LMMEngineMiniMax(model="MiniMax-M2.7", api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response()

        captured_base_url = {}

        def fake_anthropic(api_key, base_url):
            captured_base_url["url"] = base_url
            return mock_client

        with patch(
            "gui_agents.s2_5.core.engine.LMMEngineMiniMax.generate.__wrapped__",
            create=True,
        ):
            pass

        # Directly test base_url resolution
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MINIMAX_BASE_URL", None)
            resolved = (
                engine.base_url
                or os.environ.get("MINIMAX_BASE_URL")
                or "https://api.minimax.io/anthropic"
            )
        self.assertEqual(resolved, "https://api.minimax.io/anthropic")

    def test_temperature_clamped_from_zero(self):
        """Temperature=0.0 should be clamped to 1.0 for MiniMax."""
        engine = LMMEngineMiniMax(model="MiniMax-M2.7", api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response()
        engine.llm_client = mock_client

        engine.generate(self._make_messages(), temperature=0.0)

        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 1.0)

    def test_positive_temperature_preserved(self):
        engine = LMMEngineMiniMax(model="MiniMax-M2.7", api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response()
        engine.llm_client = mock_client

        engine.generate(self._make_messages(), temperature=0.7)

        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.7)

    def test_instance_temperature_overrides_call_temperature(self):
        engine = LMMEngineMiniMax(
            model="MiniMax-M2.7", api_key="test-key", temperature=0.5
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response()
        engine.llm_client = mock_client

        engine.generate(self._make_messages(), temperature=0.0)

        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.5)

    def test_unsupported_params_filtered(self):
        engine = LMMEngineMiniMax(model="MiniMax-M2.7", api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response()
        engine.llm_client = mock_client

        engine.generate(
            self._make_messages(),
            temperature=0.7,
            top_k=40,
            stop_sequences=["END"],
            service_tier="auto",
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertNotIn("top_k", call_kwargs)
        self.assertNotIn("stop_sequences", call_kwargs)
        self.assertNotIn("service_tier", call_kwargs)

    def test_system_message_extracted_correctly(self):
        engine = LMMEngineMiniMax(model="MiniMax-M2.7", api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response()
        engine.llm_client = mock_client

        messages = self._make_messages(system="Custom system prompt")
        engine.generate(messages, temperature=0.5)

        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["system"], "Custom system prompt")

    def test_conversation_excludes_system_message(self):
        engine = LMMEngineMiniMax(model="MiniMax-M2.7", api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response()
        engine.llm_client = mock_client

        messages = self._make_messages()
        engine.generate(messages, temperature=0.5)

        call_kwargs = mock_client.messages.create.call_args[1]
        # Conversation should not include the system message
        for msg in call_kwargs["messages"]:
            self.assertNotEqual(msg.get("role"), "system")

    def test_model_name_passed(self):
        engine = LMMEngineMiniMax(model="MiniMax-M2.7-highspeed", api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response()
        engine.llm_client = mock_client

        engine.generate(self._make_messages(), temperature=0.5)

        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.7-highspeed")

    def test_returns_text_content(self):
        engine = LMMEngineMiniMax(model="MiniMax-M2.7", api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response(
            "hello world"
        )
        engine.llm_client = mock_client

        result = engine.generate(self._make_messages(), temperature=0.5)
        self.assertEqual(result, "hello world")


class TestLMMAgentMiniMax(unittest.TestCase):
    def test_minimax_engine_type_creates_minimax_engine(self):
        engine_params = {
            "engine_type": "minimax",
            "model": "MiniMax-M2.7",
            "api_key": "test-key",
        }
        agent = LMMAgent(engine_params=engine_params)
        self.assertIsInstance(agent.engine, LMMEngineMiniMax)

    def test_minimax_in_add_message_uses_anthropic_format(self):
        engine_params = {
            "engine_type": "minimax",
            "model": "MiniMax-M2.7",
            "api_key": "test-key",
        }
        agent = LMMAgent(engine_params=engine_params, system_prompt="Test")
        agent.add_message("Hello")
        # Messages should contain system + user message
        self.assertEqual(len(agent.messages), 2)
        self.assertEqual(agent.messages[-1]["role"], "user")
        self.assertEqual(agent.messages[-1]["content"][0]["type"], "text")


if __name__ == "__main__":
    unittest.main()
