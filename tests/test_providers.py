import os
import unittest
from unittest.mock import patch, MagicMock
from gui_agents.s3.core.mllm import LMMAgent
from gui_agents.s3.core.engine import LMMEngineOpenAI, LMMEngineAnthropic
from gui_agents.utils import (
    anthropic_supports_temperature,
    extract_anthropic_text,
    extract_anthropic_thinking,
)


def _text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _thinking_block(thinking):
    block = MagicMock()
    block.type = "thinking"
    block.thinking = thinking
    return block


def _anthropic_response(*blocks):
    response = MagicMock()
    response.content = list(blocks)
    return response


class TestProviders(unittest.TestCase):
    def setUp(self):
        # Clear env vars before each test
        if "OLLAMA_HOST" in os.environ:
            del os.environ["OLLAMA_HOST"]
        if "DEEPSEEK_API_KEY" in os.environ:
            del os.environ["DEEPSEEK_API_KEY"]
        if "QWEN_API_KEY" in os.environ:
            del os.environ["QWEN_API_KEY"]
        if "DEEPSEEK_ENDPOINT_URL" in os.environ:
            del os.environ["DEEPSEEK_ENDPOINT_URL"]
        if "QWEN_ENDPOINT_URL" in os.environ:
            del os.environ["QWEN_ENDPOINT_URL"]

    def test_ollama_missing_config(self):
        """Test that Ollama raises ValueError if no endpoint is provided"""
        with self.assertRaises(ValueError) as cm:
            LMMAgent(engine_params={"engine_type": "ollama", "model": "llama3"})
        self.assertIn("Ollama endpoint must be provided", str(cm.exception))

    def test_ollama_valid_config_param(self):
        """Test Ollama init with base_url param"""
        agent = LMMAgent(
            engine_params={
                "engine_type": "ollama",
                "model": "llama3",
                "base_url": "http://example.com/v1",
            }
        )
        self.assertIsInstance(agent.engine, LMMEngineOpenAI)
        self.assertEqual(agent.engine.base_url, "http://example.com/v1")

    def test_ollama_valid_config_env(self):
        """Test Ollama init with OLLAMA_HOST env var"""
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://env-host:11434"}):
            agent = LMMAgent(engine_params={"engine_type": "ollama", "model": "llama3"})
            self.assertIsInstance(agent.engine, LMMEngineOpenAI)
            # Check for /v1 addition
            self.assertEqual(agent.engine.base_url, "http://env-host:11434/v1")

    def test_deepseek_init(self):
        """Test DeepSeek initialization"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}):
            agent = LMMAgent(
                engine_params={"engine_type": "deepseek", "model": "deepseek-coder"}
            )
            self.assertIsInstance(agent.engine, LMMEngineOpenAI)
            # Default URL
            self.assertEqual(agent.engine.base_url, "https://api.deepseek.com/v1")
            # (Note: engine.py logic resolves default at generate() time or if client created,
            # but init just stores what's passed. Let's verify prompt generation to ensure it doesn't crash on init)

    def test_qwen_init(self):
        """Test Qwen initialization"""
        with patch.dict(os.environ, {"QWEN_API_KEY": "sk-qwen"}):
            agent = LMMAgent(engine_params={"engine_type": "qwen", "model": "qwen-max"})
            self.assertIsInstance(agent.engine, LMMEngineOpenAI)
            self.assertEqual(
                agent.engine.base_url,
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

    def test_anthropic_supports_temperature(self):
        """Test temperature support detection for Anthropic models."""
        self.assertTrue(
            anthropic_supports_temperature("claude-sonnet-4-20250514")
        )
        self.assertTrue(
            anthropic_supports_temperature("claude-sonnet-4-5-20250929")
        )
        self.assertFalse(anthropic_supports_temperature("claude-sonnet-5"))
        self.assertFalse(
            anthropic_supports_temperature("claude-sonnet-5-20260301")
        )
        self.assertFalse(anthropic_supports_temperature("claude-opus-4-7"))
        self.assertFalse(anthropic_supports_temperature("claude-opus-4-8"))

    def test_extract_anthropic_text_skips_thinking_blocks(self):
        """Adaptive-thinking responses (thinking block first) must return text only."""
        response = _anthropic_response(
            _thinking_block("let me reason about this"),
            _text_block("the actual answer"),
        )
        self.assertEqual(extract_anthropic_text(response), "the actual answer")
        self.assertEqual(
            extract_anthropic_thinking(response), "let me reason about this"
        )

    def test_extract_anthropic_text_plain(self):
        """Plain text-only responses still return the text."""
        response = _anthropic_response(_text_block("hello"))
        self.assertEqual(extract_anthropic_text(response), "hello")

    @patch("gui_agents.s3.core.engine.Anthropic")
    def test_anthropic_sonnet_5_omits_temperature(self, mock_anthropic):
        """Sonnet 5 requests must not include temperature."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _anthropic_response(
            _text_block("ok")
        )

        engine = LMMEngineAnthropic(model="claude-sonnet-5", api_key="test")
        messages = [
            {"content": [{"text": "system prompt"}]},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]
        result = engine.generate(messages, temperature=0.0)

        _, kwargs = mock_client.messages.create.call_args
        self.assertNotIn("temperature", kwargs)
        self.assertEqual(result, "ok")

    @patch("gui_agents.s3.core.engine.Anthropic")
    def test_anthropic_sonnet_5_adaptive_thinking_response(self, mock_anthropic):
        """Sonnet 5 responses with a leading thinking block must not crash."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _anthropic_response(
            _thinking_block("reasoning"),
            _text_block("final answer"),
        )

        engine = LMMEngineAnthropic(model="claude-sonnet-5", api_key="test")
        messages = [
            {"content": [{"text": "system prompt"}]},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]
        result = engine.generate(messages, temperature=0.0)
        self.assertEqual(result, "final answer")

    @patch("gui_agents.s3.core.engine.Anthropic")
    def test_anthropic_sonnet_4_includes_temperature(self, mock_anthropic):
        """Sonnet 4 requests should still include temperature."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _anthropic_response(
            _text_block("ok")
        )

        engine = LMMEngineAnthropic(model="claude-sonnet-4-20250514", api_key="test")
        messages = [
            {"content": [{"text": "system prompt"}]},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]
        engine.generate(messages, temperature=0.0)

        _, kwargs = mock_client.messages.create.call_args
        self.assertEqual(kwargs["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
