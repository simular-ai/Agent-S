import os
import unittest
from unittest.mock import patch, MagicMock
from gui_agents.s3.core.mllm import LMMAgent
from gui_agents.s3.core.engine import LMMEngineOpenAI


class TestProviders(unittest.TestCase):
    def setUp(self):
        # Clear env vars before each test
        if "OLLAMA_HOST" in os.environ:
            del os.environ["OLLAMA_HOST"]
        if "DEEPSEEK_API_KEY" in os.environ:
            del os.environ["DEEPSEEK_API_KEY"]
        if "QWEN_API_KEY" in os.environ:
            del os.environ["QWEN_API_KEY"]

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


if __name__ == "__main__":
    unittest.main()
