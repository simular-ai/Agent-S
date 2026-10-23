import importlib
import inspect
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from anthropic import Anthropic, DefaultHttpxClient
from anthropic.resources.messages import Messages

from gui_agents.s3.core.engine import LMMEngineAnthropic
from gui_agents.s3.core.mllm import LMMAgent


class TestAnthropicProvider(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]
        self.response = SimpleNamespace(
            content=[
                SimpleNamespace(text="answer", thinking="thought"),
                SimpleNamespace(text="answer"),
            ]
        )

    def test_custom_base_url_reaches_sdk_in_all_generation_modes(self):
        for mode in ("generate", "thinking", "generate_with_thinking"):
            with self.subTest(mode=mode), patch(
                "gui_agents.s3.core.engine.Anthropic"
            ) as client_class:
                client_class.return_value.messages.create.return_value = self.response
                engine = LMMEngineAnthropic(
                    model="test-model",
                    api_key="test-key",
                    base_url="https://gateway.example/anthropic",
                    thinking=mode == "thinking",
                )
                method = "generate" if mode == "thinking" else mode

                result = getattr(engine, method)(self.messages)

                self.assertIn("answer", result)
                client_class.assert_called_once_with(
                    api_key="test-key", base_url="https://gateway.example/anthropic"
                )

    def test_lmm_agent_forwards_endpoint_with_and_without_thinking(self):
        for use_thinking in (False, True):
            with self.subTest(use_thinking=use_thinking), patch(
                "gui_agents.s3.core.engine.Anthropic"
            ) as client_class:
                client_class.return_value.messages.create.return_value = self.response
                agent = LMMAgent(
                    engine_params={
                        "engine_type": "anthropic",
                        "model": "test-model",
                        "api_key": "test-key",
                        "base_url": "https://gateway.example/anthropic",
                    }
                )
                agent.add_message("hello", role="user")

                result = agent.get_response(use_thinking=use_thinking)

                self.assertIn("answer", result)
                client_class.assert_called_once_with(
                    api_key="test-key", base_url="https://gateway.example/anthropic"
                )

    def test_default_and_environment_endpoints_keep_sdk_resolution(self):
        cases = (
            (None, None, "https://api.anthropic.com"),
            ("", None, "https://api.anthropic.com"),
            (None, "https://env.example", "https://env.example"),
            ("", "https://env.example", "https://env.example"),
            (
                "https://gateway.example",
                "https://env.example",
                "https://gateway.example",
            ),
        )
        for base_url, env_url, expected_url in cases:
            for mode in ("generate", "thinking", "generate_with_thinking"):
                with self.subTest(base_url=base_url, env_url=env_url, mode=mode):
                    clients = []

                    def make_client(**kwargs):
                        client = Anthropic(**kwargs)
                        client.messages.create = MagicMock(return_value=self.response)
                        clients.append(client)
                        return client

                    env = {"ANTHROPIC_API_KEY": "test-key"}
                    if env_url is not None:
                        env["ANTHROPIC_BASE_URL"] = env_url
                    try:
                        with patch.dict(os.environ, env, clear=True), patch(
                            "gui_agents.s3.core.engine.Anthropic",
                            side_effect=make_client,
                        ):
                            engine = LMMEngineAnthropic(
                                model="test-model",
                                base_url=base_url,
                                thinking=mode == "thinking",
                            )
                            method = "generate" if mode == "thinking" else mode
                            getattr(engine, method)(self.messages)
                            self.assertEqual(
                                str(clients[0].base_url).rstrip("/"), expected_url
                            )
                    finally:
                        for client in clients:
                            client.close()

    def assert_request_uses_custom_endpoint(self, mode):
        # The SDK's public HTTP client switched from httpx to httpx2 in 1.x.
        httpx = importlib.import_module(
            DefaultHttpxClient.__mro__[1].__module__.split(".")[0]
        )
        requests = []
        clients = []
        thinking = mode != "generate"

        def handle_request(request):
            requests.append((str(request.url), json.loads(request.content)))
            content = [{"type": "text", "text": "answer"}]
            if thinking:
                content.insert(
                    0, {"type": "thinking", "thinking": "thought", "signature": "test"}
                )
            return httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "test-model",
                    "content": content,
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        def make_client(**kwargs):
            client = Anthropic(
                **kwargs,
                http_client=DefaultHttpxClient(
                    transport=httpx.MockTransport(handle_request)
                ),
                max_retries=0,
            )
            clients.append(client)
            return client

        try:
            with patch.dict(
                os.environ, {"ANTHROPIC_BASE_URL": "https://env.example"}, clear=True
            ), patch("gui_agents.s3.core.engine.Anthropic", side_effect=make_client):
                engine = LMMEngineAnthropic(
                    model="test-model",
                    api_key="test-key",
                    base_url="https://gateway.example/anthropic",
                    thinking=mode == "thinking",
                )
                method = "generate" if mode == "thinking" else mode
                result = getattr(engine, method)(self.messages)
        finally:
            for client in clients:
                client.close()

        self.assertIn("answer", result)
        self.assertEqual(len(requests), 1)
        url, body = requests[0]
        self.assertEqual(url, "https://gateway.example/anthropic/v1/messages")
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["messages"], self.messages[1:])
        self.assertEqual(body["system"], "system")
        if thinking:
            self.assertEqual(
                body["thinking"], {"type": "enabled", "budget_tokens": 4096}
            )

    @unittest.skipUnless(
        "temperature" in inspect.signature(Messages.create).parameters,
        "This SDK removed temperature; the existing generate incompatibility is tracked in #205.",
    )
    def test_real_sdk_routes_plain_request_to_custom_endpoint(self):
        self.assert_request_uses_custom_endpoint("generate")

    def test_real_sdk_routes_thinking_requests_to_custom_endpoint(self):
        for mode in ("thinking", "generate_with_thinking"):
            with self.subTest(mode=mode):
                self.assert_request_uses_custom_endpoint(mode)


if __name__ == "__main__":
    unittest.main()
