import base64
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gui_agents.s3.core import codex
from gui_agents.s3.core.engine import LMMEngineCodex
from gui_agents.s3.core.messages import LMMMessage
from gui_agents.s3.core.mllm import LMMAgent


class TestCodexSDK(unittest.TestCase):
    def test_history_and_mixed_content_order(self):
        messages: list[LMMMessage] = [
            {"role": "system", "content": [{"type": "text", "text": "Be concise."}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Previous action"}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,YQ=="},
                    },
                    {"type": "text", "text": "Next action?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "Yg==",
                        },
                    },
                ],
            },
        ]
        instructions, inputs = codex.messages_to_run_input(messages)
        self.assertEqual(instructions, "Be concise.")
        self.assertEqual(
            inputs,
            [
                codex.TextInput("[assistant]"),
                codex.TextInput("Previous action"),
                codex.TextInput("[user]"),
                codex.ImageInput("data:image/png;base64,YQ=="),
                codex.TextInput("Next action?"),
                codex.ImageInput("data:image/jpeg;base64,Yg=="),
            ],
        )

    def test_sdk_engine_and_array_images(self):
        agent = LMMAgent(engine_params={"engine_type": "codex", "model": "test-model"})
        self.assertIsInstance(agent.engine, codex.LMMEngineCodex)
        self.assertIs(LMMEngineCodex, codex.LMMEngineCodex)
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        agent.add_message("Look", image_content=[image, b"second"], put_text_last=True)
        content = agent.messages[-1]["content"]
        self.assertEqual(
            [part["type"] for part in content], ["image_url", "image_url", "text"]
        )
        payload = content[0]["image_url"]["url"].split(",", 1)[1]
        self.assertTrue(base64.b64decode(payload).startswith(b"\x89PNG"))
        agent.replace_message_at(1, "Look again", image_content=image)
        self.assertEqual(agent.messages[1]["content"][1]["type"], "image_url")
        agent.reset()
        self.assertEqual(len(agent.messages), 1)

    def test_generation_normalizes_string_content_and_closes_client(self):
        with patch.object(codex, "Codex") as factory:
            client = factory.return_value.__enter__.return_value
            client.account.return_value.account = SimpleNamespace(
                root=SimpleNamespace(type="chatgpt")
            )
            client.thread_start.return_value.run.return_value = SimpleNamespace(
                final_response="answer"
            )
            text = codex.codex_generate([{"role": "user", "content": "hello"}], "model")
        self.assertEqual(text, "answer")
        self.assertEqual(client.thread_start.call_args.kwargs["model"], "model")
        self.assertTrue(client.thread_start.call_args.kwargs["ephemeral"])
        self.assertEqual(
            client.thread_start.return_value.run.call_args.args[0],
            [codex.TextInput("[user]"), codex.TextInput("hello")],
        )
        factory.return_value.__exit__.assert_called_once()

    def test_generation_failure_and_missing_output(self):
        with patch.object(codex, "Codex") as factory:
            client = factory.return_value.__enter__.return_value
            client.account.return_value.account = None
            with self.assertRaisesRegex(ValueError, "Sign in"):
                codex.codex_generate([{"role": "user", "content": "hello"}], "model")
            client.thread_start.assert_not_called()
            client.account.return_value.account = SimpleNamespace(
                root=SimpleNamespace(type="chatgpt")
            )
            client.thread_start.return_value.run.return_value = SimpleNamespace(
                final_response=None
            )
            with self.assertRaisesRegex(ValueError, "no planner text"):
                codex.codex_generate([{"role": "user", "content": "hello"}], "model")

    def test_timeout_closes_sdk_transport(self):
        with patch.object(codex, "Codex") as factory:
            client = factory.return_value.__enter__.return_value
            client.account.return_value.account = SimpleNamespace(
                root=SimpleNamespace(type="chatgpt")
            )
            closed = threading.Event()
            client.close.side_effect = closed.set

            def wait_for_close(*args, **kwargs):
                self.assertTrue(closed.wait(2))
                raise RuntimeError("transport closed")

            client.thread_start.return_value.run.side_effect = wait_for_close
            with self.assertRaisesRegex(TimeoutError, "timed out"):
                codex.codex_generate(
                    [{"role": "user", "content": "hello"}], "model", timeout=0.02
                )
            client.close.assert_called_once()

    def test_account_model_discovery_and_logout(self):
        with patch.object(codex, "Codex") as factory:
            client = factory.return_value.__enter__.return_value
            client.account.return_value.account = SimpleNamespace(
                root=SimpleNamespace(type="chatgpt", email="user@example.com")
            )
            client.models.return_value.data = [
                SimpleNamespace(id="one"),
                SimpleNamespace(id="two"),
            ]
            self.assertEqual(codex.codex_models(), ["one", "two"])
            self.assertTrue(codex.codex_status()["signed_in"])
            codex.codex_logout()
            client.logout.assert_called_once()


if __name__ == "__main__":
    unittest.main()
